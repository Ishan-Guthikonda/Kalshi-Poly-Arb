
"""
Fast cross-exchange contract matcher
Block -> Cached Embeddings (BGE) -> (Multiprocess) -> FAISS Top-K -> Mutual-Best 1:1 -> Safety checks
Outputs: high_similarity_matches_with_ids.csv

- Embedding cache keyed by (ID | normalized text | model)
- Multi-process encoding on CPU
- Exact FAISS (IndexFlatIP) for fast top-k cosine
"""

import os, re, math, pickle, hashlib, warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# CONFIG 
KALSHI_CSV = "kalshi_markets.csv"
POLY_CSV   = "polymarket_markets.csv"
OUTPUT_CSV = "high_similarity_matches_with_ids.csv"

ID_COL       = "ID"
TITLE_COL    = "Title"
SUBTITLE_COL = "Subtitle"   
CATEGORY_COL = "Category"   
EXPIRES_COL  = "Expires"    

# Model / retrieval
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
USE_MULTIPROCESS = True         # multi-core speedup on CPU
EMBED_BATCH = 64
TOP_K = 6

# Thresholds (precision-focused)
COSINE_THRESHOLD = 0.91
DATE_BLOCK_DAYS = 10

# Safety rules
NEGATION_TOKENS = {"not","except","excluding","unless","under","over","atleast","at-most","atmost","no"}
REQUIRE_DATE_IF_PRESENT = True
MIN_SHARED_TOKENS = 2


# --- Sorting logic abstraction ---
def _earliest_date(a, b):
    if pd.isna(a) and pd.isna(b):
        return pd.NaT
    if pd.isna(a):
        return b
    if pd.isna(b):
        return a
    return a if a <= b else b

def sort_matches_by_week_then_cosine(rows: list, k_df: pd.DataFrame, p_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of matches sorted by earliest resolution *week* (Monday-based),
    then by cosine (descending) within the same week.

    - Adds columns: 'Kalshi Expiry', 'Polymarket Expiry', 'Earliest Expiry'
    - 'Week' is computed as floor('W-MON') on 'Earliest Expiry'
    """
    if not rows:
        return pd.DataFrame(columns=[
            "Kalshi ID","Kalshi Title","Polymarket ID","Polymarket Title",
            "Cosine","Kalshi Expiry","Polymarket Expiry","Earliest Expiry"
        ])

    df = pd.DataFrame(rows)

    # Map ID -> expiry for quick lookup
    k_exp_map = k_df.set_index(ID_COL)["expires_parsed"]
    p_exp_map = p_df.set_index(ID_COL)["expires_parsed"]

    df["Kalshi Expiry"] = df["Kalshi ID"].map(k_exp_map)
    df["Polymarket Expiry"] = df["Polymarket ID"].map(p_exp_map)

    df["Earliest Expiry"] = [
        _earliest_date(k, p) for k, p in zip(df["Kalshi Expiry"], df["Polymarket Expiry"])
    ]

    # Week bucketing: same calendar week sorts together;
    # pandas handles tz-aware timestamps for floor('W-MON')
    df["Week"] = df["Earliest Expiry"].dt.floor("W-MON")

    # Sort by week ascending, then cosine descending
    df = df.sort_values(by=["Week", "Cosine"], ascending=[True, False])

    # Drop helper column 'Week' before writing to CSV
    df = df.drop(columns=["Week"])

    return df
# --- end sorting logic ---
# Infra
USE_FAISS = True
USE_HUNGARIAN = True
RANDOM_SEED = 42

# Cache
CACHE_DIR = ".embed_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
def _sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
CACHE_PATH = os.path.join(CACHE_DIR, f"cache_{_sanitize(EMBED_MODEL_NAME)}.pkl")
# =================

rng = np.random.default_rng(RANDOM_SEED)

# tiny utils 
STOPWORDS = {
    "the","a","an","of","in","on","at","to","for","by","with",
    "will","be","is","are","was","were","and","or","vs","v",
    "do","does","did","if","than","then","from","this","that"
}
NUM_PAT = re.compile(r"\d{4}|\d+(?:\.\d+)?")

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s:%\-\/]", "", s)  # keep digits, %, :, -, /
    return s.strip()

def content_tokens(s: str) -> set:
    return {t for t in s.split() if t not in STOPWORDS and len(t) > 2}

def token_overlap_ok(a_norm: str, b_norm: str, min_shared: int = MIN_SHARED_TOKENS) -> bool:
    return len(content_tokens(a_norm) & content_tokens(b_norm)) >= min_shared

def has_negation(s: str) -> bool:
    toks = set(normalize_text(s).split())
    return any(t in toks for t in NEGATION_TOKENS)

def extract_numbers(s: str) -> set:
    return set(NUM_PAT.findall(s))

def years_only(nums: set) -> set:
    return {n for n in nums if len(str(n)) == 4}

def numbers_compatible(a_norm: str, b_norm: str) -> bool:
    A, B = extract_numbers(a_norm), extract_numbers(b_norm)
    if not A and not B:
        return True
    Ya, Yb = years_only(A), years_only(B)
    if Ya or Yb:
        return len(Ya & Yb) > 0
    return True

def parse_date(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", utc=True)
    except Exception:
        return pd.to_datetime(pd.Series([pd.NaT]*len(series)), errors="coerce", utc=True)

def augment_title(row: pd.Series) -> str:
    bits = [str(row.get(TITLE_COL, ""))]
    if SUBTITLE_COL in row and pd.notna(row[SUBTITLE_COL]):
        bits.append(str(row[SUBTITLE_COL]))
    cat = str(row.get(CATEGORY_COL, "")).strip().lower()
    if cat and cat != "nan":
        bits.append(f"[cat:{cat}]")
    exp = row.get("expires_parsed", pd.NaT)
    if pd.notna(exp):
        bits.append(f"[date:{exp.date().isoformat()}]")
    return " ".join(bits)

# data 
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ID_COL not in df.columns or TITLE_COL not in df.columns:
        raise ValueError(f"{path} must contain columns: {ID_COL}, {TITLE_COL}")
    if EXPIRES_COL in df.columns:
        df["expires_parsed"] = parse_date(df[EXPIRES_COL])
    else:
        df["expires_parsed"] = pd.NaT
    if CATEGORY_COL not in df.columns:
        df[CATEGORY_COL] = np.nan
    if SUBTITLE_COL not in df.columns:
        df[SUBTITLE_COL] = np.nan

    df["neg_flag"] = df[TITLE_COL].fillna("").map(has_negation)
    df["aug_text"] = df.apply(augment_title, axis=1)
    df["aug_text_norm"] = df["aug_text"].map(normalize_text)
    return df

def block_indices(k_df: pd.DataFrame, p_df: pd.DataFrame) -> Dict[int, List[int]]:
    p_by_cat = {}
    for cat, subdf in p_df.groupby(p_df[CATEGORY_COL].fillna("unknown")):
        p_by_cat[cat] = subdf.index.to_numpy()

    cand: Dict[int, List[int]] = {}
    for i, krow in k_df.iterrows():
        cat = krow[CATEGORY_COL] if pd.notna(krow[CATEGORY_COL]) else "unknown"
        base = p_by_cat.get(cat, p_df.index.to_numpy())
        if pd.notna(krow["expires_parsed"]):
            t = krow["expires_parsed"]
            lo, hi = t - pd.Timedelta(days=DATE_BLOCK_DAYS), t + pd.Timedelta(days=DATE_BLOCK_DAYS)
            mask = (p_df.loc[base, "expires_parsed"].isna()) | (
                (p_df.loc[base, "expires_parsed"] >= lo) & (p_df.loc[base, "expires_parsed"] <= hi)
            )
            idxs = base[mask.to_numpy()]
        else:
            idxs = base
        if len(idxs) == 0 and len(base) > 0:
            idxs = base
        cand[i] = idxs.tolist()
    return cand

#  cache
def _key_for(id_val: str, text_norm: str, model_name: str) -> str:
    raw = f"{id_val}|{text_norm}|{model_name}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def load_cache(path: str) -> Dict[str, np.ndarray]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            try:
                cache = pickle.load(f)
                if isinstance(cache, dict):
                    return cache
            except Exception:
                pass
    return {}

def save_cache(path: str, cache: Dict[str, np.ndarray]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

# embeddings (with cache + multiprocess)
def embed_texts_with_cache(ids: List[str], texts_norm: List[str], model_name: str,
                           batch: int = 64, normalize: bool = True,
                           multiprocess: bool = True, cache_path: str = CACHE_PATH) -> np.ndarray:
    from sentence_transformers import SentenceTransformer, util
    cache = load_cache(cache_path)

    keys = [_key_for(i, t, model_name) for i, t in zip(ids, texts_norm)]
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]

    if missing_idx:
        model = SentenceTransformer(model_name)
        to_encode = [texts_norm[i] for i in missing_idx]

        if multiprocess:
            # Multi-process CPU encoding
            enc = model.encode(
                to_encode,
                batch_size=batch,
                normalize_embeddings=normalize,
                show_progress_bar=True,
                num_proc=os.cpu_count()  # use all CPU cores
            )

        else:
            enc = model.encode(to_encode, batch_size=batch, show_progress_bar=True, normalize_embeddings=normalize)

        for idx, vec in zip(missing_idx, enc):
            cache[keys[idx]] = vec.astype(np.float32)

        save_cache(cache_path, cache)

    # Rebuild embeddings array in input order
    dim = len(next(iter(cache.values()))) if cache else 1024
    out = np.zeros((len(texts_norm), dim), dtype=np.float32)
    for i, k in enumerate(keys):
        out[i] = cache[k]
    return out

# FAISS / retrieval 
def try_import_faiss():
    if not USE_FAISS: return None
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None

def ann_topk(query_vecs: np.ndarray, base_vecs: np.ndarray, k: int,
             candidate_map: Optional[Dict[int, List[int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    faiss = try_import_faiss()
    Q, D = query_vecs.shape
    P = base_vecs.shape[0]

    if candidate_map is None and faiss is not None:
        index = faiss.IndexFlatIP(D)  # inner product == cosine (normalized)
        index.add(base_vecs.astype(np.float32))
        sims, idxs = index.search(query_vecs.astype(np.float32), k)
        return idxs, sims

    # filtered brute force
    all_idxs = np.full((Q, k), -1, dtype=int)
    all_sims = np.full((Q, k), -1.0, dtype=float)
    for i in range(Q):
        cand = candidate_map[i] if candidate_map is not None else list(range(P))
        if not cand:
            continue
        sub = base_vecs[cand]   # [C, D]
        sims = (query_vecs[i:i+1] @ sub.T).ravel()
        topk = np.argsort(-sims)[:k]
        chosen = [cand[t] for t in topk]
        all_idxs[i, :len(chosen)] = np.array(chosen, dtype=int)
        all_sims[i, :len(chosen)] = sims[topk]
    return all_idxs, all_sims

# 1:1 assignment
def mutual_best_one_to_one(pairs: List[Tuple[int, int, float]],
                           q_size: int, d_size: int, use_hungarian: bool) -> List[Tuple[int, int, float]]:
    if not pairs:
        return []
    if use_hungarian:
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            q_idxs = sorted({i for i, _, _ in pairs})
            d_idxs = sorted({j for _, j, _ in pairs})
            qi = {q: idx for idx, q in enumerate(q_idxs)}
            dj = {d: idx for idx, d in enumerate(d_idxs)}
            M = np.full((len(q_idxs), len(d_idxs)), 1e6, dtype=float)
            for i, j, s in pairs:
                M[qi[i], dj[j]] = -s
            r, c = linear_sum_assignment(M)
            out = []
            for rr, cc in zip(r, c):
                if M[rr, cc] <= 0:
                    out.append((q_idxs[rr], d_idxs[cc], -M[rr, cc]))
            return out
        except Exception:
            pass

    # greedy mutual-best fallback
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    best_d_for_q: Dict[int, Tuple[int, float]] = {}
    best_q_for_d: Dict[int, Tuple[int, float]] = {}
    for i, j, s in pairs_sorted:
        if i not in best_d_for_q or s > best_d_for_q[i][1]:
            best_d_for_q[i] = (j, s)
        if j not in best_q_for_d or s > best_q_for_d[j][1]:
            best_q_for_d[j] = (i, s)

    used_q, used_d, out = set(), set(), []
    for i, j, s in pairs_sorted:
        if i in used_q or j in used_d:
            continue
        if best_d_for_q.get(i, (-1, -math.inf))[0] == j and best_q_for_d.get(j, (-1, -math.inf))[0] == i:
            out.append((i, j, s))
            used_q.add(i); used_d.add(j)
    return out

# ---------- safety ----------
def safety_filter(i: int, j: int, cos: float, k_df: pd.DataFrame, p_df: pd.DataFrame) -> bool:
    # Dates
    if REQUIRE_DATE_IF_PRESENT:
        kd = k_df.loc[i, "expires_parsed"]
        pd_ = p_df.loc[j, "expires_parsed"]
        if pd.notna(kd) and pd.notna(pd_):
            if abs((kd - pd_).days) > DATE_BLOCK_DAYS:
                return False
    # Negation
    if k_df.loc[i, "neg_flag"] != p_df.loc[j, "neg_flag"]:
        if cos < (COSINE_THRESHOLD + 0.05):
            return False
    # Token overlap
    if not token_overlap_ok(k_df.loc[i, "aug_text_norm"], p_df.loc[j, "aug_text_norm"]):
        return False
    # Numbers/years
    if not numbers_compatible(k_df.loc[i, "aug_text_norm"], p_df.loc[j, "aug_text_norm"]):
        return False
    return True

# ---------- main ----------
def main():
    print("Loading data…")
    k_df = load_data(KALSHI_CSV)
    p_df = load_data(POLY_CSV)
    print(f"Kalshi: {len(k_df)} | Polymarket: {len(p_df)}")

    print("Blocking by category + ±date…")
    cand_map = block_indices(k_df, p_df)

    # Embeddings (cached + multiprocess)
    print(f"Embedding (cached) with {EMBED_MODEL_NAME} …")
    k_ids = k_df[ID_COL].astype(str).tolist()
    p_ids = p_df[ID_COL].astype(str).tolist()
    k_txt = k_df["aug_text_norm"].tolist()
    p_txt = p_df["aug_text_norm"].tolist()

    k_emb = embed_texts_with_cache(k_ids, k_txt, EMBED_MODEL_NAME,
                                   batch=EMBED_BATCH, normalize=True,
                                   multiprocess=USE_MULTIPROCESS, cache_path=CACHE_PATH)
    p_emb = embed_texts_with_cache(p_ids, p_txt, EMBED_MODEL_NAME,
                                   batch=EMBED_BATCH, normalize=True,
                                   multiprocess=USE_MULTIPROCESS, cache_path=CACHE_PATH)

    print("🔎 Retrieving top-k candidates…")
    idxs, sims = ann_topk(k_emb, p_emb, TOP_K, candidate_map=cand_map)

    # Build candidate pairs
    cand_pairs: List[Tuple[int, int, float]] = []
    for i in range(idxs.shape[0]):
        for pos in range(TOP_K):
            j = int(idxs[i, pos])
            if j < 0: continue
            s = float(sims[i, pos])
            if s >= COSINE_THRESHOLD and safety_filter(i, j, s, k_df, p_df):
                cand_pairs.append((i, j, s))

    print(f"Candidates ≥ {COSINE_THRESHOLD}: {len(cand_pairs)}")
    if not cand_pairs:
        pd.DataFrame(columns=["Kalshi ID","Kalshi Title","Polymarket ID","Polymarket Title","Cosine"]).to_csv(OUTPUT_CSV, index=False)
        print("No candidates. Consider relaxing COSINE_THRESHOLD a bit.")
        return

    print("🔗 Enforcing 1:1 (mutual-best / Hungarian)…")
    matched = mutual_best_one_to_one(cand_pairs, len(k_df), len(p_df), use_hungarian=USE_HUNGARIAN)
    print(f"🎯 Final matches: {len(matched)}")

    rows = []
    for i, j, cos in matched:
        rows.append({
            "Kalshi ID": k_df.loc[i, ID_COL],
            "Kalshi Title": k_df.loc[i, TITLE_COL],
            "Polymarket ID": p_df.loc[j, ID_COL],
            "Polymarket Title": p_df.loc[j, TITLE_COL],
            "Cosine": round(float(cos), 5)
        })

    out_df = sort_matches_by_week_then_cosine(rows, k_df, p_df)
    #out_df = pd.DataFrame(rows).sort_values("Cosine", ascending=False)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out_df)} matches -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()