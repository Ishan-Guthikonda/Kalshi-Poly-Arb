import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- Load model (BAAI bge-small) ---
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# --- Load all titles ---
kalshi_data = pd.read_csv("kalshi_markets.csv")
polymarket_data = pd.read_csv("polymarket_markets.csv")

kalshi_titles = kalshi_data["Title"].dropna().tolist()
polymarket_titles = polymarket_data["Title"].dropna().tolist()

print(f"Loaded {len(kalshi_titles)} Kalshi titles and {len(polymarket_titles)} Polymarket titles.")

# --- Embed all titles (batching handled internally) ---
print("Embedding Kalshi titles...")
kalshi_embeddings = model.encode(kalshi_titles, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

print("Embedding Polymarket titles...")
polymarket_embeddings = model.encode(polymarket_titles, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

# --- Compute similarity matrix (n x m) ---
print("Computing similarity matrix...")
similarity_matrix = cosine_similarity(kalshi_embeddings, polymarket_embeddings)

# --- Find best matches ---
top_indices = np.argmax(similarity_matrix, axis=1)
top_scores = np.max(similarity_matrix, axis=1)

# --- Show matches above threshold ---
threshold = 0.92
print("\nTop Matches:")
for i, (score, j) in enumerate(zip(top_scores, top_indices)):
    if score >= threshold:
        print(f"With score={score:.3f} match found between:")
        print(f"Kalshi:     {kalshi_titles[i]}")
        print(f"Polymarket: {polymarket_titles[j]}\n")

