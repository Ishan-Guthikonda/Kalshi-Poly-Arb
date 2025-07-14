import requests
import csv
import time
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Config ===
base_url = "https://gamma-api.polymarket.com/markets"
limit = 500
max_offset = 60000
threads = 6
output_path = "polymarket_markets.csv"
batch_size = 500

# Fetch Pages
def fetch_page(offset):
    try:
        response = requests.get(
            base_url,
            params={"limit": limit, "offset": offset, "active": "true"},
            timeout=(5, 10)
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else data.get("markets", [])
    except Exception as e:
        print(f"❌ Failed at offset {offset}: {e}")
        return []

# Get Sell Prices Using Api Call 
def get_sell_prices(token_ids):
    PRICE_URL = "https://clob.polymarket.com/prices"
    prices = {}
    session = requests.Session()

    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i:i + batch_size]
        payload = [
            {"token_id": str(token_id), "side": "SELL"}
            for token_id in batch
        ]
        try:
            response = session.post(PRICE_URL, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            for tid in batch:
                tid_str = str(tid)
                value = data.get(tid_str)
                if isinstance(value, dict) and "SELL" in value:
                    prices[tid_str] = round(float(value["SELL"]) * 100, 2)
                else:
                    prices[tid_str] = "N/A"
        except Exception as e:
            print(f"❌ Price batch failed: {e}")
            for token_id in batch:
                prices[str(token_id)] = "N/A"

    return prices

# Flatten Market
def flatten_market(market, price_map):
    if not market.get("active", False) or market.get("closed", True):
        return None

    market_id = market.get("id", "")
    title = market.get("question", "").strip()
    status = "active"
    end_time = market.get("endDate", "")

    outcomes = market.get("outcomes", [])
    tokens = market.get("clobTokenIds", [])

    if isinstance(outcomes, str):
        try:
            outcomes = ast.literal_eval(outcomes)
        except Exception:
            outcomes = []

    if isinstance(tokens, str):
        try:
            tokens = ast.literal_eval(tokens)
        except Exception:
            tokens = []

    row = {
        "ID": market_id,
        "Title": title,
        "Status": status,
        "Expires": end_time
    }

    for i, outcome in enumerate(outcomes):
        row[f"Option {i+1}"] = outcome
        token_id = str(tokens[i]) if i < len(tokens) else None
        row[f"Option {i+1} Ask (¢)"] = price_map.get(token_id, "N/A")

    return row

# Main
start_time = time.time()
rows = []
all_markets = []

try:
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(fetch_page, offset) for offset in range(0, max_offset, limit)]
        for future in as_completed(futures):
            markets = future.result()
            if isinstance(markets, list):
                all_markets.extend(markets)
except KeyboardInterrupt:
    print("Interrupted by user. Writing partial results...")

print(f"Total markets fetched: {len(all_markets)}")

# Extract all unique token IDs 
all_token_ids = set()
valid_markets = []
for market in all_markets:
    if not market.get("active", False) or market.get("closed", True):
        continue
    try:
        tokens = market.get("clobTokenIds", [])
        if isinstance(tokens, str):
            tokens = ast.literal_eval(tokens)
        for token in tokens:
            all_token_ids.add(str(token))
        valid_markets.append(market)
    except Exception:
        continue

print(f"🔍 Found {len(all_token_ids)} unique token IDs")
price_map = get_sell_prices(list(all_token_ids))

# Flatten rows and determine max outcomes
max_outcomes = 0
for market in valid_markets:
    row = flatten_market(market, price_map)
    if row:
        rows.append(row)
        i = 1
        while f"Option {i}" in row:
            i += 1
        max_outcomes = max(max_outcomes, i - 1)

# Write CSV
fieldnames = ["ID", "Title"]
for i in range(1, max_outcomes + 1):
    fieldnames.append(f"Option {i}")
for i in range(1, max_outcomes + 1):
    fieldnames.append(f"Option {i} Ask (¢)")
fieldnames += ["Status", "Expires"]

with open(output_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} active markets to {output_path}")
