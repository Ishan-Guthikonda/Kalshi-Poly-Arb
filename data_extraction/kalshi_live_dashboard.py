import requests
import csv
import time

# Config
base_url = "https://api.elections.kalshi.com/trade-api/v2/markets"
limit = 1000
output_path = "kalshi_markets.csv"

fieldnames = [
    "ID", "Title", "Yes Event Title",
    "Option 1", "Option 2",
    "Option 1 Ask (¢)", "Option 2 Ask (¢)",
    "Status", "Expires"
]

start_time = time.time()
cursor = None
total_rows = 0
page_num = 0
MAX_PAGES = 200

session = requests.Session()
rows = []

while True:
    url = f"{base_url}?limit={limit}&status=open"
    if cursor:
        url += f"&cursor={cursor}"

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        break

    markets = data.get("markets", [])
    if not markets:
        print("No markets found on this page. Stopping.")
        break

    for m in markets:
        title = m.get("title", "")
        subtitle = m.get("subtitle", "")
        yes_sub = m.get("yes_sub_title", "")
        yes_label = m.get("yes_label", "")

        # Construct full title with optional subtitle and yes_sub_title
        if subtitle and yes_sub:
            full_title = f"{title} ({subtitle}) [{yes_sub}]"
        elif subtitle:
            full_title = f"{title} ({subtitle})"
        elif yes_sub:
            full_title = f"{title} [{yes_sub}]"
        else:
            full_title = title

        rows.append({
            "ID": m.get("ticker", ""),
            "Title": full_title,
            "Yes Event Title": yes_label,
            "Option 1": "Yes",
            "Option 2": "No",
            "Option 1 Ask (¢)": round(m.get("yes_ask") or 0, 2),
            "Option 2 Ask (¢)": round(m.get("no_ask") or 0, 2),
            "Status": m.get("status", ""),
            "Expires": m.get("expected_expiration_time", "")
        })
        total_rows += 1

    page_num += 1
    cursor = data.get("cursor")
    if not cursor:
        print("No more pages.")
        break
    if page_num >= MAX_PAGES:
        print("Max page limit reached.")
        break

# Write everything to CSV once
with open(output_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

end_time = time.time()
print(f"\n✅ Finished. Total open markets saved: {total_rows}")
print(f"📁 Saved to: {output_path}")
print(f"⏱️ Elapsed time: {round(end_time - start_time, 2)} seconds")
