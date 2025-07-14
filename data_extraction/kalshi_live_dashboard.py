import requests
import csv
import time

# === Config ===
base_url = "https://api.elections.kalshi.com/trade-api/v2/markets"
limit = 1000
url = f"{base_url}?limit={limit}"
output_path = "kalshi_markets.csv"

fieldnames = [
    "ID", "Title",
    "Option 1", "Option 2",
    "Option 1 Ask (¢)", "Option 2 Ask (¢)",
    "Status", "Expires"
]

seen_cursors = set()
seen_first_titles = set()
MAX_PAGES = 200  # Safety limit
total_rows = 0
page_num = 0
start_time = time.time()

with open(output_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    while url:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except KeyboardInterrupt:
            print("❌ Interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Request failed: {e}")
            break

        markets = data.get("markets", [])
        count_this_page = len(markets)
        page_num += 1

        if count_this_page == 0:
            print("⚠️ Empty page. Exiting.")
            break

        first_title = markets[0].get("title", "None")
        if first_title in seen_first_titles:
            break
        seen_first_titles.add(first_title)

        saved_this_page = 0
        for m in markets:
            if m.get("status") == "active":
                writer.writerow({
                    "ID": m.get("ticker", ""),
                    "Title": m.get("title", ""),
                    "Option 1": "Yes",
                    "Option 2": "No",
                    "Option 1 Ask (¢)": round(m.get("yes_ask") or 0, 2),
                    "Option 2 Ask (¢)": round(m.get("no_ask") or 0, 2),
                    "Status": m.get("status", ""),
                    "Expires": m.get("expected_expiration_time", "")
                })
                total_rows += 1
                saved_this_page += 1

        cursor = data.get("cursor")
        if not cursor or cursor in seen_cursors:
            print("Done. No more pages.")
            break

        if page_num >= MAX_PAGES:
            print("Max page limit reached. Stopping.")
            break

        seen_cursors.add(cursor)
        url = f"{base_url}?limit={limit}&cursor={cursor}"

end_time = time.time()
print(f"\nFinished. Total active markets saved: {total_rows}")
print(f"Saved to: {output_path}")



