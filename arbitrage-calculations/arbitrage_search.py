import pandas as pd

# Load matched pairs and market data 
matches_df = pd.read_csv("high_similarity_matches_with_ids.csv")
kalshi_df = pd.read_csv("kalshi_markets.csv").set_index("ID")
polymarket_df = pd.read_csv("polymarket_markets.csv").set_index("ID")
 
opportunities = []

for _, row in matches_df.iterrows():
    kalshi_id = row["Kalshi ID"]
    poly_id = row["Polymarket ID"]

    try:
        k_yes = float(kalshi_df.loc[kalshi_id]["Option 1 Ask (¢)"]) / 100
        k_no = float(kalshi_df.loc[kalshi_id]["Option 2 Ask (¢)"]) / 100
        p_yes = float(polymarket_df.loc[poly_id]["Option 1 Ask (¢)"]) / 100
        p_no = float(polymarket_df.loc[poly_id]["Option 2 Ask (¢)"]) / 100
    except:
        continue  # skip if price data is missing or invalid

    # Arbitrage direction 1: Kalshi NO + Polymarket YES
    total1 = k_no + p_yes
    if total1 < 1:
        opportunities.append({
            "Direction": "Kalshi NO + Polymarket YES",
            "Kalshi ID": kalshi_id,
            "Kalshi Title": row["Kalshi Title"],
            "Kalshi Price": k_no,
            "Polymarket ID": poly_id,
            "Polymarket Title": row["Polymarket Title"],
            "Polymarket Price": p_yes,
            "Total Price": round(total1, 4)
        })

    # Arbitrage direction 2: Kalshi YES + Polymarket NO
    total2 = k_yes + p_no
    if total2 < 1:
        opportunities.append({
            "Direction": "Kalshi YES + Polymarket NO",
            "Kalshi ID": kalshi_id,
            "Kalshi Title": row["Kalshi Title"],
            "Kalshi Price": k_yes,
            "Polymarket ID": poly_id,
            "Polymarket Title": row["Polymarket Title"],
            "Polymarket Price": p_no,
            "Total Price": round(total2, 4)
        })

# Save results to CSV 
arb_df = pd.DataFrame(opportunities)
arb_df.to_csv("arbitrage_opportunities.csv", index=False)
print("Saved to arbitrage_opportunities.csv")



