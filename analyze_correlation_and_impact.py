"""
Phase 4: Statistical & Econometric Analysis: PlayStation vs. Xbox Sentiment & Sales Impact.

Computes:
  1. Pearson & Spearman Correlation (Sentiment vs. Sales, Volume vs. Sales)
  2. Lagged Correlation (t-1 Sentiment vs. t Sales - Predictive / Leading Indicator Test)
  3. Simple Linear Regression (R^2, coefficient, p-value)
  4. Brand Share vs. Discourse Volume Share Analysis
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INPUT_FILE = os.path.join(DATA_DIR, "quarterly_sentiment_vs_sales.csv")
OUTPUT_METRICS_FILE = os.path.join(DATA_DIR, "statistical_analysis_results.csv")

def analyze():
    df = pd.read_csv(INPUT_FILE)

    ps_df = df[df["brand"] == "PlayStation"].sort_values("quarter").reset_index(drop=True)
    xb_df = df[df["brand"] == "Xbox"].sort_values("quarter").reset_index(drop=True)

    results = []

    for brand_name, b_df in [("PlayStation", ps_df), ("Xbox", xb_df)]:
        sales = b_df["hardware_sales_millions"].values
        mean_compound = b_df["mean_compound_score"].values
        net_sentiment = b_df["net_sentiment"].values
        volume = b_df["discourse_volume"].values

        # 1. Contemporaneous Correlation: Mean Compound vs Sales
        r_comp, p_comp = stats.pearsonr(mean_compound, sales)
        rho_comp, sp_p_comp = stats.spearmanr(mean_compound, sales)

        # 2. Contemporaneous Correlation: Net Sentiment vs Sales
        r_net, p_net = stats.pearsonr(net_sentiment, sales)
        rho_net, sp_p_net = stats.spearmanr(net_sentiment, sales)

        # 3. Contemporaneous Correlation: Discourse Volume vs Sales
        r_vol, p_vol = stats.pearsonr(volume, sales)
        rho_vol, sp_p_vol = stats.spearmanr(volume, sales)

        # 4. Lagged Correlation: Sentiment at t-1 vs Sales at t (Leading Indicator)
        # 11 data points for lag-1
        lag_comp = mean_compound[:-1]
        lag_net = net_sentiment[:-1]
        lag_vol = volume[:-1]
        lead_sales = sales[1:]

        r_lag_comp, p_lag_comp = stats.pearsonr(lag_comp, lead_sales)
        r_lag_net, p_lag_net = stats.pearsonr(lag_net, lead_sales)
        r_lag_vol, p_lag_vol = stats.pearsonr(lag_vol, lead_sales)

        # 5. OLS Regression: Sales ~ Mean Compound
        slope_c, intercept_c, r_value_c, p_value_c, std_err_c = stats.linregress(mean_compound, sales)
        r2_c = r_value_c ** 2

        # 6. OLS Regression: Sales ~ Discourse Volume
        slope_v, intercept_v, r_value_v, p_value_v, std_err_v = stats.linregress(volume, sales)
        r2_v = r_value_v ** 2

        results.append({
            "brand": brand_name,
            "pearson_r_compound_vs_sales": round(r_comp, 4),
            "p_val_compound_vs_sales": round(p_comp, 4),
            "spearman_rho_compound_vs_sales": round(rho_comp, 4),
            "pearson_r_net_sentiment_vs_sales": round(r_net, 4),
            "p_val_net_sentiment_vs_sales": round(p_net, 4),
            "pearson_r_volume_vs_sales": round(r_vol, 4),
            "p_val_volume_vs_sales": round(p_vol, 4),
            "spearman_rho_volume_vs_sales": round(rho_vol, 4),
            "lagged_r_sentiment_t_minus_1_vs_sales_t": round(r_lag_net, 4),
            "p_val_lagged_sentiment": round(p_lag_net, 4),
            "lagged_r_volume_t_minus_1_vs_sales_t": round(r_lag_vol, 4),
            "p_val_lagged_volume": round(p_lag_vol, 4),
            "r_squared_sentiment_model": round(r2_c, 4),
            "r_squared_volume_model": round(r2_v, 4),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_METRICS_FILE, index=False)

    print("=" * 70)
    print("  STATISTICAL & CORRELATION ANALYSIS (PHASE 4)")
    print("=" * 70)
    print("\n--- Correlation & Impact Matrix ---")
    for r in results:
        print(f"\nBrand: {r['brand']}")
        print(f"  • Sentiment vs. Sales (Pearson r)      : r = {r['pearson_r_compound_vs_sales']:+.4f} (p = {r['p_val_compound_vs_sales']:.4f})")
        print(f"  • Net Sentiment vs. Sales (Pearson r)  : r = {r['pearson_r_net_sentiment_vs_sales']:+.4f} (p = {r['p_val_net_sentiment_vs_sales']:.4f})")
        print(f"  • Discourse Volume vs. Sales (Pearson r): r = {r['pearson_r_volume_vs_sales']:+.4f} (p = {r['p_val_volume_vs_sales']:.4f})")
        print(f"  • Lagged Sentiment (t-1) vs. Sales (t) : r = {r['lagged_r_sentiment_t_minus_1_vs_sales_t']:+.4f} (p = {r['p_val_lagged_sentiment']:.4f})")
        print(f"  • Lagged Volume (t-1) vs. Sales (t)    : r = {r['lagged_r_volume_t_minus_1_vs_sales_t']:+.4f} (p = {r['p_val_lagged_volume']:.4f})")
        print(f"  • Regression R² (Sales ~ Sentiment)    : R² = {r['r_squared_sentiment_model']:.4f}")
        print(f"  • Regression R² (Sales ~ Volume)       : R² = {r['r_squared_volume_model']:.4f}")

    # Comparative Market Share vs Discourse Volume Share
    print("\n" + "=" * 70)
    print("  MARKET SHARE VS DISCOURSE VOLUME SHARE (COMPARATIVE)")
    print("=" * 70)
    
    comp_rows = []
    for q in ps_df["quarter"].values:
        ps_v = ps_df[ps_df["quarter"] == q]["discourse_volume"].values[0]
        xb_v = xb_df[xb_df["quarter"] == q]["discourse_volume"].values[0]
        total_v = ps_v + xb_v
        
        ps_s = ps_df[ps_df["quarter"] == q]["hardware_sales_millions"].values[0]
        xb_s = xb_df[xb_df["quarter"] == q]["hardware_sales_millions"].values[0]
        total_s = ps_s + xb_s

        comp_rows.append({
            "quarter": q,
            "ps5_share_of_discourse": round((ps_v / total_v) * 100, 1),
            "ps5_share_of_sales": round((ps_s / total_s) * 100, 1),
            "xbox_share_of_discourse": round((xb_v / total_v) * 100, 1),
            "xbox_share_of_sales": round((xb_s / total_s) * 100, 1),
        })

    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))

    # Overall correlation across combined market share
    r_share, p_share = stats.pearsonr(comp_df["ps5_share_of_discourse"], comp_df["ps5_share_of_sales"])
    print(f"\nDiscourse Share vs. Sales Share Correlation: r = {r_share:+.4f} (p = {p_share:.4f})")

if __name__ == "__main__":
    analyze()
