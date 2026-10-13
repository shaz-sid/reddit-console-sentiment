"""
Phase 3: Domain-Adapted Sentiment Analysis & Quarterly Brand Aggregation.

Features:
  - Custom Gaming & Console Discourse Lexicon for VADER (re-weights inverted slang like 'sick', 'insane', 'peak', 'mid', 'bricked', etc.)
  - Normalization of gaming shorthand ('huge W', 'massive L', 'take the L')
  - Community-level attribution (r/PS5 & r/PlayStation -> PlayStation, r/Xbox & r/XboxSeriesX -> Xbox, r/consoles -> keyword tagged)
  - Strict quarterly aggregation matching 2023-Q1 through 2025-Q4
  - Merges sentiment scores with official hardware sales figures
"""

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Safety limit for large fields
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POSTS_FILE = os.path.join(DATA_DIR, "posts_clean.csv")
COMMENTS_FILE = os.path.join(DATA_DIR, "comments_clean.csv")
SALES_FILE = os.path.join(DATA_DIR, "quarterly_sales.csv")
OUTPUT_SUMMARY_FILE = os.path.join(DATA_DIR, "quarterly_sentiment_vs_sales.csv")

# 12 Quarter Windows (2023-01-01 to 2026-01-01)
QUARTERS = [
    ("2023-Q1", 1672531200, 1680307199),
    ("2023-Q2", 1680307200, 1688169599),
    ("2023-Q3", 1688169600, 1696118399),
    ("2023-Q4", 1696118400, 1704067199),
    ("2024-Q1", 1704067200, 1711929599),
    ("2024-Q2", 1711929600, 1719791999),
    ("2024-Q3", 1719792000, 1727740799),
    ("2024-Q4", 1727740800, 1735689599),
    ("2025-Q1", 1735689600, 1743465599),
    ("2025-Q2", 1743465600, 1751327999),
    ("2025-Q3", 1751328000, 1759276799),
    ("2025-Q4", 1759276800, 1767225599),
]

def get_quarter(utc_ts: int):
    for q_name, q_start, q_end in QUARTERS:
        if q_start <= utc_ts <= q_end:
            return q_name
    return None

# Gaming-Specific Domain Lexicon Adaptations
GAMING_SLANG_LEXICON = {
    # Re-calibrating inverted slang (words that look negative to generic models but are positive in gaming)
    "sick": 2.8,
    "sickness": 1.5,
    "wicked": 2.5,
    "insane": 2.4,
    "insanely": 2.5,
    "nasty": 2.2,
    "filthy": 2.2,
    "badass": 2.8,
    "beast": 2.6,
    "killer": 2.2,
    "killing it": 3.0,
    "wild": 1.8,
    
    # Positive gaming & online culture praise
    "goat": 3.4,
    "g.o.a.t": 3.4,
    "peak": 3.0,
    "banger": 2.9,
    "bangers": 2.9,
    "cooking": 2.5,
    "cooked with": 2.5,
    "clutch": 2.6,
    "buttery": 2.5,
    "butter": 2.2,
    "goty": 3.2,
    "masterpiece": 3.5,
    "underrated": 2.0,
    "hyped": 2.6,
    "hype": 2.5,
    "crisp": 2.2,
    "slaps": 2.6,
    "fire": 2.8,
    "dub": 2.5,
    "dubs": 2.5,
    "smooth": 2.2,
    "gorgeous": 3.0,
    "stunning": 3.0,

    # Negative gaming & online culture criticism
    "mid": -1.8,
    "dogwater": -2.8,
    "bricked": -3.4,
    "brick": -2.5,
    "unplayable": -3.0,
    "dumpster fire": -3.2,
    "flop": -2.8,
    "flopped": -2.8,
    "copium": -2.0,
    "hopium": -1.2,
    "overpriced": -2.4,
    "ripoff": -2.8,
    "rip off": -2.8,
    "scam": -3.0,
    "cashgrab": -2.8,
    "cash grab": -2.8,
    "downgrade": -2.4,
    "downgraded": -2.4,
    "clunky": -2.0,
    "stutter": -2.2,
    "stuttery": -2.2,
    "stuttering": -2.2,
    "washed": -2.2,
    "dead game": -2.8,
    "dead console": -2.8,
    "anticonsumer": -2.8,
    "anti-consumer": -2.8,
    "paywall": -2.4,
    "paywalled": -2.4,
    "microtransactions": -2.0,
    "mtx": -2.0,
    "predatory": -2.8,
    "trash": -2.8,
    "garbage": -2.8,
    "crap": -2.2,
}

RE_HUGE_W = re.compile(r'\b(huge|big|massive|great|major|common)\s+[wW]\b', re.IGNORECASE)
RE_HUGE_L = re.compile(r'\b(huge|big|massive|major|common)\s+[lL]\b', re.IGNORECASE)
RE_W_FOR = re.compile(r'\b[wW]\s+for\s+', re.IGNORECASE)
RE_L_FOR = re.compile(r'\b[lL]\s+for\s+', re.IGNORECASE)
RE_TAKE_L = re.compile(r'\btake\s+the\s+[lL]\b', re.IGNORECASE)
RE_TAKE_W = re.compile(r'\btake\s+the\s+[wW]\b', re.IGNORECASE)

def normalize_gaming_text(text: str) -> str:
    text = RE_HUGE_W.sub(r'\1 victory', text)
    text = RE_HUGE_L.sub(r'\1 defeat', text)
    text = RE_W_FOR.sub('win for ', text)
    text = RE_L_FOR.sub('loss for ', text)
    text = RE_TAKE_L.sub('take the defeat', text)
    text = RE_TAKE_W.sub('take the win', text)
    return text

# Brand keywords for neutral r/consoles
PS_KEYWORDS = re.compile(r'\b(ps5|ps4|playstation|playstation\s*5|sony|dualsense|spiderman|god of war|naughty dog|helldivers)\b', re.IGNORECASE)
XBOX_KEYWORDS = re.compile(r'\b(xbox|series\s*[xs]|microsoft|game\s*pass|gamepass|halo|starfield|forza|gears)\b', re.IGNORECASE)

def attribute_brand(subreddit: str, text: str) -> str | None:
    sub = subreddit.lower()
    if sub in ("ps5", "playstation"):
        return "PlayStation"
    if sub in ("xbox", "xboxseriesx"):
        return "Xbox"
    if sub == "consoles":
        has_ps = bool(PS_KEYWORDS.search(text))
        has_xbox = bool(XBOX_KEYWORDS.search(text))
        if has_ps and not has_xbox:
            return "PlayStation"
        if has_xbox and not has_ps:
            return "Xbox"
        # If both or neither are mentioned, omit from binary brand perception
        return None
    return None


class SentimentAggregator:
    def __init__(self):
        # stats[(quarter, brand)] = {'count': 0, 'compound_sum': 0.0, 'pos': 0, 'neu': 0, 'neg': 0}
        self.stats = defaultdict(lambda: {'count': 0, 'compound_sum': 0.0, 'pos': 0, 'neu': 0, 'neg': 0})

    def add_score(self, quarter: str, brand: str, compound: float):
        entry = self.stats[(quarter, brand)]
        entry['count'] += 1
        entry['compound_sum'] += compound
        if compound >= 0.05:
            entry['pos'] += 1
        elif compound <= -0.05:
            entry['neg'] += 1
        else:
            entry['neu'] += 1


def main():
    print("=" * 60)
    print("  PHASE 3: GAMING SENTIMENT ANALYSIS & QUARTERLY AGGREGATION")
    print("=" * 60)
    start_time = time.time()

    # 1. Initialize Domain-Adapted VADER
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(GAMING_SLANG_LEXICON)
    analyzer.lexicon.update({'victory': 2.8, 'defeat': -2.5, 'win': 2.5, 'loss': -2.5})
    print("  Initialized Domain-Adapted VADER with gaming slang lexicon.")

    aggregator = SentimentAggregator()

    # 2. Process Posts
    print("\n--- Scoring Posts (posts_clean.csv) ---")
    p_start = time.time()
    posts_scored = 0

    with open(POSTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader)
        sub_idx = header.index("subreddit")
        title_idx = header.index("title")
        selftext_idx = header.index("selftext")
        utc_idx = header.index("created_utc")

        for row in reader:
            if len(row) <= max(sub_idx, title_idx, selftext_idx, utc_idx):
                continue

            try:
                ts = int(row[utc_idx])
            except ValueError:
                continue

            q = get_quarter(ts)
            if not q:
                continue

            title = row[title_idx]
            selftext = row[selftext_idx]
            full_text = f"{title}. {selftext}".strip()
            
            sub = row[sub_idx]
            brand = attribute_brand(sub, full_text)
            if not brand:
                continue

            cleaned_text = normalize_gaming_text(full_text[:500])
            score = analyzer.polarity_scores(cleaned_text)['compound']

            aggregator.add_score(q, brand, score)
            posts_scored += 1

            if posts_scored % 200000 == 0:
                print(f"  Scored {posts_scored:>8,} posts | Elapsed: {time.time()-p_start:.1f}s")

    print(f"  -- Finished Posts: {posts_scored:,} scored in {time.time()-p_start:.1f}s")

    # 3. Process Comments
    print("\n--- Scoring Comments (comments_clean.csv) ---")
    c_start = time.time()
    comments_scored = 0

    with open(COMMENTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader)
        sub_idx = header.index("subreddit")
        body_idx = header.index("body")
        utc_idx = header.index("created_utc")

        for row in reader:
            if len(row) <= max(sub_idx, body_idx, utc_idx):
                continue

            try:
                ts = int(row[utc_idx])
            except ValueError:
                continue

            q = get_quarter(ts)
            if not q:
                continue

            body = row[body_idx]
            sub = row[sub_idx]
            brand = attribute_brand(sub, body)
            if not brand:
                continue

            cleaned_text = normalize_gaming_text(body[:350])
            score = analyzer.polarity_scores(cleaned_text)['compound']

            aggregator.add_score(q, brand, score)
            comments_scored += 1

            if comments_scored % 1000000 == 0:
                rate = comments_scored / (time.time() - c_start)
                print(f"  Scored {comments_scored:>10,} comments | Rate: {rate:>6,.0f}/s | Elapsed: {time.time()-c_start:.1f}s")

    print(f"  -- Finished Comments: {comments_scored:,} scored in {time.time()-c_start:.1f}s")

    # 4. Load Sales Table and Merge
    sales_df = pd.read_csv(SALES_FILE)

    rows = []
    for q_name, _, _ in QUARTERS:
        sales_row = sales_df[sales_df["quarter"] == q_name].iloc[0]
        for brand in ("PlayStation", "Xbox"):
            entry = aggregator.stats.get((q_name, brand), {'count': 0, 'compound_sum': 0.0, 'pos': 0, 'neu': 0, 'neg': 0})
            total_n = entry['count']
            if total_n > 0:
                mean_compound = entry['compound_sum'] / total_n
                pct_pos = (entry['pos'] / total_n) * 100.0
                pct_neg = (entry['neg'] / total_n) * 100.0
                pct_neu = (entry['neu'] / total_n) * 100.0
                net_sentiment = pct_pos - pct_neg
            else:
                mean_compound = pct_pos = pct_neg = pct_neu = net_sentiment = 0.0

            sales_val = sales_row["ps5_units_millions"] if brand == "PlayStation" else sales_row["xbox_units_millions"]

            rows.append({
                "quarter": q_name,
                "brand": brand,
                "discourse_volume": total_n,
                "mean_compound_score": round(mean_compound, 4),
                "pct_positive": round(pct_pos, 2),
                "pct_negative": round(pct_neg, 2),
                "pct_neutral": round(pct_neu, 2),
                "net_sentiment": round(net_sentiment, 2),
                "hardware_sales_millions": sales_val,
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)

    print("\n" + "=" * 60)
    print("  PHASE 3 COMPLETE: Quarterly Sentiment & Sales Table Ready")
    print(f"  Saved to: {OUTPUT_SUMMARY_FILE}")
    print("=" * 60)
    print("\nPreview:")
    print(result_df[["quarter", "brand", "discourse_volume", "mean_compound_score", "net_sentiment", "hardware_sales_millions"]].to_string(index=False))


if __name__ == "__main__":
    main()
