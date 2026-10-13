"""
Phase 1: Data Hygiene, Deduplication & Noise Filtering for Reddit Console Discourse.

Input:
  - data/posts.csv
  - data/comments.csv

Output:
  - data/posts_clean.csv
  - data/comments_clean.csv

Filters applied:
  1. Deduplication by unique 'id'.
  2. Removal of deleted/removed content ('[deleted]', '[removed]', empty).
  3. Removal of AutoModerator automated bot comments.
  4. Cleanup of '[removed]' selftext placeholders in posts.
"""

import csv
import os
import sys
import time
from collections import Counter

# Handle large CSV fields safely on Windows
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POSTS_IN = os.path.join(DATA_DIR, "posts.csv")
POSTS_OUT = os.path.join(DATA_DIR, "posts_clean.csv")
COMMENTS_IN = os.path.join(DATA_DIR, "comments.csv")
COMMENTS_OUT = os.path.join(DATA_DIR, "comments_clean.csv")


def clean_posts():
    print("=" * 60)
    print("  PHASE 1: CLEANING POSTS (data/posts.csv)")
    print("=" * 60)
    start_time = time.time()

    if not os.path.exists(POSTS_IN):
        print(f"Error: {POSTS_IN} not found.")
        return

    seen_ids = set()
    total_read = 0
    dupes = 0
    empty_deleted = 0
    kept = 0
    sub_counts = Counter()

    with open(POSTS_IN, "r", encoding="utf-8", errors="ignore", newline="") as fin, \
         open(POSTS_OUT, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        try:
            header = next(reader)
        except StopIteration:
            print("Empty posts file.")
            return

        writer.writerow(header)

        id_idx = header.index("id")
        sub_idx = header.index("subreddit")
        title_idx = header.index("title")
        selftext_idx = header.index("selftext")

        for row in reader:
            total_read += 1

            if len(row) <= max(id_idx, sub_idx, title_idx, selftext_idx):
                continue

            row_id = row[id_idx].strip()
            if not row_id or row_id in seen_ids:
                dupes += 1
                continue
            seen_ids.add(row_id)

            title = row[title_idx].strip()
            if not title or title in ("[deleted]", "[removed]"):
                empty_deleted += 1
                continue

            # Standardize selftext: if [removed] or [deleted], clear it so it doesn't pollute text NLP
            if row[selftext_idx].strip() in ("[deleted]", "[removed]"):
                row[selftext_idx] = ""

            sub_counts[row[sub_idx].lower()] += 1
            writer.writerow(row)
            kept += 1

            if total_read % 500000 == 0:
                print(f"  Processed {total_read:>10,} posts | Kept: {kept:>9,} | Dupes: {dupes:>7,} | Elapsed: {time.time()-start_time:.1f}s")

    elapsed = time.time() - start_time
    in_size_mb = os.path.getsize(POSTS_IN) / (1024 * 1024)
    out_size_mb = os.path.getsize(POSTS_OUT) / (1024 * 1024)

    print(f"\n  -- Posts Cleaning Complete in {elapsed:.1f}s")
    print(f"     Total read     : {total_read:>10,}")
    print(f"     Duplicates     : {dupes:>10,} ({(dupes/total_read*100) if total_read else 0:.1f}%)")
    print(f"     Empty/Deleted  : {empty_deleted:>10,} ({(empty_deleted/total_read*100) if total_read else 0:.1f}%)")
    print(f"     Clean Posts    : {kept:>10,} ({(kept/total_read*100) if total_read else 0:.1f}%)")
    print(f"     File size      : {in_size_mb:.1f} MB -> {out_size_mb:.1f} MB")
    print(f"     By subreddit   : {dict(sub_counts)}")


def clean_comments():
    print("\n" + "=" * 60)
    print("  PHASE 1: CLEANING COMMENTS (data/comments.csv)")
    print("=" * 60)
    start_time = time.time()

    if not os.path.exists(COMMENTS_IN):
        print(f"Error: {COMMENTS_IN} not found.")
        return

    seen_ids = set()
    total_read = 0
    dupes = 0
    empty_deleted = 0
    automod = 0
    kept = 0
    sub_counts = Counter()

    with open(COMMENTS_IN, "r", encoding="utf-8", errors="ignore", newline="") as fin, \
         open(COMMENTS_OUT, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        try:
            header = next(reader)
        except StopIteration:
            print("Empty comments file.")
            return

        writer.writerow(header)

        id_idx = header.index("id")
        sub_idx = header.index("subreddit")
        author_idx = header.index("author")
        body_idx = header.index("body")

        for row in reader:
            total_read += 1

            if len(row) <= max(id_idx, sub_idx, author_idx, body_idx):
                continue

            row_id = row[id_idx].strip()
            if not row_id or row_id in seen_ids:
                dupes += 1
                continue
            seen_ids.add(row_id)

            body = row[body_idx].strip()
            if not body or body in ("[deleted]", "[removed]"):
                empty_deleted += 1
                continue

            author = row[author_idx].strip().lower()
            if author == "automoderator":
                automod += 1
                continue

            sub_counts[row[sub_idx].lower()] += 1
            writer.writerow(row)
            kept += 1

            if total_read % 1000000 == 0:
                print(f"  Processed {total_read:>10,} comments | Kept: {kept:>9,} | Dupes: {dupes:>7,} | Elapsed: {time.time()-start_time:.1f}s")

    elapsed = time.time() - start_time
    in_size_mb = os.path.getsize(COMMENTS_IN) / (1024 * 1024)
    out_size_mb = os.path.getsize(COMMENTS_OUT) / (1024 * 1024)

    print(f"\n  -- Comments Cleaning Complete in {elapsed:.1f}s")
    print(f"     Total read     : {total_read:>10,}")
    print(f"     Duplicates     : {dupes:>10,} ({(dupes/total_read*100) if total_read else 0:.1f}%)")
    print(f"     Empty/Deleted  : {empty_deleted:>10,} ({(empty_deleted/total_read*100) if total_read else 0:.1f}%)")
    print(f"     AutoModerator  : {automod:>10,} ({(automod/total_read*100) if total_read else 0:.1f}%)")
    print(f"     Clean Comments : {kept:>10,} ({(kept/total_read*100) if total_read else 0:.1f}%)")
    print(f"     File size      : {in_size_mb:.1f} MB -> {out_size_mb:.1f} MB")
    print(f"     By subreddit   : {dict(sub_counts)}")


def main():
    clean_posts()
    clean_comments()
    print("\n" + "=" * 60)
    print("  PHASE 1 COMPLETE: Clean datasets ready in data/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
