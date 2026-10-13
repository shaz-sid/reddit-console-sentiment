"""
Collect Reddit posts and comments via Arctic Shift API (not the Reddit API).

Date range : 2023-01-01 to 2026-01-01
Subreddits : XboxSeriesX, Xbox, PS5, PlayStation, consoles

Output      : data/posts.csv, data/comments.csv  (append-safe, resumable)

Usage       : python collect_reddit_data.py
"""

import csv
import json
import os
import time
from datetime import datetime, timezone

import requests

# ── Config ──────────────────────────────────────────────────────────────────
BASE = "https://arctic-shift.photon-reddit.com/api"
SUBREDDITS = ["XboxSeriesX", "Xbox", "PS5", "PlayStation", "consoles"]
AFTER = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())  # 1672531200
BEFORE = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())  # 1767225600
LIMIT = 100  # API max; higher values return 400
RATE_LIMIT_SECS = 1.0  # be polite
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")

POST_FIELDS = [
    "id", "subreddit", "author", "title", "selftext", "url",
    "score", "upvote_ratio", "num_comments", "created_utc",
    "permalink", "link_flair_text", "is_self", "domain",
]
COMMENT_FIELDS = [
    "id", "subreddit", "author", "body", "score",
    "created_utc", "parent_id", "link_id", "permalink",
    "is_submitter", "controversiality",
]

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    """Load {subreddit: {posts: last_utc, comments: last_utc}}."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def utc_str(ts):
    """Epoch → readable date string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fetch_page(endpoint: str, subreddit: str, after_utc: int) -> list[dict] | None:
    """Fetch one page of results from Arctic Shift. Returns None on network/API failure."""
    url = f"{BASE}/{endpoint}/search"
    params = {
        "subreddit": subreddit,
        "after": after_utc,
        "before": BEFORE,
        "limit": LIMIT,
        "sort": "asc",
    }
    for attempt in range(10):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [rate-limited] waiting {wait}s ...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json().get("data")
            if data is not None:
                return data
        except (requests.RequestException, json.JSONDecodeError) as e:
            wait = min(2 ** attempt * 2, 60)
            print(f"  [warn attempt {attempt+1}/10] {e} - retrying in {wait}s ...")
            time.sleep(wait)
    return None


def init_csv(path: str, fieldnames: list[str]):
    """Write header row if the file doesn't exist yet."""
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_rows(path: str, fieldnames: list[str], rows: list[dict]):
    """Append rows to CSV, picking only the fields we care about."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def collect(kind: str, subreddit: str, progress: dict):
    """
    Collect all posts or comments for one subreddit.
    kind: 'posts' or 'comments'
    """
    endpoint = "posts" if kind == "posts" else "comments"
    fields = POST_FIELDS if kind == "posts" else COMMENT_FIELDS
    csv_path = os.path.join(DATA_DIR, f"{kind}.csv")

    init_csv(csv_path, fields)

    # Resume from last saved position
    sub_prog = progress.setdefault(subreddit, {})
    after_utc = sub_prog.get(kind, AFTER)

    if after_utc >= BEFORE:
        print(f"  r/{subreddit} - {kind} already complete (up to {utc_str(after_utc)})")
        return 0

    total = 0

    print(f"\n{'-'*60}")
    print(f"  r/{subreddit} - {kind}  (from {utc_str(after_utc)})")
    print(f"{'-'*60}")

    while True:
        items = fetch_page(endpoint, subreddit, after_utc)
        if items is None:
            print(f"  [error] Temporary API/connection issue at timestamp {after_utc}. Retrying in 15s...")
            time.sleep(15)
            continue

        if not items:
            sub_prog[kind] = BEFORE
            save_progress(progress)
            break

        append_rows(csv_path, fields, items)
        total += len(items)

        last_utc = max(int(item.get("created_utc", 0)) for item in items)
        after_utc = max(last_utc, after_utc + 1)
        sub_prog[kind] = after_utc
        save_progress(progress)

        ts = utc_str(last_utc)
        print(f"  + {total:>7,} {kind}  |  latest: {ts}")

        if len(items) < LIMIT:
            sub_prog[kind] = BEFORE
            save_progress(progress)
            break  # no more pages

        time.sleep(RATE_LIMIT_SECS)

    print(f"  -- Done: {total:,} {kind} collected for r/{subreddit}")
    return total


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    progress = load_progress()

    grand_total = {"posts": 0, "comments": 0}

    for sub in SUBREDDITS:
        for kind in ("posts", "comments"):
            n = collect(kind, sub, progress)
            grand_total[kind] += n

    print(f"\n{'='*60}")
    print(f"  ALL DONE")
    print(f"  Posts    : {grand_total['posts']:>10,}")
    print(f"  Comments : {grand_total['comments']:>10,}")
    print(f"  Files    : {os.path.join(DATA_DIR, 'posts.csv')}")
    print(f"             {os.path.join(DATA_DIR, 'comments.csv')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
