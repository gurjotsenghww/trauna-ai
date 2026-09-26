#!/usr/bin/env python3
"""
Tràuna AI — Bulk Dataset Scraper (Top 25 Indian Gaming Creators)
Scrapes 5,000 - 8,000 thumbnails, captions, metadata, and video descriptions.

Target Structure:
  dataset/creators/<Creator>/<Game>/
      ├── thumbnails/
      │   ├── <Creator>_<Game>_1.jpg
      │   └── ...
      ├── captions/
      │   ├── <Creator>_<Game>_1.txt
      │   └── ...
      └── metadata/
          ├── <Creator>_<Game>_1.json
          └── ...
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

# Ensure UTF-8 output encoding for Windows command line
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from youtube_crawler import crawl, DEFAULT_DATASET_DIR, DEFAULT_COOKIE_FILE

# India's Top 25 Gaming & Content Creators
TOP_25_CREATORS = [
    {"name": "TotalGaming",             "handle": "totalgaming093",          "limit": 260},
    {"name": "TechnoGamerz",            "handle": "TechnoGamerzOfficial",    "limit": 260},
    {"name": "Mythpat",                 "handle": "Mythpat",                 "limit": 260},
    {"name": "CarryisLive",             "handle": "CarryisLive",             "limit": 260},
    {"name": "LiveInsaan",              "handle": "liveinsaan",              "limit": 260},
    {"name": "BeastBoyShub",            "handle": "BeastBoyShub",            "limit": 260},
    {"name": "ASGaming",                "handle": "ASGamingSahil",           "limit": 260},
    {"name": "LokeshGamer",             "handle": "LOKESHGAMER",             "limit": 260},
    {"name": "GyanGaming",              "handle": "GyanGaming",              "limit": 260},
    {"name": "DesiGamers",              "handle": "DesiGamers_",             "limit": 260},
    {"name": "TheRawKneeGames",         "handle": "TheRawKneeGames",         "limit": 260},
    {"name": "Scout",                   "handle": "scout",                   "limit": 260},
    {"name": "Mortal",                  "handle": "Mortalgamingyt",          "limit": 260},
    {"name": "JonathanGaming",          "handle": "JONATHANGAMINGYT",        "limit": 260},
    {"name": "DynamoGaming",            "handle": "DynamoGaming",            "limit": 260},
    {"name": "PayalGaming",             "handle": "PayalGaming",             "limit": 260},
    {"name": "GamerFleet",              "handle": "GamerFleet",              "limit": 260},
    {"name": "AnshuBisht",              "handle": "AnshuBisht",              "limit": 260},
    {"name": "ChapatiHindustaniGamer",  "handle": "ChapatiHindustaniGamer",  "limit": 260},
    {"name": "HindustanGamer",          "handle": "HindustanGamer",          "limit": 260},
    {"name": "YesSmartyPie",            "handle": "YesSmartyPie",            "limit": 260},
    {"name": "TwoSideGamers",           "handle": "TWO_SIDE_GAMERS",         "limit": 260},
    {"name": "AntaryamiGaming",         "handle": "AntaryamiGaming",         "limit": 260},
    {"name": "AlphaClasher",            "handle": "AlphaClasher",            "limit": 260},
    {"name": "KaashPlays",              "handle": "KaashPlays",              "limit": 260},
]


def run_batch():
    t_start = time.time()
    manifest_path = os.path.join(DEFAULT_DATASET_DIR, "dataset_manifest.json")
    os.makedirs(DEFAULT_DATASET_DIR, exist_ok=True)

    print("=" * 70)
    print("  Tràuna AI — Top 25 Indian Creators Batch Scraper")
    print(f"  Target: 25 creators × ~260 videos = ~6,500 thumbnails")
    print(f"  Output directory: {DEFAULT_DATASET_DIR}")
    print("=" * 70)

    summary = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "creators_processed": {},
        "total_new_scraped": 0,
        "completed": False,
    }

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                old = json.load(f)
                summary["creators_processed"] = old.get("creators_processed", {})
        except Exception:
            pass

    for idx, c in enumerate(TOP_25_CREATORS, start=1):
        creator_name = c["name"]
        handle = c["handle"]
        limit = c["limit"]
        channel_url = f"https://www.youtube.com/@{handle}/videos"

        print(f"\n[{idx}/25] >>> Starting Creator: {creator_name} (@{handle}) - Limit: {limit}")

        try:
            res = crawl(
                target=channel_url,
                creator_name=creator_name,
                limit=limit,
                output_base_dir=DEFAULT_DATASET_DIR,
                cookie_file=DEFAULT_COOKIE_FILE,
            )

            new_count = res.get("total_scraped", 0)
            summary["total_new_scraped"] += new_count
            summary["creators_processed"][creator_name] = {
                "handle": handle,
                "scraped_in_this_run": new_count,
                "by_game": res.get("by_game", {}),
            }

            # Update manifest checkpoint
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"[{idx}/25] Finished {creator_name}: +{new_count} thumbnails saved.")

        except Exception as e:
            print(f"[{idx}/25] Error processing {creator_name}: {e}")
            summary["creators_processed"][creator_name] = {"error": str(e)}

    elapsed_mins = (time.time() - t_start) / 60
    summary["completed"] = True
    summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["elapsed_minutes"] = round(elapsed_mins, 2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"  Batch Scraping Completed in {elapsed_mins:.1f} minutes!")
    print(f"  Total New Thumbnails Added: {summary['total_new_scraped']}")
    print(f"  Dataset Manifest Saved: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_batch()
