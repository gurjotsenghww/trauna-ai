#!/usr/bin/env python3
"""
Tràuna AI — Bulk Dataset Scraper (Top 25+ International Creators)
Scrapes thousands of thumbnails, captions, metadata, and video descriptions
across top international gaming and content creators requested by the user.

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

# Top 25+ International Creators (Includes all requested creators with verified handles)
INTERNATIONAL_CREATORS = [
    {"name": "MrBeast",         "handle": "MrBeast",        "limit": 180},
    {"name": "Markiplier",      "handle": "markiplier",     "limit": 180},
    {"name": "RyanTrahan",      "handle": "ryan",           "limit": 180},
    {"name": "Jacksepticeye",   "handle": "jacksepticeye",  "limit": 180},
    {"name": "IShowSpeed",      "handle": "IShowSpeed",     "limit": 180},
    {"name": "JakePaul",        "handle": "jakepaul",       "limit": 180},
    {"name": "Sidemen",         "handle": "Sidemen",        "limit": 180},
    {"name": "KaiCenat",        "handle": "KaiCenat",       "limit": 180},
    {"name": "PewDiePie",       "handle": "PewDiePie",      "limit": 180},
    {"name": "Caylus",          "handle": "Infinite",       "limit": 180},
    {"name": "DanTDM",          "handle": "DanTDM",         "limit": 180},
    {"name": "BigPanda",        "handle": "BigPanda",       "limit": 180},
    {"name": "Dream",           "handle": "dream",          "limit": 180},
    {"name": "LazarBeam",       "handle": "LazarBeam",      "limit": 180},
    {"name": "Technoblade",     "handle": "Technoblade",    "limit": 180},
    {"name": "IGN",             "handle": "IGN",            "limit": 180},
    {"name": "Mazepti",         "handle": "mazepti",        "limit": 180},
    {"name": "Gunscreen",       "handle": "Gunscreen1",     "limit": 180},
    {"name": "JeeJYT",          "handle": "jeeJYT",         "limit": 180},
    {"name": "22luke",          "handle": "22luke",         "limit": 180},
    {"name": "Qzeq",            "handle": "qzeq",           "limit": 180},
    {"name": "Javeus",          "handle": "jjaveus",        "limit": 180},
    {"name": "CoryxKenshin",    "handle": "CoryxKenshin",   "limit": 180},
    {"name": "SSundee",         "handle": "SSundee",        "limit": 180},
    {"name": "Ninja",           "handle": "Ninja",          "limit": 180},
    {"name": "Aphmau",          "handle": "Aphmau",         "limit": 180},
    {"name": "AliA",            "handle": "AliA",           "limit": 180},
]


def run_batch():
    t_start = time.time()
    manifest_path = os.path.join(DEFAULT_DATASET_DIR, "dataset_manifest_international.json")
    os.makedirs(DEFAULT_DATASET_DIR, exist_ok=True)

    total_target = sum(c["limit"] for c in INTERNATIONAL_CREATORS)
    print("=" * 75)
    print("  Tràuna AI — Top International Creators Batch Scraper")
    print(f"  Target: {len(INTERNATIONAL_CREATORS)} creators × ~180 videos = ~{total_target} thumbnails")
    print(f"  Output directory: {DEFAULT_DATASET_DIR}")
    print("=" * 75)

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

    for idx, c in enumerate(INTERNATIONAL_CREATORS, start=1):
        creator_name = c["name"]
        handle = c["handle"]
        limit = c["limit"]
        channel_url = f"https://www.youtube.com/@{handle}/videos"

        print(f"\n[{idx}/{len(INTERNATIONAL_CREATORS)}] >>> Starting Creator: {creator_name} (@{handle}) - Limit: {limit}")

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

            print(f"[{idx}/{len(INTERNATIONAL_CREATORS)}] Finished {creator_name}: +{new_count} thumbnails saved.")

        except Exception as e:
            print(f"[{idx}/{len(INTERNATIONAL_CREATORS)}] Error processing {creator_name}: {e}")
            summary["creators_processed"][creator_name] = {"error": str(e)}

    elapsed_mins = (time.time() - t_start) / 60
    summary["completed"] = True
    summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["elapsed_minutes"] = round(elapsed_mins, 2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"  International Batch Scraping Completed in {elapsed_mins:.1f} minutes!")
    print(f"  Total New Thumbnails Added: {summary['total_new_scraped']}")
    print(f"  Dataset Manifest Saved: {manifest_path}")
    print("=" * 75)


if __name__ == "__main__":
    run_batch()
