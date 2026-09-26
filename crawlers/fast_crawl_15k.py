#!/usr/bin/env python3
"""
Tràuna AI — High-Speed 15,000 Thumbnails Crawler (Indian & International Vlogs + Gaming)
Scrapes ~10,000 - 15,000 high-resolution YouTube thumbnails, captions, and metadata.

Target Directory Structure (Preserved):
  dataset/creators/<Creator>/<Category>/
      ├── thumbnails/
      │   ├── <Creator>_<Category>_1.jpg
      │   └── ...
      ├── captions/
      │   ├── <Creator>_<Category>_1.txt
      │   └── ...
      └── metadata/
          ├── <Creator>_<Category>_1.json
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

# 55 Curated Top Channels across Indian & International Vlogging + Gaming
CURATED_CHANNELS = [
    # --- Top Indian Vloggers & Entertainment ---
    {"name": "SouravJoshiVlogs",       "handle": "souravjoshivlogs7028",      "limit": 260, "category": "Vlogs"},
    {"name": "FlyingBeast",            "handle": "FlyingBeast320",            "limit": 260, "category": "Vlogs"},
    {"name": "MumbikerNikhil",         "handle": "MumbikerNikhil",           "limit": 260, "category": "Vlogs"},
    {"name": "CarryMinati",            "handle": "CarryMinati",               "limit": 200, "category": "Comedy_Roast"},
    {"name": "Round2Hell",             "handle": "Round2hell",                "limit": 120, "category": "Comedy_Skits"},
    {"name": "BBKiVines",              "handle": "BBKiVines",                 "limit": 200, "category": "Comedy_Skits"},
    {"name": "AshishChanchlani",       "handle": "ashishchanchlanivines",     "limit": 160, "category": "Comedy_Skits"},
    {"name": "HarshBeniwal",           "handle": "HarshBeniwal",              "limit": 150, "category": "Comedy_Skits"},
    {"name": "TriggeredInsaan",        "handle": "triggeredinsaan",           "limit": 260, "category": "Roast_Vlogs"},
    {"name": "FukraInsaan",            "handle": "FukraInsaan",               "limit": 260, "category": "Challenges_Vlogs"},
    {"name": "Thugesh",                "handle": "Thugesh",                   "limit": 260, "category": "Roast_Vlogs"},
    {"name": "WanderersHub",           "handle": "WanderersHub",              "limit": 260, "category": "Travel_Vlogs"},
    {"name": "RimoravVlogs",           "handle": "RimoravVlogs",              "limit": 260, "category": "Challenges_Vlogs"},

    # --- Top Indian Gaming Creators ---
    {"name": "TechnoGamerz",           "handle": "TechnoGamerzOfficial",     "limit": 260, "category": None},
    {"name": "TotalGaming",            "handle": "totalgaming093",           "limit": 260, "category": None},
    {"name": "Mythpat",                "handle": "Mythpat",                  "limit": 260, "category": None},
    {"name": "CarryisLive",            "handle": "CarryisLive",              "limit": 260, "category": None},
    {"name": "BeastBoyShub",           "handle": "BeastBoyShub",             "limit": 260, "category": None},
    {"name": "LiveInsaan",             "handle": "liveinsaan",               "limit": 260, "category": None},
    {"name": "ASGaming",               "handle": "ASGamingSahil",            "limit": 260, "category": None},
    {"name": "LokeshGamer",            "handle": "LOKESHGAMER",              "limit": 260, "category": None},
    {"name": "GyanGaming",             "handle": "GyanGaming",               "limit": 260, "category": None},
    {"name": "DesiGamers",             "handle": "DesiGamers_",              "limit": 260, "category": None},
    {"name": "TheRawKneeGames",        "handle": "TheRawKneeGames",          "limit": 260, "category": None},
    {"name": "Scout",                  "handle": "scout",                    "limit": 260, "category": None},
    {"name": "Mortal",                 "handle": "Mortalgamingyt",           "limit": 260, "category": None},
    {"name": "JonathanGaming",         "handle": "JONATHANGAMINGYT",         "limit": 260, "category": None},
    {"name": "DynamoGaming",           "handle": "DynamoGaming",             "limit": 260, "category": None},
    {"name": "PayalGaming",            "handle": "PayalGaming",              "limit": 260, "category": None},
    {"name": "GamerFleet",             "handle": "GamerFleet",               "limit": 260, "category": None},
    {"name": "AnshuBisht",             "handle": "AnshuBisht",               "limit": 260, "category": None},
    {"name": "ChapatiHindustaniGamer", "handle": "ChapatiHindustaniGamer",  "limit": 260, "category": None},
    {"name": "YesSmartyPie",           "handle": "YesSmartyPie",             "limit": 260, "category": None},
    {"name": "TwoSideGamers",          "handle": "TWO_SIDE_GAMERS",          "limit": 260, "category": None},
    {"name": "AntaryamiGaming",        "handle": "AntaryamiGaming",          "limit": 260, "category": None},
    {"name": "AlphaClasher",           "handle": "AlphaClasher",             "limit": 260, "category": None},

    # --- Top International Creators (Vlogs + Gaming) ---
    {"name": "MrBeast",                "handle": "MrBeast",                  "limit": 260, "category": "Challenges_IRL"},
    {"name": "CaseyNeistat",           "handle": "casey",                    "limit": 260, "category": "Cinematic_Vlogs"},
    {"name": "MarkRober",              "handle": "MarkRober",                "limit": 150, "category": "Science_IRL"},
    {"name": "DudePerfect",            "handle": "DudePerfect",              "limit": 260, "category": "Trickshots_Challenges"},
    {"name": "Sidemen",                "handle": "Sidemen",                  "limit": 260, "category": "Challenges_Vlogs"},
    {"name": "LoganPaul",              "handle": "loganpaulvlogs",           "limit": 260, "category": "Vlogs"},
    {"name": "KSI",                    "handle": "KSI",                      "limit": 260, "category": "Reaction_Comedy"},
    {"name": "PewDiePie",              "handle": "PewDiePie",                "limit": 260, "category": None},
    {"name": "Markiplier",             "handle": "markiplier",               "limit": 260, "category": None},
    {"name": "Jacksepticeye",          "handle": "jacksepticeye",            "limit": 260, "category": None},
    {"name": "DanTDM",                 "handle": "DanTDM",                   "limit": 260, "category": None},
    {"name": "LazarBeam",              "handle": "LazarBeam",                "limit": 260, "category": None},
    {"name": "AliA",                   "handle": "AliA",                     "limit": 260, "category": None},
    {"name": "SypherPK",               "handle": "SypherPK",                 "limit": 260, "category": None},
    {"name": "CoryxKenshin",           "handle": "CoryxKenshin",             "limit": 260, "category": None},
    {"name": "Dream",                  "handle": "dream",                    "limit": 150, "category": None},
    {"name": "TommyInnit",             "handle": "TommyInnit",               "limit": 260, "category": None},
    {"name": "Aphmau",                 "handle": "Aphmau",                   "limit": 260, "category": None},
    {"name": "SSundee",                "handle": "SSundee",                  "limit": 260, "category": None},
    {"name": "PrestonPlayz",           "handle": "PrestonPlayz",             "limit": 260, "category": None},
]


def main():
    t_start = time.time()
    manifest_path = os.path.join(DEFAULT_DATASET_DIR, "crawl_manifest_15k.json")
    os.makedirs(DEFAULT_DATASET_DIR, exist_ok=True)

    print("=" * 75)
    print("  Tràuna AI — High-Speed 15,000 Dataset Scraper (Vlogs + Gaming)")
    print(f"  Target: {len(CURATED_CHANNELS)} channels × ~260 videos = ~13,500 - 15,000 thumbnails")
    print(f"  Output directory: {DEFAULT_DATASET_DIR}")
    print("=" * 75)

    summary = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "channels_total": len(CURATED_CHANNELS),
        "creators_processed": {},
        "total_new_scraped": 0,
        "completed": False,
    }

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                old = json.load(f)
                summary["creators_processed"] = old.get("creators_processed", {})
                summary["total_new_scraped"] = old.get("total_new_scraped", 0)
        except Exception:
            pass

    for idx, c in enumerate(CURATED_CHANNELS, start=1):
        creator_name = c["name"]
        handle = c["handle"]
        limit = c["limit"]
        category_override = c.get("category")
        channel_url = f"https://www.youtube.com/@{handle}/videos"

        # Check if already completed
        prev = summary["creators_processed"].get(creator_name)
        if prev and prev.get("scraped_in_this_run", 0) >= (limit * 0.8):
            print(f"[{idx}/{len(CURATED_CHANNELS)}] Skipping {creator_name} (already has {prev.get('scraped_in_this_run')} thumbnails).")
            continue

        print(f"\n[{idx}/{len(CURATED_CHANNELS)}] >>> Starting: {creator_name} (@{handle}) - Target: {limit}")

        try:
            res = crawl(
                target=channel_url,
                creator_name=creator_name,
                limit=limit,
                output_base_dir=DEFAULT_DATASET_DIR,
                cookie_file=DEFAULT_COOKIE_FILE,
                game_override=category_override,
            )

            new_count = res.get("total_scraped", 0)
            summary["total_new_scraped"] += new_count
            summary["creators_processed"][creator_name] = {
                "handle": handle,
                "scraped_in_this_run": new_count,
                "by_game": res.get("by_game", {}),
            }

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"[{idx}/{len(CURATED_CHANNELS)}] Finished {creator_name}: +{new_count} thumbnails saved. (Running Total: {summary['total_new_scraped']})")

        except Exception as e:
            print(f"[{idx}/{len(CURATED_CHANNELS)}] Error processing {creator_name}: {e}")
            summary["creators_processed"][creator_name] = {"error": str(e)}

    elapsed_mins = (time.time() - t_start) / 60
    summary["completed"] = True
    summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["elapsed_minutes"] = round(elapsed_mins, 2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"  Batch Scraping Completed in {elapsed_mins:.1f} minutes!")
    print(f"  Total Thumbnails Collected: {summary['total_new_scraped']}")
    print(f"  Dataset Manifest: {manifest_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
