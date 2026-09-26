#!/usr/bin/env python3
"""
Tràuna AI — High-Speed 15,000 Thumbnails Crawler (Strictly NEW Creators)
Crawls 10,000 - 15,000 thumbnails from top Indian Vloggers, Entertainment,
and International creators that were NOT previously scraped or trained.

Strict Exclusions:
  - Dynamically reads dataset/training_ready_full/manifest.json and automatically
    skips all 51 previously trained creators.
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

# 55 High-Impact NEW Creators (Zero overlap with the previous 51)
NEW_CURATED_CHANNELS = [
    # --- Top Indian Vloggers, Comedy & Entertainment ---
    {"name": "AshishChanchlani",   "handle": "ashishchanchlanivines", "limit": 260, "category": "Comedy_Skits"},
    {"name": "BBKiVines",          "handle": "BBKiVines",             "limit": 260, "category": "Comedy_Skits"},
    {"name": "CarryMinati",        "handle": "CarryMinati",           "limit": 260, "category": "Comedy_Roast"},
    {"name": "FlyingBeast",        "handle": "FlyingBeast320",        "limit": 260, "category": "Vlogs"},
    {"name": "FukraInsaan",        "handle": "FukraInsaan",           "limit": 260, "category": "Challenges_Vlogs"},
    {"name": "MumbikerNikhil",     "handle": "MumbikerNikhil",       "limit": 260, "category": "Vlogs"},
    {"name": "RimoravVlogs",       "handle": "RimoravVlogs",          "limit": 260, "category": "Challenges_Vlogs"},
    {"name": "Round2Hell",         "handle": "Round2hell",            "limit": 260, "category": "Comedy_Skits"},
    {"name": "Thugesh",            "handle": "Thugesh",               "limit": 260, "category": "Roast_Vlogs"},
    {"name": "TriggeredInsaan",    "handle": "triggeredinsaan",       "limit": 260, "category": "Roast_Vlogs"},
    {"name": "WanderersHub",       "handle": "WanderersHub",          "limit": 260, "category": "Travel_Vlogs"},
    {"name": "HarshBeniwal",       "handle": "HarshBeniwal",          "limit": 260, "category": "Comedy_Skits"},
    {"name": "TanmayBhat",         "handle": "TanmayBhatYT",          "limit": 260, "category": "Reaction_Vlogs"},
    {"name": "SamayRaina",         "handle": "samayraina",            "limit": 260, "category": "Comedy_Gaming"},
    {"name": "SlayyPoint",         "handle": "SlayyPoint",            "limit": 260, "category": "Roast_Vlogs"},
    {"name": "TheUK07Rider",       "handle": "TheUK07Rider",          "limit": 260, "category": "Moto_Vlogs"},
    {"name": "LakshayChaudhary",   "handle": "LakshayChaudhary",      "limit": 260, "category": "Commentary_Vlogs"},
    {"name": "TechBurner",         "handle": "TechBurner",            "limit": 260, "category": "Tech_Lifestyle"},
    {"name": "TechnicalGuruji",    "handle": "TechnicalGuruji",       "limit": 260, "category": "Tech_Reviews"},
    {"name": "DhruvRathee",        "handle": "dhruvrathee",           "limit": 260, "category": "Infotainment"},
    {"name": "NitishRajput",       "handle": "NitishRajput",          "limit": 260, "category": "Documentary_IRL"},
    {"name": "AbhiAndNiyu",        "handle": "AbhiandNiyu",           "limit": 260, "category": "Infotainment"},
    {"name": "AmitBhadana",        "handle": "AmitBhadana",           "limit": 260, "category": "Comedy_Skits"},
    {"name": "GauravZone",         "handle": "gauravzone",            "limit": 260, "category": "Vlogs"},

    # --- Top International Creators (Vlogs, IRL & Fresh Gaming) ---
    {"name": "CaseyNeistat",       "handle": "casey",                 "limit": 260, "category": "Cinematic_Vlogs"},
    {"name": "MarkRober",          "handle": "MarkRober",             "limit": 260, "category": "Science_IRL"},
    {"name": "DudePerfect",        "handle": "DudePerfect",           "limit": 260, "category": "Trickshots_Challenges"},
    {"name": "LoganPaul",          "handle": "loganpaulvlogs",        "limit": 260, "category": "Vlogs"},
    {"name": "KSI",                "handle": "KSI",                   "limit": 260, "category": "Reaction_Comedy"},
    {"name": "SypherPK",           "handle": "SypherPK",              "limit": 260, "category": "Battle_Royale"},
    {"name": "TommyInnit",         "handle": "TommyInnit",            "limit": 260, "category": "Minecraft_Vlogs"},
    {"name": "GeorgeNotFound",     "handle": "GeorgeNotFound",        "limit": 260, "category": "Minecraft"},
    {"name": "PrestonPlayz",       "handle": "PrestonPlayz",          "limit": 260, "category": "Minecraft_Challenges"},
    {"name": "Unspeakable",        "handle": "Unspeakable",           "limit": 260, "category": "Challenges_IRL"},
    {"name": "Airrack",            "handle": "airrack",               "limit": 260, "category": "IRL_Challenges"},
    {"name": "ZachKing",           "handle": "ZachKing",              "limit": 260, "category": "Visual_Effects"},
    {"name": "Veritasium",         "handle": "veritasium",            "limit": 260, "category": "Science_IRL"},
    {"name": "Kurzgesagt",         "handle": "kurzgesagt",            "limit": 260, "category": "Animation_Infotainment"},
    {"name": "Vsauce",             "handle": "Vsauce",                "limit": 260, "category": "Science_IRL"},
    {"name": "Jelly",              "handle": "Jelly",                 "limit": 260, "category": "Gaming_Fun"},
    {"name": "Slogoman",           "handle": "Slogoman",              "limit": 260, "category": "Gaming_Challenges"},
    {"name": "Kwebbelkop",         "handle": "kwebbelkop",            "limit": 260, "category": "Gaming_IRL"},
    {"name": "TypicalGamer",       "handle": "TypicalGamer",          "limit": 260, "category": "Fortnite_GTA"},
    {"name": "NickEh30",           "handle": "NickEh30",              "limit": 260, "category": "Fortnite_Family"},
    {"name": "Tfue",               "handle": "TTfue",                 "limit": 260, "category": "FPS_BattleRoyale"},
    {"name": "Shroud",             "handle": "shroud",                "limit": 260, "category": "Tactical_Shooter"},
    {"name": "TimTheTatman",       "handle": "timthetatman",          "limit": 260, "category": "Warzone_Humor"},
    {"name": "DrDisrespect",       "handle": "DrDisrespect",          "limit": 260, "category": "Cinematic_Gaming"},
    {"name": "Valkyrae",           "handle": "valkyrae",              "limit": 260, "category": "Variety_Gaming"},
    {"name": "Pokimane",           "handle": "pokimane",              "limit": 260, "category": "Variety_IRL"},
    {"name": "Ludwig",             "handle": "ludwig",                "limit": 260, "category": "Show_IRL"},
    {"name": "MoistCr1TiKaL",      "handle": "penguinz0",             "limit": 260, "category": "Commentary_IRL"},
]


def load_excluded_creators() -> set:
    """Loads all previously processed/trained creator names to guarantee zero duplication."""
    manifest_full = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "training_ready_full", "manifest.json")
    )
    excluded = set()
    if os.path.exists(manifest_full):
        try:
            with open(manifest_full, "r", encoding="utf-8") as f:
                data = json.load(f)
                excluded.update(data.get("by_creator", {}).keys())
        except Exception as e:
            print(f"[Warning] Failed to read full manifest exclusions: {e}")

    # Case-insensitive lookup set
    return {c.lower() for c in excluded}


def main():
    t_start = time.time()
    manifest_path = os.path.join(DEFAULT_DATASET_DIR, "crawl_manifest_15k_new.json")
    os.makedirs(DEFAULT_DATASET_DIR, exist_ok=True)

    excluded = load_excluded_creators()
    print("=" * 75)
    print("  Tràuna AI — 15,000 Thumbnails Crawler (Strictly NEW Creators)")
    print(f"  Excluded previous creators from training manifest: {len(excluded)}")
    print(f"  Target: {len(NEW_CURATED_CHANNELS)} fresh creators × ~260 videos = ~14,000 thumbnails")
    print(f"  Output directory: {DEFAULT_DATASET_DIR}")
    print("=" * 75)

    summary = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "channels_total": len(NEW_CURATED_CHANNELS),
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

    for idx, c in enumerate(NEW_CURATED_CHANNELS, start=1):
        creator_name = c["name"]
        handle = c["handle"]
        limit = c["limit"]
        category_override = c.get("category")
        channel_url = f"https://www.youtube.com/@{handle}/videos"

        # Hard check: Skip if previously completed in 51 trained creators
        if creator_name.lower() in excluded:
            print(f"[{idx}/{len(NEW_CURATED_CHANNELS)}] STRICT SKIP: {creator_name} was already trained in full dataset.")
            continue

        # Check if already processed in current directory
        creator_dir = os.path.join(DEFAULT_DATASET_DIR, creator_name)
        if os.path.exists(creator_dir):
            existing_count = 0
            for root, _, files in os.walk(creator_dir):
                existing_count += len([f for f in files if f.endswith(".jpg")])
            if existing_count >= int(limit * 0.75):
                print(f"[{idx}/{len(NEW_CURATED_CHANNELS)}] Skipping {creator_name} (already has {existing_count} thumbnails).")
                summary["creators_processed"][creator_name] = {
                    "handle": handle,
                    "scraped_in_this_run": existing_count,
                }
                continue

        print(f"\n[{idx}/{len(NEW_CURATED_CHANNELS)}] >>> Starting NEW Creator: {creator_name} (@{handle}) - Target: {limit}")

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

            print(f"[{idx}/{len(NEW_CURATED_CHANNELS)}] Finished {creator_name}: +{new_count} thumbnails saved. (Running Total: {summary['total_new_scraped']})")

        except Exception as e:
            print(f"[{idx}/{len(NEW_CURATED_CHANNELS)}] Error processing {creator_name}: {e}")
            summary["creators_processed"][creator_name] = {"error": str(e)}

    elapsed_mins = (time.time() - t_start) / 60
    summary["completed"] = True
    summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["elapsed_minutes"] = round(elapsed_mins, 2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"  Scraping Completed in {elapsed_mins:.1f} minutes!")
    print(f"  Total Fresh Thumbnails Collected: {summary['total_new_scraped']}")
    print(f"  Dataset Manifest: {manifest_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
