#!/usr/bin/env python3
"""
Tràuna AI — Training Dataset Preprocessor
Prepares and formats the crawled YouTube thumbnails and captions
for LoRA fine-tuning in Diffusers / Kohya / PEFT.

Features:
  - Scans all creator folders in dataset/creators/
  - Verifies paired thumbnail (.jpg) and caption (.txt)
  - Inserts custom trigger word: 'traunathumb, youtube gaming thumbnail style'
  - Validates image integrity and resolution
  - Organizes training-ready dataset with metadata manifest
"""

import os
import sys
import json
import shutil
from PIL import Image
from typing import Dict, Any, List, Optional

# Ensure UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "creators"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "training_ready"))
TRIGGER_WORD = "traunathumb, professional youtube gaming thumbnail"


def prepare_dataset(
    target_creators: Optional[List[str]] = None,
    target_games: Optional[List[str]] = None,
    max_images: Optional[int] = None,
    trigger_word: str = TRIGGER_WORD,
):
    print("=" * 70)
    print("  Tràuna AI — Dataset Preparation for LoRA Training")
    print(f"  Source:      {SOURCE_DIR}")
    print(f"  Destination: {OUTPUT_DIR}")
    print(f"  Trigger:     '{trigger_word}'")
    print("=" * 70)

    images_out = os.path.join(OUTPUT_DIR, "images")
    captions_out = os.path.join(OUTPUT_DIR, "captions")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(captions_out, exist_ok=True)

    paired_count = 0
    skipped_corrupted = 0
    by_creator = {}
    by_game = {}

    for root, dirs, files in os.walk(SOURCE_DIR):
        if os.path.basename(root) == "thumbnails":
            thumb_dir = root
            game_dir = os.path.dirname(thumb_dir)
            caption_dir = os.path.join(game_dir, "captions")

            rel = os.path.relpath(game_dir, SOURCE_DIR).split(os.sep)
            creator = rel[0] if len(rel) > 0 else "Unknown"
            game = rel[1] if len(rel) > 1 else "General"

            # Filter creators if specified
            if target_creators and creator not in target_creators:
                continue
            # Filter games if specified
            if target_games and game not in target_games:
                continue

            for fname in files:
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                base_name = os.path.splitext(fname)[0]
                thumb_path = os.path.join(thumb_dir, fname)
                caption_path = os.path.join(caption_dir, f"{base_name}.txt")

                if not os.path.exists(caption_path):
                    continue

                # Verify image integrity
                try:
                    with Image.open(thumb_path) as im:
                        w, h = im.size
                        if w < 480 or h < 270:
                            skipped_corrupted += 1
                            continue
                except Exception:
                    skipped_corrupted += 1
                    continue

                # Read and enrich caption
                try:
                    with open(caption_path, "r", encoding="utf-8") as f:
                        raw_caption = f.read().strip()
                except Exception:
                    raw_caption = f"YouTube gaming thumbnail in {creator} style, {game} gameplay"

                if trigger_word and not raw_caption.startswith(trigger_word):
                    final_caption = f"{trigger_word}, {raw_caption}"
                else:
                    final_caption = raw_caption

                # Write to training_ready
                dest_img = os.path.join(images_out, f"{base_name}.jpg")
                dest_cap = os.path.join(captions_out, f"{base_name}.txt")

                # Copy image and write formatted caption
                if not os.path.exists(dest_img):
                    shutil.copy2(thumb_path, dest_img)

                with open(dest_cap, "w", encoding="utf-8") as f:
                    f.write(final_caption)

                paired_count += 1
                by_creator[creator] = by_creator.get(creator, 0) + 1
                by_game[game] = by_game.get(game, 0) + 1

                if max_images and paired_count >= max_images:
                    break
            if max_images and paired_count >= max_images:
                break

    manifest = {
        "total_training_pairs": paired_count,
        "trigger_word": trigger_word,
        "by_creator": by_creator,
        "by_game": by_game,
        "skipped_corrupted": skipped_corrupted,
    }

    manifest_path = os.path.join(OUTPUT_DIR, "training_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"  Training Dataset Prepared Successfully!")
    print(f"  Total Validated Pairs: {paired_count}")
    print(f"  Output Directory:      {OUTPUT_DIR}")
    print(f"  Manifest:              {manifest_path}")
    print("=" * 70)
    return manifest


if __name__ == "__main__":
    prepare_dataset()
