#!/usr/bin/env python3
"""
Tràuna AI — High-Speed Resumable 768x768 Dataset Converter & Validator
Approach A: Smart Subject & Face Portrait Crop (768x768 Lanczos3)

Features:
  - Resumable Checkpoint State: If interrupted, resumes immediately without data loss.
  - Smart Spatial Crop: Extracts high-resolution 768x768 portrait crops (right-side streamer reaction focus).
  - High-Speed Multiprocessing: Converts thousands of images in parallel.
  - Pairing & Verification: Ensures caption and image pairing, prepending trigger word.
  - Generates manifest and summary statistics.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SOURCE_DIR = Path(r"D:\Tràuna AI\trauna-ai\dataset\creators")
OUTPUT_DIR = Path(r"D:\Tràuna AI\trauna-ai\dataset\training_ready_768")
IMAGES_OUT = OUTPUT_DIR / "images"
CAPS_OUT = OUTPUT_DIR / "captions"
STATE_FILE = OUTPUT_DIR / "checkpoint_state.json"
MANIFEST_FILE = OUTPUT_DIR / "manifest_768.json"

TARGET_RES = 768
TRIGGER = "traunathumb, professional youtube gaming thumbnail"


def process_single_thumbnail(args):
    """
    Worker task: loads image, extracts 768x768 smart portrait crop, saves image and caption.
    """
    thumb_path_str, cap_path_str, meta_path_str, creator_name, game_name, stem = args
    thumb_path = Path(thumb_path_str)
    cap_path = Path(cap_path_str)

    try:
        # Verify and read caption
        raw_caption = cap_path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw_caption:
            raw_caption = f"gaming streamer reaction portrait, {creator_name}, {game_name}"

        # Enhance caption with trigger word and tags
        tags = []
        if meta_path_str and os.path.exists(meta_path_str):
            try:
                meta = json.loads(Path(meta_path_str).read_text(encoding="utf-8", errors="replace"))
                if meta.get("game"):
                    tags.append(meta["game"].lower().replace(" ", "_"))
                if meta.get("channel"):
                    tags.append(f"creator_{meta['channel'].lower().replace(' ', '_')}")
            except Exception:
                pass

        tag_str = (", " + ", ".join(tags)) if tags else ""
        if not raw_caption.startswith(TRIGGER):
            final_caption = f"{TRIGGER}, {raw_caption}{tag_str}"
        else:
            final_caption = f"{raw_caption}{tag_str}"

        # Load and process image
        with Image.open(thumb_path) as im:
            im = im.convert("RGB")
            w, h = im.size

            if w < 320 or h < 240:
                return None, "too_small"

            # Approach A: Smart Subject & Face Crop
            # For 1280x720, the face/streamer is typically on the right 60% of the canvas.
            # We crop a 1:1 square favoring the right/center where the creator is positioned.
            if w > h:
                # Square side is h
                side = h
                # Anchor crop to right side (offset to capture the streamer face portrait)
                # left edge at w - side
                crop_box = (max(0, w - side), 0, w, side)
                cropped = im.crop(crop_box)
            elif h > w:
                crop_box = (0, 0, w, w)
                cropped = im.crop(crop_box)
            else:
                cropped = im

            # High-quality Lanczos resampling to exactly 768x768
            rescaled = cropped.resize((TARGET_RES, TARGET_RES), Image.Resampling.LANCZOS)

            # Unique output stem
            safe_creator = creator_name.replace(" ", "_")[:20]
            safe_game = game_name.replace(" ", "_")[:20]
            out_stem = f"{safe_creator}_{safe_game}_{stem}"[:90]

            out_img = IMAGES_OUT / f"{out_stem}.jpg"
            out_cap = CAPS_OUT / f"{out_stem}.txt"

            rescaled.save(out_img, format="JPEG", quality=95, optimize=True)
            out_cap.write_text(final_caption, encoding="utf-8")

            return str(thumb_path), "ok"

    except Exception as e:
        return str(thumb_path), f"error: {str(e)}"


def run_conversion():
    print("=" * 70)
    print("  Tràuna AI — 768x768 Smart Dataset Converter (Resumable)")
    print(f"  Source:      {SOURCE_DIR}")
    print(f"  Destination: {OUTPUT_DIR}")
    print(f"  Target Res:  {TARGET_RES}x{TARGET_RES} (Lanczos3 Face Crop)")
    print("=" * 70)

    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    CAPS_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load Checkpoint State
    state = {"processed_files": {}, "total_processed": 0, "last_updated": 0}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            print(f"[Checkpoint] Resuming from existing state: {len(state.get('processed_files', {}))} files already processed.")
        except Exception as e:
            print(f"[Checkpoint Warning] Could not read existing state: {e}. Starting fresh.")

    processed_set = set(state.get("processed_files", {}).keys())

    # 2. Discover files
    print("\n[Discovery] Scanning creator directories for paired thumbnails...")
    tasks = []
    skipped_existing = 0

    for creator_dir in sorted(SOURCE_DIR.iterdir()):
        if not creator_dir.is_dir():
            continue
        creator_name = creator_dir.name

        for game_dir in sorted(creator_dir.iterdir()):
            if not game_dir.is_dir():
                continue
            game_name = game_dir.name

            thumb_dir = game_dir / "thumbnails"
            cap_dir = game_dir / "captions"
            meta_dir = game_dir / "metadata"

            if not thumb_dir.exists():
                continue

            for thumb_path in thumb_dir.glob("*.jpg"):
                cap_path = cap_dir / f"{thumb_path.stem}.txt"
                if not cap_path.exists():
                    continue

                thumb_str = str(thumb_path)
                if thumb_str in processed_set:
                    skipped_existing += 1
                    continue

                meta_path = meta_dir / f"{thumb_path.stem}.json"
                tasks.append((
                    thumb_str,
                    str(cap_path),
                    str(meta_path) if meta_path.exists() else None,
                    creator_name,
                    game_name,
                    thumb_path.stem,
                ))

    print(f"[Discovery] Found {len(tasks)} new files to process (Already processed: {skipped_existing}).")

    if not tasks:
        print("[Done] All available thumbnails are already processed to 768x768!")
        return

    # 3. Parallel Execution with periodic checkpoint saving
    t0 = time.time()
    batch_size = 200
    total_new = len(tasks)
    completed_count = 0
    errors_count = 0

    # Use CPU cores efficiently
    max_workers = min(os.cpu_count() or 4, 8)
    print(f"[Worker Pool] Launching {max_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit in chunks for low memory overhead
        for i in range(0, total_new, batch_size):
            chunk = tasks[i : i + batch_size]
            futures = [executor.submit(process_single_thumbnail, t) for t in chunk]

            for future in as_completed(futures):
                path_str, status = future.result()
                if status == "ok":
                    state["processed_files"][path_str] = True
                    completed_count += 1
                else:
                    errors_count += 1

            # Save checkpoint state after every batch
            state["total_processed"] = len(state["processed_files"])
            state["last_updated"] = time.time()
            STATE_FILE.write_text(json.dumps(state), encoding="utf-8")

            elapsed = time.time() - t0
            speed = completed_count / max(0.1, elapsed)
            percent = (completed_count / total_new) * 100
            print(f"[Progress] {completed_count:5d}/{total_new} ({percent:5.1f}%) | Speed: {speed:5.1f} img/s | Checkpoint Saved", flush=True)

    # 4. Final Manifest
    manifest = {
        "dataset_name": "trauna_gaming_768x768",
        "resolution": TARGET_RES,
        "crop_method": "approach_a_portrait_lanczos3",
        "total_pairs": len(state["processed_files"]),
        "errors_skipped": errors_count,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  🎉 768x768 Conversion Complete in {total_time/60:.2f} minutes!")
    print(f"  Total Validated 768x768 Images: {len(state['processed_files'])}")
    print(f"  Images Directory:               {IMAGES_OUT}")
    print(f"  Captions Directory:             {CAPS_OUT}")
    print(f"  Manifest File:                  {MANIFEST_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    run_conversion()
