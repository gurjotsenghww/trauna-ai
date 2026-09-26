"""
compile_full_dataset.py
-----------------------
Scans ALL dataset/creators/<creator>/<game>/thumbnails/*.jpg
Pairs each thumbnail with its caption (and optional metadata).
Validates image integrity + dimensions >= 480x270.
Prepends trigger word: "traunathumb, professional youtube gaming thumbnail, "
Writes to: dataset/training_ready_full/images/ & dataset/training_ready_full/captions/
Saves:      dataset/training_ready_full/manifest.json
"""

import os
import sys
# Force UTF-8 stdout on Windows to avoid cp1252 codec errors
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import json
import shutil
import hashlib
from pathlib import Path
from PIL import Image

# ─── Config ───────────────────────────────────────────────────────────────────
TRIGGER = "traunathumb, professional youtube gaming thumbnail, "
MIN_W, MIN_H = 480, 270
CREATORS_DIR = Path(__file__).parent / "creators"
OUTPUT_DIR   = Path(__file__).parent / "training_ready_full"
IMAGES_OUT   = OUTPUT_DIR / "images"
CAPS_OUT     = OUTPUT_DIR / "captions"

# ─── Setup ────────────────────────────────────────────────────────────────────
IMAGES_OUT.mkdir(parents=True, exist_ok=True)
CAPS_OUT.mkdir(parents=True, exist_ok=True)

# ─── Tracking ─────────────────────────────────────────────────────────────────
manifest = {
    "total": 0,
    "skipped_no_caption": 0,
    "skipped_bad_image": 0,
    "skipped_too_small": 0,
    "duplicates_removed": 0,
    "by_creator": {}
}

seen_hashes = set()  # MD5 dedup

# ─── Scan ─────────────────────────────────────────────────────────────────────
if not CREATORS_DIR.exists():
    print(f"[ERROR] Creators dir not found: {CREATORS_DIR}")
    sys.exit(1)

creators = sorted([d for d in CREATORS_DIR.iterdir() if d.is_dir()])
print(f"Found {len(creators)} creators in {CREATORS_DIR}\n")

total_written = 0

for creator_dir in creators:
    creator_name = creator_dir.name
    creator_count = 0
    creator_skips = 0

    # Each creator can have multiple game subdirectories
    game_dirs = [d for d in creator_dir.iterdir() if d.is_dir()]

    for game_dir in game_dirs:
        thumbnails_dir = game_dir / "thumbnails"
        captions_dir   = game_dir / "captions"
        metadata_dir   = game_dir / "metadata"

        if not thumbnails_dir.exists():
            continue

        thumb_files = sorted(thumbnails_dir.glob("*.jpg")) + \
                      sorted(thumbnails_dir.glob("*.jpeg")) + \
                      sorted(thumbnails_dir.glob("*.png"))

        for thumb_path in thumb_files:
            stem = thumb_path.stem

            # ── 1. Caption must exist ─────────────────────────────────────
            cap_path = captions_dir / f"{stem}.txt"
            if not cap_path.exists():
                manifest["skipped_no_caption"] += 1
                creator_skips += 1
                continue

            # ── 2. Validate image ─────────────────────────────────────────
            try:
                img = Image.open(thumb_path)
                img.verify()
                img = Image.open(thumb_path)  # re-open after verify
                w, h = img.size
            except Exception as e:
                manifest["skipped_bad_image"] += 1
                creator_skips += 1
                print(f"  [SKIP] Bad image: {thumb_path.name} — {e}")
                continue

            if w < MIN_W or h < MIN_H:
                manifest["skipped_too_small"] += 1
                creator_skips += 1
                continue

            # ── 3. Dedup by MD5 hash ──────────────────────────────────────
            with open(thumb_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in seen_hashes:
                manifest["duplicates_removed"] += 1
                creator_skips += 1
                continue
            seen_hashes.add(file_hash)

            # ── 4. Build caption ──────────────────────────────────────────
            raw_caption = cap_path.read_text(encoding="utf-8").strip()

            # Optionally append metadata tags if available
            meta_path = metadata_dir / f"{stem}.json" if metadata_dir.exists() else None
            meta_tags = ""
            if meta_path and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    tags = []
                    if meta.get("game"):
                        tags.append(meta["game"].lower().replace(" ", "_"))
                    if meta.get("channel"):
                        tags.append(f"creator_{meta['channel'].lower().replace(' ', '_')}")
                    if tags:
                        meta_tags = ", " + ", ".join(tags)
                except Exception:
                    pass

            final_caption = TRIGGER + raw_caption + meta_tags

            # ── 5. Write output ───────────────────────────────────────────
            # Unique filename: creator_game_stem.jpg
            safe_creator = creator_name.replace(" ", "_")[:20]
            safe_game    = game_dir.name.replace(" ", "_")[:20]
            out_stem     = f"{safe_creator}_{safe_game}_{stem}"[:100]

            out_img  = IMAGES_OUT  / f"{out_stem}.jpg"
            out_cap  = CAPS_OUT    / f"{out_stem}.txt"

            # Convert to JPEG if needed
            if thumb_path.suffix.lower() in (".png", ".jpeg"):
                img = img.convert("RGB")
                img.save(str(out_img), "JPEG", quality=95)
            else:
                shutil.copy2(str(thumb_path), str(out_img))

            out_cap.write_text(final_caption, encoding="utf-8")

            creator_count += 1
            total_written += 1

    if creator_count > 0:
        manifest["by_creator"][creator_name] = {
            "written": creator_count,
            "skipped": creator_skips
        }
        print(f"  [OK]  {creator_name:30s}  written={creator_count:4d}  skipped={creator_skips}")
    elif creator_skips > 0:
        print(f"  [--]  {creator_name:30s}  written=   0  skipped={creator_skips}")

# ─── Manifest ─────────────────────────────────────────────────────────────────
manifest["total"] = total_written
manifest_path = OUTPUT_DIR / "manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"\n{'='*60}")
print(f"  Total written        : {total_written}")
print(f"  Skipped (no caption) : {manifest['skipped_no_caption']}")
print(f"  Skipped (bad image)  : {manifest['skipped_bad_image']}")
print(f"  Skipped (too small)  : {manifest['skipped_too_small']}")
print(f"  Duplicates removed   : {manifest['duplicates_removed']}")
print(f"  Manifest saved at    : {manifest_path}")
print(f"{'='*60}")
print(f"\n  Output dir: {OUTPUT_DIR}")
print("  Done. Ready for training.\n")
