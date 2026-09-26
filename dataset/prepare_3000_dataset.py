#!/usr/bin/env python3
"""
Tràuna AI — prepare_3000_dataset.py
Gathers 3,000 balanced thumbnail-caption pairs from the 51 creator folders.
"""

import os
import shutil
import random
from collections import defaultdict

CREATORS_DIR = r"D:\Tràuna AI\trauna-ai\dataset\creators"
OUTPUT_DIR = r"D:\Tràuna AI\trauna-ai\dataset\training_creators_3000"
TARGET_TOTAL = 3000

def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "captions"), exist_ok=True)

    creator_pairs = defaultdict(list)

    print("Scanning creators dataset for matched pairs...")
    for root, dirs, files in os.walk(CREATORS_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                base = os.path.splitext(f)[0]
                img_path = os.path.join(root, f)

                # Find matching caption
                txt_in_same = os.path.join(root, base + '.txt')
                txt_in_caps = os.path.join(os.path.dirname(root), 'captions', base + '.txt')
                cap_path = None
                if os.path.exists(txt_in_same):
                    cap_path = txt_in_same
                elif os.path.exists(txt_in_caps):
                    cap_path = txt_in_caps

                if cap_path:
                    # Identify creator name (folder right under creators/)
                    rel = os.path.relpath(img_path, CREATORS_DIR)
                    creator = rel.split(os.sep)[0]
                    creator_pairs[creator].append((img_path, cap_path, base))

    total_available = sum(len(v) for v in creator_pairs.values())
    num_creators = len(creator_pairs)
    print(f"Found {total_available} total pairs across {num_creators} creators.")

    # Calculate quota per creator
    quota_per_creator = max(1, TARGET_TOTAL // num_creators + 15)
    selected = []
    
    random.seed(42)
    # First pass: collect up to quota from each creator
    for creator, pairs in sorted(creator_pairs.items()):
        random.shuffle(pairs)
        picked = pairs[:quota_per_creator]
        selected.extend(picked)

    random.shuffle(selected)
    final_selection = selected[:TARGET_TOTAL]
    print(f"Selected {len(final_selection)} balanced samples across all {num_creators} creators.")

    print("Copying/Linking samples to training folder...")
    count = 0
    for img_path, cap_path, base in final_selection:
        count += 1
        ext = os.path.splitext(img_path)[1].lower()
        new_base = f"sample_{count:04d}_{base}"
        dest_img = os.path.join(OUTPUT_DIR, "images", f"{new_base}{ext}")
        dest_cap = os.path.join(OUTPUT_DIR, "captions", f"{new_base}.txt")

        # Copy image and caption
        shutil.copy2(img_path, dest_img)
        shutil.copy2(cap_path, dest_cap)

        if count % 500 == 0 or count == len(final_selection):
            print(f"  Processed {count}/{len(final_selection)} samples...")

    print(f"\n[DONE] Prepared {count} training pairs in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
