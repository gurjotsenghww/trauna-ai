#!/usr/bin/env python3
"""
Tràuna AI — test_orientation_fix.py

Tests the 3 scenarios that previously failed:
  1. Minecraft + BeastBoyShub → was blocky 3D avatar, should now be photorealistic face
  2. Horror scene → was rear silhouette, should now be front-facing screaming face
  3. GTA city → was two walking silhouettes, should now be single front-facing face

All prompts use the new fixed format with front-facing + photorealistic overrides.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

ARTIFACT_DIR = r"C:\Users\Gurjotpal Singh\.gemini\antigravity\brain\17c0e50d-0f61-4094-8fdc-7395fdfb8ea7"

TEST_CASES = [
    {
        "name": "minecraft_realism_fix",
        "label": "Minecraft + BeastBoyShub (photorealistic override)",
        "prompt": "traunathumb, youtube thumbnail, solo photorealistic human BeastBoyShub screaming reaction face with headphones, front-facing, looking at camera on the right, blocky 3D Minecraft world with glowing torches on the left, vibrant saturated rim lighting, cinematic lighting, 16:9",
    },
    {
        "name": "horror_facing_fix",
        "label": "Horror (front-facing override)",
        "prompt": "traunathumb, youtube thumbnail, solo gaming streamer screaming terrified face, front-facing, looking at camera on the right, dark terrifying abandoned corridor with fog and eerie shadows on the left, horror flashlight rim lighting, cinematic lighting, 16:9",
    },
    {
        "name": "gta_facing_fix",
        "label": "GTA Los Santos (single subject fix)",
        "prompt": "traunathumb, youtube thumbnail, solo screaming shocked gaming streamer with LED headset, front-facing, looking at camera on the right, Los Santos city street with neon lights and sports car on the left, neon night city rim lights, cinematic lighting, 16:9",
    },
]

def main():
    from models.sdxl_turbo import SDXLTurboModel
    from PIL import Image

    print("[Test] Loading SDXL Turbo + Trauna LoRA...", flush=True)
    model = SDXLTurboModel()
    print("[Test] Model loaded. Starting orientation fix tests.\n", flush=True)

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"[Test {i}/3] {case['label']}", flush=True)
        print(f"  Prompt: {case['prompt'][:100]}...", flush=True)

        t0 = time.time()
        img = model.generate(
            prompt=case["prompt"],
            width=1024,
            height=576,
            num_inference_steps=4,
            guidance_scale=0.0,
            seed=42 + i,  # fixed seeds for reproducibility
        )
        elapsed = time.time() - t0

        # Upscale to 1280x720 for display
        img_hd = img.resize((1280, 720), Image.LANCZOS)

        out_path = os.path.join(ARTIFACT_DIR, f"orient_fix_{case['name']}.png")
        img_hd.save(out_path)
        print(f"  [OK] Saved ({elapsed:.1f}s): {out_path}\n", flush=True)
        results.append(out_path)

    print("\n[Test] ALL 3 ORIENTATION FIX TESTS COMPLETE!", flush=True)
    print("[Test] Check the images - each should show a FRONT-FACING face, not a silhouette.", flush=True)
    for r in results:
        print(f"  -> {r}", flush=True)

if __name__ == "__main__":
    main()
