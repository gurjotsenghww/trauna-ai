#!/usr/bin/env python3
"""
Tràuna AI — generate_5_showcase.py
Generates 5 distinct BeastBoyShub showcase thumbnails with SDXL Turbo + LoRA.
Loads model ONCE for high-speed sequential batch generation.
"""

import os
import sys
import time
import shutil
import torch
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

AI_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AI_DIR)

from models.sdxl_turbo import SDXLTurboModel

ARTIFACT_DIR = r"C:\Users\Gurjotpal Singh\.gemini\antigravity\brain\17c0e50d-0f61-4094-8fdc-7395fdfb8ea7"
BACKEND_OUT_DIR = os.path.abspath(os.path.join(AI_DIR, "..", "backend", "outputs"))
os.makedirs(BACKEND_OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

PROMPTS = [
    {
        "id": 1,
        "title": "Minecraft Hardcore — Ender Dragon Battle",
        "raw_prompt": "BeastBoyShub screaming in shock with glowing headphones, Minecraft hardcore Ender Dragon flying in dark purple starry void sky with lightning",
        "prompt": "traunathumb, youtube thumbnail, solo BeastBoyShub screaming in shock with glowing headphones on the right, Minecraft hardcore Ender Dragon flying in dark purple starry void sky with lightning on the left, cinematic lighting, 16:9",
        "seed": 101,
    },
    {
        "id": 2,
        "title": "Survival Horror — Abandoned Asylum Ghost",
        "raw_prompt": "BeastBoyShub terrified reaction face screaming with headphones, dark abandoned horror asylum corridor with single flashlight beam and eerie ghost silhouette in fog",
        "prompt": "traunathumb, youtube thumbnail, solo BeastBoyShub terrified reaction face screaming with headphones on the right, dark abandoned horror asylum corridor with single flashlight beam and eerie ghost silhouette in fog on the left, cinematic lighting, 16:9",
        "seed": 202,
    },
    {
        "id": 3,
        "title": "GTA 5 Stunt Heist — Ramp Jump Police Explosion",
        "raw_prompt": "BeastBoyShub laughing excited reaction face with headset, GTA 5 neon sports car jumping over police cars with massive fiery explosion in Los Santos",
        "prompt": "traunathumb, youtube thumbnail, solo BeastBoyShub laughing excited reaction face with headset on the right, GTA 5 neon sports car jumping over police cars with massive fiery explosion in Los Santos on the left, cinematic lighting, 16:9",
        "seed": 303,
    },
    {
        "id": 4,
        "title": "Minecraft Nether — Lava Falls & Blaze Attack",
        "raw_prompt": "BeastBoyShub shocked reaction face with open mouth, Minecraft nether fortress with glowing lava falls and floating fire blazes shooting sparks",
        "prompt": "traunathumb, youtube thumbnail, solo BeastBoyShub shocked reaction face with open mouth on the right, Minecraft nether fortress with glowing lava falls and floating fire blazes shooting sparks on the left, cinematic lighting, 16:9",
        "seed": 404,
    },
    {
        "id": 5,
        "title": "Only Up Rage Challenge — Vertigo Tower Fall",
        "raw_prompt": "BeastBoyShub rage screaming reaction with hands on head, high altitude obstacle course tower with dizzying drop and colorful sky",
        "prompt": "traunathumb, youtube thumbnail, solo BeastBoyShub rage screaming reaction with hands on head on the right, high altitude obstacle course tower with dizzying drop and colorful sky on the left, cinematic lighting, 16:9",
        "seed": 505,
    },
]

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

def main():
    print("=" * 70, flush=True)
    print("  Tràuna AI — 5-Image Showcase Generator (SDXL Turbo + LoRA)", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    print("[1/2] Initializing model & 5,000-Step Gaming LoRA once into memory...", flush=True)
    model = SDXLTurboModel()
    print(f"[Model Ready] Loaded in {time.time() - t_start:.1f}s\n", flush=True)

    results = []

    for item in PROMPTS:
        p_id = item["id"]
        title = item["title"]
        prompt = item["prompt"]
        seed = item["seed"]

        print(f"--- [Image {p_id}/5] {title} ---", flush=True)
        print(f"Prompt: {prompt}", flush=True)
        print(f"Seed: {seed} | Resolution: 1024x576 -> 1280x720", flush=True)

        t0 = time.time()
        raw_img = model.generate(
            prompt=prompt,
            width=1024,
            height=576,
            num_inference_steps=4,
            seed=seed,
        )
        gen_time = time.time() - t0

        # High-quality Lanczos resize to strictly 1280x720
        final_img = raw_img.resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.Resampling.LANCZOS)

        backend_path = os.path.join(BACKEND_OUT_DIR, f"showcase_{p_id}.png")
        artifact_path = os.path.join(ARTIFACT_DIR, f"showcase_{p_id}.png")

        final_img.save(backend_path, format="PNG", quality=95)
        shutil.copyfile(backend_path, artifact_path)

        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0

        print(f"Saved: {artifact_path} ({TARGET_WIDTH}x{TARGET_HEIGHT}) in {gen_time:.1f}s | Peak VRAM: {peak_vram:.2f} GB\n", flush=True)

        results.append({
            "id": p_id,
            "title": title,
            "prompt": prompt,
            "raw_prompt": item["raw_prompt"],
            "seed": seed,
            "time": f"{gen_time:.1f}s",
            "peak_vram": f"{peak_vram:.2f} GB",
            "artifact_path": artifact_path,
            "backend_path": backend_path,
        })

    total_time = time.time() - t_start
    print("=" * 70, flush=True)
    print(f"Batch generation completed! 5/5 thumbnails generated in {total_time:.1f}s total.", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
