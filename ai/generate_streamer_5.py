#!/usr/bin/env python3
"""
Tràuna AI — generate_streamer_5.py
Generates 5 distinct thumbnails featuring the gaming streamer ("him")
across 5 iconic game environments using SDXL Turbo + LoRA.
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
        "title": "Minecraft Hardcore — Cave of Diamonds",
        "scenario": "Excited discovery of a rare underground diamond cavern",
        "prompt": "traunathumb, youtube thumbnail, solo excited gaming streamer wearing LED headset screaming with mouth open on the right, blocky 3D Minecraft underground cave with glowing diamonds and blue torches on the left, volumetric dust, cinematic rim lighting, 16:9",
        "seed": 111,
    },
    {
        "id": 2,
        "title": "Survival Horror — Haunted Asylum Jump-Scare",
        "scenario": "Terrified pure jump-scare in an abandoned corridor",
        "prompt": "traunathumb, youtube thumbnail, solo terrified gaming streamer with glowing headset screaming in pure horror on the right, dark abandoned haunted asylum hallway with single flashlight beam and creepy shadow on the left, atmospheric fog, cold rim lighting, 16:9",
        "seed": 222,
    },
    {
        "id": 3,
        "title": "GTA 5 Heist — Supercar Mega Ramp Explosion",
        "scenario": "Adrenaline-fueled laughing reaction during a high-speed stunt",
        "prompt": "traunathumb, youtube thumbnail, solo laughing gaming streamer with headphones looking forward on the right, Los Santos neon city street with speeding sports car and massive fire explosion on the left, motion blur, cinematic lighting, 16:9",
        "seed": 333,
    },
    {
        "id": 4,
        "title": "Minecraft Nether — Lava Fortress & Blaze Attack",
        "scenario": "Shock and panic facing a fire boss in the Nether",
        "prompt": "traunathumb, youtube thumbnail, solo shocked gaming streamer with glowing headphones shouting in disbelief on the right, fiery Minecraft nether fortress with glowing lava falls and hovering blazes shooting sparks on the left, warm fire glow, 16:9",
        "seed": 444,
    },
    {
        "id": 5,
        "title": "Futuristic Arena — 1v5 Clutch Victory",
        "scenario": "Intense focused victory roar in a competitive cyberpunk arena",
        "prompt": "traunathumb, youtube thumbnail, solo intense focused gaming streamer with headset shouting in victory on the right, futuristic neon cyberpunk arena with glowing holograms and smoke on the left, electric cyan rim lighting, 16:9",
        "seed": 555,
    },
]

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

def main():
    print("=" * 70, flush=True)
    print("  Tràuna AI — 5 Streamer Showcase Generations (SDXL Turbo + LoRA)", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()
    print("[1/2] Loading SDXL Turbo & Tràuna Gaming LoRA...", flush=True)
    model = SDXLTurboModel()
    print(f"[Model Ready] Loaded in {time.time() - t_start:.1f}s\n", flush=True)

    for item in PROMPTS:
        p_id = item["id"]
        title = item["title"]
        prompt = item["prompt"]
        seed = item["seed"]

        print(f"--- [Thumbnail {p_id}/5] {title} ---", flush=True)
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

        final_img = raw_img.resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.Resampling.LANCZOS)

        backend_path = os.path.join(BACKEND_OUT_DIR, f"streamer_showcase_{p_id}.png")
        artifact_path = os.path.join(ARTIFACT_DIR, f"streamer_showcase_{p_id}.png")

        final_img.save(backend_path, format="PNG", quality=95)
        shutil.copyfile(backend_path, artifact_path)

        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
        print(f"Saved: {artifact_path} in {gen_time:.1f}s | Peak VRAM: {peak_vram:.2f} GB\n", flush=True)

    print("=" * 70, flush=True)
    print(f"Batch generation completed in {time.time() - t_start:.1f}s!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
