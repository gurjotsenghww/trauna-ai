#!/usr/bin/env python3
"""
Tràuna AI — Dual-Stream Generation Engine (768x768 Portrait + 1024x576 Scene Composite)
The 0 -> 50 Quality Architecture:
  1. Stream A (Subject): Dedicated 768x768 1:1 portrait generation.
     - 100% of the attention tokens are dedicated to the human streamer.
     - Forces camera-facing portrait mode.
     - Eliminates rear silhouettes and conjoined multi-person nucleation.
  2. Stream B (World/Scene): Dedicated 1024x576 16:9 gaming environment generation.
     - Pure environment tokens (Minecraft, Horror, GTA, Valorant, etc.).
     - Zero human tokens in background -> zero hybrid/zombie mutant glitches.
  3. Spatial Composite:
     - Scales background to 1280x720.
     - Blends streamer portrait onto the right side with feathered cosine transition mask.
     - Produces master 1280x720 PNG ready for YouTube.
"""

import sys
import os
import time
import argparse
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_cosine_feather_mask(width: int, height: int, feather_width: int = 180) -> Image.Image:
    """
    Creates a horizontal alpha mask for blending the face crop onto the background.
    Left 0 to feather_width transitions smoothly from 0.0 to 1.0 using a cosine S-curve.
    Rest of the image is 1.0 (fully opaque).
    """
    mask = np.ones((height, width), dtype=np.float32)
    # Cosine ramp: 0.5 * (1 - cos(pi * x / feather_width))
    x_indices = np.arange(feather_width)
    ramp = 0.5 * (1.0 - np.cos(np.pi * x_indices / feather_width))
    mask[:, :feather_width] = ramp
    mask_uint8 = (mask * 255.0).astype(np.uint8)
    return Image.fromarray(mask_uint8, mode="L")


def composite_dual_stream(bg_img: Image.Image, face_img: Image.Image, target_w: int = 1280, target_h: int = 720) -> Image.Image:
    """
    Composites 768x768 portrait onto 1280x720 background.
    """
    # 1. Resize background to target canvas
    bg_resized = bg_img.resize((target_w, target_h), Image.Resampling.LANCZOS).convert("RGBA")

    # 2. Scale face to fill full height (720x720)
    face_side = target_h
    face_scaled = face_img.resize((face_side, face_side), Image.Resampling.LANCZOS).convert("RGBA")

    # 3. Position face on the right side
    face_x = target_w - face_side  # 1280 - 720 = 560
    face_y = 0

    # 4. Generate smooth feathered mask
    feather_width = int(face_side * 0.28)  # ~200px smooth blend zone
    alpha_mask = build_cosine_feather_mask(face_side, face_side, feather_width=feather_width)

    # Put mask into face alpha channel
    face_scaled.putalpha(alpha_mask)

    # 5. Composite over background
    composite = Image.alpha_composite(bg_resized, Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0)))
    composite.paste(face_scaled, (face_x, face_y), face_scaled)

    return composite.convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser(description="Tràuna AI — Dual-Stream Generation Engine")
    parser.add_argument("--prompt", type=str, default="", help="Combined or user prompt")
    parser.add_argument("--subject_prompt", type=str, default="", help="Specific subject/face prompt (768x768)")
    parser.add_argument("--world_prompt", type=str, default="", help="Specific game/world prompt (1024x576)")
    parser.add_argument("--output", type=str, required=True, help="Output image file path (.png)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Determine subject and world prompts
    subject_prompt = args.subject_prompt.strip()
    world_prompt = args.world_prompt.strip()

    if not subject_prompt or not world_prompt:
        # Fallback / split from combined prompt
        raw = args.prompt.strip() or "excited gaming streamer with headphones, Minecraft nether fortress"
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) >= 2:
            subject_part = parts[0]
            world_part = ", ".join(parts[1:])
        else:
            subject_part = raw
            world_part = "dramatic gaming world with neon rim lighting"

        if not subject_prompt:
            subject_prompt = f"traunathumb, solo {subject_part}, expressive reaction face with open mouth, wearing LED gaming headphones, front-facing portrait, looking directly at camera, sharp focus, 1:1 portrait"
        if not world_prompt:
            world_prompt = f"traunathumb, {world_part}, cinematic lighting, vibrant saturated atmosphere, 16:9 widescreen"

    print("=" * 70, flush=True)
    print("  🚀 Tràuna AI — Dual-Stream Generation (768x768 Face + 1024x576 Scene)", flush=True)
    print(f"  Stream A (Face 768x768):  {subject_prompt[:80]}...", flush=True)
    print(f"  Stream B (World 1024x576): {world_prompt[:80]}...", flush=True)
    print(f"  Output Master Target:     1280x720 HD -> {args.output}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    from models.sdxl_turbo import SDXLTurboModel
    model = SDXLTurboModel()

    # ── Step 1: Render 768x768 Portrait Face ──
    print("\n[Stream A] Generating 768x768 Dedicated Portrait...", flush=True)
    t_face = time.time()
    face_img = model.generate(
        prompt=subject_prompt,
        width=768,
        height=768,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        seed=args.seed,
    )
    print(f"[Stream A] Face portrait rendered in {time.time() - t_face:.1f}s", flush=True)

    # ── Step 2: Render 1024x576 Background Scene ──
    print("\n[Stream B] Generating 1024x576 Clean Game World...", flush=True)
    t_world = time.time()
    world_seed = (args.seed + 100) if args.seed is not None else None
    world_img = model.generate(
        prompt=world_prompt,
        width=1024,
        height=576,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        seed=world_seed,
    )
    print(f"[Stream B] World scene rendered in {time.time() - t_world:.1f}s", flush=True)

    # ── Step 3: Composite Master 1280x720 ──
    print("\n[Composite] Blending face onto background (1280x720)...", flush=True)
    final_thumbnail = composite_dual_stream(world_img, face_img, target_w=1280, target_h=720)
    final_thumbnail.save(args.output, format="PNG")

    total_elapsed = time.time() - t0
    print(f"[Complete] Master YouTube Thumbnail saved -> {args.output} ({total_elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
