#!/usr/bin/env python3
"""
Tràuna AI — generate.py

Model-agnostic image generation entry point.
Called by the Node.js backend via subprocess.

Usage:
  python generate.py --prompt "..." --output out.png --width 1024 --height 576 --steps 4
"""

import argparse
import sys
import os
import time
import gc
from typing import Optional
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Native Aspect-Ratio Buckets
# ---------------------------------------------------------------------------
# Canonical multi-aspect buckets divisible by 64:
#   - 'standard': 1344x768 (~1.03 MP, 16:9 / 1.75:1) — primary SDXL quality
#   - 'fast':     1024x576 (~0.59 MP, 16:9 / 1.778:1) — low-latency preview
# ---------------------------------------------------------------------------
BUCKET_PRESETS = {
    'standard': (1344, 768),
    'fast':     (1024, 576),
}

DEFAULT_WIDTH = 1344
DEFAULT_HEIGHT = 768
VRAM_CEILING_GB = 5.8
MODEL_BACKEND = os.environ.get('TRAUNA_MODEL', 'sdxl_turbo')


def get_model(backend: Optional[str] = None):
    """Load and return the model backend."""
    if backend is None:
        backend = os.environ.get('TRAUNA_MODEL', 'sdxl_turbo')

    if backend == 'flux_schnell':
        from models.flux_schnell import FluxSchnellModel
        return FluxSchnellModel()
    elif backend == 'sdxl_turbo':
        from models.sdxl_turbo import SDXLTurboModel
        return SDXLTurboModel()
    else:
        raise ValueError(f'Unknown model backend: {backend}')


def parse_args():
    parser = argparse.ArgumentParser(description='Tràuna AI — Image Generator')
    parser.add_argument('--prompt',           type=str,   required=True, help='Image generation prompt')
    parser.add_argument('--output',           type=str,   required=True, help='Output file path (.png)')
    parser.add_argument('--negative_prompt',  type=str,   default=None,  help='Negative prompt for quality suppression')
    parser.add_argument('--guidance_scale',   type=float, default=None,  help='CFG scale (e.g. 1.5)')
    parser.add_argument('--width',            type=int,   default=None,  help=f'Output width (default {DEFAULT_WIDTH} for standard 16:9)')
    parser.add_argument('--height',           type=int,   default=None,  help=f'Output height (default {DEFAULT_HEIGHT} for standard 16:9)')
    parser.add_argument('--bucket',           type=str,   default=None,  choices=['standard', 'fast'], help='Aspect ratio bucket: standard (1344x768) or fast (1024x576)')
    parser.add_argument('--steps',            type=int,   default=4,     help='Inference steps')
    parser.add_argument('--seed',             type=int,   default=None,  help='Random seed (optional)')
    parser.add_argument('--model',            type=str,   default=None,  choices=['sdxl_turbo', 'flux_schnell'], help='Model backend override')
    return parser.parse_args()


def resolve_resolution(args):
    """Resolve width and height from CLI args, bucket presets, and defaults."""
    if args.bucket == 'fast':
        width  = args.width if args.width is not None else BUCKET_PRESETS['fast'][0]
        height = args.height if args.height is not None else BUCKET_PRESETS['fast'][1]
    elif args.bucket == 'standard':
        width  = args.width if args.width is not None else BUCKET_PRESETS['standard'][0]
        height = args.height if args.height is not None else BUCKET_PRESETS['standard'][1]
    else:
        width  = args.width if args.width is not None else DEFAULT_WIDTH
        height = args.height if args.height is not None else DEFAULT_HEIGHT

    if (width, height) == BUCKET_PRESETS['standard']:
        bucket_label = 'standard 16:9 (1344x768, ~1.03 MP)'
    elif (width, height) == BUCKET_PRESETS['fast']:
        bucket_label = 'fast 16:9 (1024x576, ~0.59 MP)'
    else:
        bucket_label = f'custom ({width}x{height})'

    return width, height, bucket_label


def clean_vram():
    """Perform garbage collection and release cached PyTorch CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    width, height, bucket_label = resolve_resolution(args)
    backend = args.model or os.environ.get('TRAUNA_MODEL', 'sdxl_turbo')

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f'[Tràuna AI] Backend: {backend}', flush=True)
    print(f'[Tràuna AI] Prompt:  {args.prompt[:80]}...', flush=True)
    if args.negative_prompt:
        print(f'[Tràuna AI] Negative: {args.negative_prompt[:80]}...', flush=True)
    print(f'[Tràuna AI] Bucket:  {bucket_label}', flush=True)
    print(f'[Tràuna AI] Size:    {width}x{height}', flush=True)
    print(f'[Tràuna AI] Steps:   {args.steps}', flush=True)

    if (width, height) == (1280, 720):
        print('[Tràuna AI Notice] 1280x720 is non-canonical for SDXL UNet. Recommended native buckets are standard (1344x768) or fast (1024x576), downscaled to 1280x720 in post-processing.', flush=True)

    # Pre-generation memory cleanup & baseline stats
    clean_vram()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f'[Tràuna AI] Initial VRAM: {free_bytes / (1024**3):.2f} GB free / {total_bytes / (1024**3):.2f} GB total', flush=True)

    t0 = time.time()

    model = get_model(backend)
    image = model.generate(
        prompt=args.prompt,
        width=width,
        height=height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )

    image.save(args.output)
    elapsed = time.time() - t0

    # Post-generation VRAM tracking and ceiling assertion
    if torch.cuda.is_available():
        peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_res_gb   = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f'[Tràuna AI] Peak VRAM Allocated: {peak_alloc_gb:.2f} GB', flush=True)
        print(f'[Tràuna AI] Peak VRAM Reserved:  {peak_res_gb:.2f} GB', flush=True)

        assert peak_alloc_gb < VRAM_CEILING_GB, (
            f"VRAM budget exceeded: Peak allocated VRAM {peak_alloc_gb:.2f} GB >= {VRAM_CEILING_GB} GB ceiling!"
        )
        print(f'[Tràuna AI] VRAM Safety Check: PASSED ({peak_alloc_gb:.2f} GB < {VRAM_CEILING_GB} GB)', flush=True)

    print(f'[Tràuna AI] Saved:   {args.output}', flush=True)
    print(f'[Tràuna AI] Time:    {elapsed:.1f}s', flush=True)
    print(f'[Tràuna AI] Status:  SUCCESS (Size: {image.width}x{image.height})', flush=True)

    clean_vram()


if __name__ == '__main__':
    main()

