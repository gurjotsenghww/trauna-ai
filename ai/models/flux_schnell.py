"""
Tràuna AI — models/flux_schnell.py

FLUX.1-schnell model backend.
  Repository: black-forest-labs/FLUX.1-schnell
  Framework:  Hugging Face Diffusers

Optimised for RTX 3050 6 GB VRAM:
  - enable_sequential_cpu_offload()  — layers offloaded to CPU when not in use
  - torch.float16                    — halves VRAM usage
  - Attention slice                  — lower peak VRAM during attention

If this model still OOMs, set TRAUNA_MODEL=sdxl_turbo in environment.
"""

import gc
import os
import shutil
from typing import Optional
from PIL import Image
import torch

from .base_model import BaseModel

MODEL_ID = 'black-forest-labs/FLUX.1-schnell'


def _clean_vram():
    """Perform garbage collection and empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_system_specs():
    """Query available RAM and disk space safely."""
    ram_gb = 0.0
    avail_ram_gb = 0.0
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_gb = vm.total / (1024 ** 3)
        avail_ram_gb = vm.available / (1024 ** 3)
    except Exception:
        pass

    cache_dir = os.path.expanduser(r"~\.cache\huggingface\hub")
    cache_drive = os.path.splitdrive(os.path.abspath(cache_dir))[0] or "C:"
    free_disk_gb = 0.0
    try:
        free_disk_gb = shutil.disk_usage(cache_drive).free / (1024 ** 3)
    except Exception:
        pass

    return {
        "ram_gb": ram_gb,
        "avail_ram_gb": avail_ram_gb,
        "free_disk_gb": free_disk_gb,
        "cache_dir": cache_dir,
    }


def _is_flux_cached():
    """Check if FLUX weights are already cached locally."""
    cache_dir = os.path.expanduser(r"~\.cache\huggingface\hub")
    folder_name = "models--" + MODEL_ID.replace("/", "--")
    model_dir = os.path.join(cache_dir, folder_name)
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if os.path.exists(snapshots_dir):
        snaps = os.listdir(snapshots_dir)
        return len(snaps) > 0
    return False


class FluxSchnellModel(BaseModel):
    """
    FLUX.1-schnell — fast few-step image generation.
    Includes safe hardware/auth pre-flight checks and automatic fallback
    to SDXL Turbo to prevent host OS memory exhaustion and process crashes
    on memory-constrained systems (e.g. 6 GB VRAM / < 32 GB RAM).
    """

    def __init__(self):
        self._pipe = None
        self._fallback = None
        self._is_fallback = False
        self._load()

    def _load(self):
        specs = _get_system_specs()
        cached = _is_flux_cached()
        has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        strict_mode = os.environ.get("TRAUNA_FLUX_STRICT", "0").lower() in ("1", "true")

        # Unquantized FLUX.1-schnell (12B parameters, ~34 GB FP16 weights)
        # requires minimum 32 GB system RAM and 35 GB free disk space.
        is_memory_safe = specs["ram_gb"] >= 30.0
        is_auth_safe = cached or has_token

        if not is_memory_safe or not is_auth_safe:
            guidance = (
                f"\n{'=' * 70}\n"
                f"[FLUX Safety Guardrail Triggered]\n"
                f"Unquantized FLUX.1-schnell (12B parameters, ~34 GB FP16) cannot run safely on this system:\n"
                f"  - System RAM: {specs['ram_gb']:.1f} GB total, {specs['avail_ram_gb']:.1f} GB available (32+ GB required)\n"
                f"  - GPU VRAM:   6.0 GB physical (Unquantized requires 4-bit quantization on 6GB GPUs)\n"
                f"  - Disk Space: {specs['free_disk_gb']:.1f} GB free on cache drive (35+ GB required)\n"
                f"  - Cached:     {'Yes' if cached else 'No'}\n"
                f"  - Auth Token: {'Found' if has_token else 'Missing (black-forest-labs/FLUX.1-schnell is gated)'}\n\n"
                f"Attempting to download and load unquantized FLUX will cause host OS memory exhaustion / crash.\n"
            )

            if strict_mode:
                guidance += "TRAUNA_FLUX_STRICT is enabled. Aborting without fallback.\n" + ('=' * 70)
                print(guidance, flush=True)
                raise RuntimeError(
                    f"FLUX.1-schnell requires 32 GB+ RAM and Hugging Face authentication. "
                    f"Detected {specs['ram_gb']:.1f} GB RAM. Use 4-bit quantized FLUX or set TRAUNA_MODEL=sdxl_turbo."
                )

            guidance += (
                f"[FLUX Fallback] Engaging SDXL Turbo backend (1344x768 native 16:9 bucket, ~3.01 GB peak VRAM).\n"
                f"{'=' * 70}"
            )
            print(guidance, flush=True)

            from .sdxl_turbo import SDXLTurboModel
            self._fallback = SDXLTurboModel()
            self._is_fallback = True
            return

        # Attempt to load FLUX if system resources and auth are verified
        try:
            from diffusers import FluxPipeline

            print(f'[FLUX] Loading {MODEL_ID} ...', flush=True)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype  = torch.float16 if device == 'cuda' else torch.float32
            print(f'[FLUX] Device: {device}  dtype: {dtype}', flush=True)

            pipe = FluxPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
            )

            if device == 'cuda':
                pipe.enable_sequential_cpu_offload()
                print('[FLUX] Sequential CPU offload enabled.', flush=True)
                if hasattr(pipe, 'vae') and pipe.vae is not None:
                    pipe.vae.enable_tiling()
                    pipe.vae.enable_slicing()
            else:
                pipe = pipe.to(device)

            self._pipe = pipe
            print('[FLUX] Model loaded successfully.', flush=True)

        except Exception as exc:
            print(f'[FLUX Load Error] Failed to load {MODEL_ID}: {exc}', flush=True)
            if strict_mode:
                raise

            print('[FLUX Fallback] Automatically falling back to SDXL Turbo ...', flush=True)
            from .sdxl_turbo import SDXLTurboModel
            self._fallback = SDXLTurboModel()
            self._is_fallback = True

    def generate(
        self,
        prompt: str,
        width: int = 1344,
        height: int = 768,
        num_inference_steps: int = 4,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        if self._is_fallback and self._fallback is not None:
            print('[FLUX -> SDXL Fallback] Generating via SDXL Turbo ...', flush=True)
            return self._fallback.generate(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                seed=seed,
            )

        _clean_vram()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        generator = None
        if seed is not None:
            generator = torch.Generator(device='cpu').manual_seed(seed)

        print(f'[FLUX] Generating {width}x{height} @ {num_inference_steps} steps ...', flush=True)

        try:
            result = self._pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                generator=generator,
            )
            image = result.images[0]

            if torch.cuda.is_available():
                peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"[FLUX] Peak VRAM: {peak_alloc_gb:.2f} GB allocated", flush=True)
                assert peak_alloc_gb < 5.8, f"VRAM ceiling breached: {peak_alloc_gb:.2f} GB >= 5.8 GB limit"

            return image
        finally:
            _clean_vram()

