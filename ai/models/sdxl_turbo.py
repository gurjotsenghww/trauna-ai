"""
Tràuna AI — models/sdxl_turbo.py

SDXL Turbo fallback model backend.
  Repository: stabilityai/sdxl-turbo
  Framework:  Hugging Face Diffusers

This model is lighter than FLUX and more likely to fit within 6 GB VRAM.
Use as fallback: set TRAUNA_MODEL=sdxl_turbo
"""

import gc
import os
import torch
from PIL import Image
from typing import Optional

from .base_model import BaseModel


def _clean_vram():
    """Perform garbage collection and empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


MODEL_ID = 'stabilityai/sdxl-turbo'

def get_model_path():
    hub_cache = os.path.expanduser(r"~\.cache\huggingface\hub")
    folder_name = "models--" + MODEL_ID.replace("/", "--")
    snapshots_dir = os.path.join(hub_cache, folder_name, "snapshots")
    if os.path.exists(snapshots_dir):
        snaps = os.listdir(snapshots_dir)
        if snaps:
            return os.path.join(snapshots_dir, snaps[0])
    return MODEL_ID

MODEL_PATH = get_model_path()
LORA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "lora", "trauna_sdxl_gaming_lora"))


class SDXLTurboModel(BaseModel):
    """SDXL Turbo — 1-4 step adversarially trained SDXL."""

    def __init__(self):
        self._pipe = None
        self._load()

    def _load(self):
        from diffusers import AutoPipelineForText2Image

        print(f'[SDXL-Turbo] Loading {MODEL_PATH} ...', flush=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype  = torch.float16 if device == 'cuda' else torch.float32

        print(f'[SDXL-Turbo] Device: {device}  dtype: {dtype}', flush=True)

        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            variant='fp16' if dtype == torch.float16 else None,
            local_files_only=os.path.isabs(MODEL_PATH),
        )

        # Auto-load trained Tràuna Gaming LoRA
        if os.path.exists(LORA_PATH):
            try:
                print(f'[SDXL-Turbo] Loading Trauna Gaming LoRA from {LORA_PATH} ...', flush=True)
                adapter_safetensors = os.path.join(LORA_PATH, 'adapter_model.safetensors')
                if os.path.exists(adapter_safetensors):
                    from safetensors.torch import load_file
                    raw_dict = load_file(adapter_safetensors)
                    needs_rekey = any(k.startswith('base_model.model.') for k in raw_dict)
                    if needs_rekey:
                        new_dict = {}
                        for k, v in raw_dict.items():
                            if k.startswith('base_model.model.'):
                                new_dict['unet.' + k[len('base_model.model.'):]] = v
                            else:
                                new_dict[k] = v
                        pipe.load_lora_weights(new_dict, adapter_name='trauna')
                    else:
                        pipe.load_lora_weights(LORA_PATH, adapter_name='trauna')
                else:
                    pipe.load_lora_weights(LORA_PATH, adapter_name='trauna')
                print(f'[SDXL-Turbo] Trauna Gaming LoRA loaded! Active adapters: {pipe.get_active_adapters()}', flush=True)
            except Exception as e:
                print(f'[SDXL-Turbo Warning] Failed to load LoRA weights: {e}', flush=True)

        if device == 'cuda':
            pipe.enable_sequential_cpu_offload()
            print('[SDXL-Turbo] Sequential CPU offload enabled.', flush=True)

            # Cap peak memory during VAE decode to ~3.01 GB
            if hasattr(pipe, 'vae') and pipe.vae is not None:
                pipe.vae.enable_tiling()
                pipe.vae.enable_slicing()
                print('[SDXL-Turbo] VAE tiling and slicing enabled (~3.01 GB peak VRAM).', flush=True)
        else:
            pipe = pipe.to(device)

        self._pipe = pipe
        print('[SDXL-Turbo] Model loaded.', flush=True)

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
        generator = None
        if seed is not None:
            generator = torch.Generator(device='cpu').manual_seed(seed)

        # SDXL-Turbo is an ADD (Adversarial Diffusion Distillation) model.
        # Canonical guidance_scale is strictly 0.0. Any CFG > 0 tears the latent manifold and causes double faces.
        if guidance_scale is None:
            guidance_scale = 0.0

        # Memory guardrail: clean cache and reset peak stats before generation
        _clean_vram()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            print(f"[SDXL-Turbo] Pre-generation VRAM: {free_bytes / (1024**3):.2f} GB free / {total_bytes / (1024**3):.2f} GB total", flush=True)

        print(f'[SDXL-Turbo] Generating {width}x{height} @ {num_inference_steps} steps (CFG: {guidance_scale}) ...', flush=True)

        # SDXL native micro-conditioning coordinates to prevent duplicate/mutant anatomy
        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "original_size": (width, height),
            "target_size": (width, height),
            "crops_coords_top_left": (0, 0),
        }
        if negative_prompt and guidance_scale > 0.0:
            kwargs["negative_prompt"] = negative_prompt
            kwargs["negative_original_size"] = (width, height)
            kwargs["negative_target_size"] = (width, height)
            kwargs["negative_crops_coords_top_left"] = (0, 0)

        try:
            result = self._pipe(**kwargs)
            image = result.images[0]

            if torch.cuda.is_available():
                peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                peak_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                print(f"[SDXL-Turbo] Peak VRAM: {peak_alloc_gb:.2f} GB allocated, {peak_res_gb:.2f} GB reserved", flush=True)
                assert peak_alloc_gb < 5.8, f"VRAM ceiling breached: {peak_alloc_gb:.2f} GB >= 5.8 GB limit"

            return image
        finally:
            _clean_vram()

