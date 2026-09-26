#!/usr/bin/env python3
"""
Tràuna AI — SDXL Gaming Thumbnail LoRA Trainer
Fine-tunes a YouTube Gaming Thumbnail LoRA on local NVIDIA RTX 3050 6GB GPU.

Optimized for 6GB VRAM:
  - Phase 1: Pre-cache latents (VAE) and text embeddings (CLIP 1 & 2) to disk/RAM
  - Phase 2: Unload VAE & text encoders from VRAM (freeing ~3GB VRAM)
  - Phase 3: Train UNet with LoRA (r=4) using 8-bit AdamW + Gradient Checkpointing
  - Peak training VRAM: ~5.48 GB (fits comfortably in 6GB)
"""

import os
import sys
import glob
import time
import argparse
from typing import List, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# Ensure UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_ID = "stabilityai/sdxl-turbo"

def get_model_path(repo_id=MODEL_ID):
    hub_cache = os.path.expanduser(r"~\.cache\huggingface\hub")
    folder_name = "models--" + repo_id.replace("/", "--")
    snapshots_dir = os.path.join(hub_cache, folder_name, "snapshots")
    if os.path.exists(snapshots_dir):
        snaps = os.listdir(snapshots_dir)
        if snaps:
            return os.path.join(snapshots_dir, snaps[0])
    return repo_id

MODEL_PATH = get_model_path(MODEL_ID)
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache", "sdxl_precomputed"))


def precompute_dataset(data_dir: str, cache_dir: str, max_samples: int = None, resolution: int = 512):
    """Pre-computes and caches VAE latents and dual CLIP text embeddings to disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cache_dir, exist_ok=True)

    images_dir = os.path.join(data_dir, "images")
    captions_dir = os.path.join(data_dir, "captions")

    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_files.extend(glob.glob(os.path.join(images_dir, ext)))
    img_files = sorted(img_files)

    if max_samples:
        img_files = img_files[:max_samples]

    samples = []
    for img_path in img_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        cap_path = os.path.join(captions_dir, f"{base}.txt")
        if os.path.exists(cap_path):
            samples.append((base, img_path, cap_path))

    print(f"\n[Pre-Cache] Pre-computing latents & text embeddings for {len(samples)} samples...")

    # Check if already cached
    cache_index = os.path.join(cache_dir, "cache_index.pt")
    if os.path.exists(cache_index):
        try:
            cached_data = torch.load(cache_index)
            if len(cached_data) >= len(samples):
                print(f"[Pre-Cache] Found existing cache with {len(cached_data)} items. Skipping computation!")
                return cached_data
        except Exception:
            pass

    # Load Tokenizers & Encoders
    print(f"[Pre-Cache] Loading Text Encoders & VAE from: {MODEL_PATH}...")
    tok1 = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer", local_files_only=True)
    tok2 = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2", local_files_only=True)
    te1 = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float16, local_files_only=True).to(device)
    te2 = CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder="text_encoder_2", variant="fp16", torch_dtype=torch.float16, local_files_only=True).to(device)
    vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae", variant="fp16", local_files_only=True).to(device, dtype=torch.float32)

    te1.eval().requires_grad_(False)
    te2.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    cached_records = []
    t0 = time.time()
    BATCH_SIZE = 4

    for b_idx in range(0, len(samples), BATCH_SIZE):
        batch_slice = samples[b_idx : b_idx + BATCH_SIZE]
        batch_pixels = []
        batch_captions = []

        for base_id, img_path, cap_path in batch_slice:
            try:
                with open(cap_path, "r", encoding="utf-8") as f:
                    cap_text = f.read().strip()
                img = Image.open(img_path).convert("RGB")
                batch_pixels.append(transform(img))
                batch_captions.append(cap_text)
            except Exception as e:
                print(f"[Pre-Cache Warning] Skipping {base_id}: {e}")

        if not batch_pixels:
            continue

        pixel_batch = torch.stack(batch_pixels).to(device, dtype=torch.float32)

        with torch.no_grad():
            # 1. Encode VAE latents in FP32 to prevent SDXL overflow, then cast to FP16
            latents_batch = (vae.encode(pixel_batch).latent_dist.sample() * vae.config.scaling_factor).to(torch.float16)

            # 2. Encode Text 1
            in1 = tok1(batch_captions, max_length=77, padding="max_length", truncation=True, return_tensors="pt").to(device)
            h1 = te1(in1.input_ids, output_hidden_states=True).hidden_states[-2]

            # 3. Encode Text 2
            in2 = tok2(batch_captions, max_length=77, padding="max_length", truncation=True, return_tensors="pt").to(device)
            out2 = te2(in2.input_ids, output_hidden_states=True)
            h2 = out2.hidden_states[-2]
            pooled_batch = out2.text_embeds

            # Concatenate hidden states
            prompt_embeds_batch = torch.concat([h1, h2], dim=-1)

        # Store individual records on CPU
        for i in range(latents_batch.shape[0]):
            cached_records.append({
                "latent": latents_batch[i].cpu(),
                "prompt_embeds": prompt_embeds_batch[i].cpu(),
                "pooled_embeds": pooled_batch[i].cpu(),
            })

        processed_count = min(b_idx + BATCH_SIZE, len(samples))
        if processed_count % 40 == 0 or processed_count == len(samples):
            rate = processed_count / max(0.1, time.time() - t0)
            print(f"[Pre-Cache] Processed {processed_count:4d}/{len(samples)} ({rate:.1f} items/s)...", flush=True)

    # Save to disk
    torch.save(cached_records, cache_index)
    print(f"[Pre-Cache] Saved {len(cached_records)} pre-computed pairs to {cache_index}", flush=True)

    # Completely unload VAE and Text Encoders from GPU
    del tok1, tok2, te1, te2, vae
    torch.cuda.empty_cache()
    print("[Pre-Cache] VAE and Text Encoders unloaded from GPU VRAM!", flush=True)
    return cached_records


class PrecomputedDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Tràuna AI — SDXL Gaming Thumbnail LoRA Trainer")
    print(f"  Device:       {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Model ID:     {MODEL_ID}")
    print(f"  Data Dir:     {args.data_dir}")
    print(f"  Output Dir:   {args.output_dir}")
    print(f"  Max Steps:    {args.max_train_steps}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  LoRA Rank:    {args.lora_rank}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 1: Precompute Latents & Text Embeddings ─────────────────────
    cache_directory = args.cache_dir if args.cache_dir else CACHE_DIR
    records = precompute_dataset(
        data_dir=args.data_dir,
        cache_dir=cache_directory,
        max_samples=args.max_samples,
        resolution=args.resolution,
    )

    dataset = PrecomputedDataset(records)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ── Phase 2: Load Noise Scheduler ─────────────────────────────────────
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler", local_files_only=True)

    # ── Phase 3: Load UNet & Apply LoRA ───────────────────────────────────
    print(f"\n[Training Setup] Loading SDXL UNet from local cache in FP16 ({MODEL_PATH})...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_PATH,
        subfolder="unet",
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    unet.enable_gradient_checkpointing()

    global_step = 0
    if getattr(args, "resume_from_checkpoint", None) and os.path.exists(args.resume_from_checkpoint):
        print(f"\n[Training Setup] Resuming LoRA from checkpoint: {args.resume_from_checkpoint}...", flush=True)
        from peft import PeftModel
        unet = PeftModel.from_pretrained(unet, args.resume_from_checkpoint, is_trainable=True)
        base_name = os.path.basename(args.resume_from_checkpoint.rstrip("/\\"))
        if "step-" in base_name:
            try:
                global_step = int(base_name.split("step-")[1])
                print(f"[Training Setup] Resumed at step {global_step}/{args.max_train_steps}!", flush=True)
            except Exception:
                global_step = 0
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet = get_peft_model(unet, lora_config)

    unet.to(device)
    unet.print_trainable_parameters()

    # ── Phase 4: 8-bit AdamW Optimizer ────────────────────────────────────
    print("[Training Setup] Initializing 8-bit AdamW Optimizer...", flush=True)
    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )

    # ── Phase 5: Training Loop ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🚀 SDXL LoRA TRAINING COMMENCED!")
    print(f"  Target Steps: {args.max_train_steps} | Checkpoints every {args.save_steps} steps")
    print(f"  Current VRAM Reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB / 6.00 GB")
    print("=" * 70 + "\n")

    initial_step = global_step
    t_start = time.time()
    unet.train()

    # Fixed time_ids for 512x512 SDXL condition: (orig_h, orig_w, crop_y, crop_x, target_h, target_w)
    time_ids = torch.tensor([[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]], device=device, dtype=torch.float16)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            latents = batch["latent"].to(device, dtype=torch.float16)
            prompt_embeds = batch["prompt_embeds"].to(device, dtype=torch.float16)
            pooled_embeds = batch["pooled_embeds"].to(device, dtype=torch.float16)

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            added_cond_kwargs = {
                "text_embeds": pooled_embeds,
                "time_ids": time_ids.repeat(latents.shape[0], 1),
            }

            with torch.amp.autocast('cuda', dtype=torch.float16):
                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # SDXL Turbo: target is noise - latents or pure noise depending on scheduler
                target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % 10 == 0 or global_step == 1:
                elapsed = time.time() - t_start
                it_per_sec = max(global_step - initial_step, 1) / max(elapsed, 0.001)
                vram_gb = torch.cuda.memory_reserved() / (1024**3)
                print(
                    f"[Step {global_step:4d}/{args.max_train_steps}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Speed: {it_per_sec:.2f} it/s | "
                    f"VRAM: {vram_gb:.2f} GB | "
                    f"Elapsed: {elapsed:.1f}s",
                    flush=True,
                )

            # Save Checkpoint
            if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                ckpt_dir = os.path.join(args.output_dir, f"sdxl-lora-step-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                unet.save_pretrained(ckpt_dir)
                print(f"\n[Checkpoint Saved] -> {ckpt_dir}\n", flush=True)

            if global_step >= args.max_train_steps:
                break

    # Save final LoRA model
    final_dir = os.path.join(args.output_dir, "trauna_sdxl_gaming_lora")
    os.makedirs(final_dir, exist_ok=True)
    unet.save_pretrained(final_dir)

    total_time = (time.time() - t_start) / 60
    print("\n" + "=" * 70)
    print(f"  🎉 SDXL LoRA TRAINING COMPLETED in {total_time:.2f} minutes!")
    print(f"  Final LoRA Saved: {final_dir}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Tràuna AI — SDXL Gaming LoRA Trainer")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.abspath("dataset/training_ready"),
        help="Path to training directory containing images/ and captions/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.abspath("ai/models/lora"),
        help="Directory to save trained LoRA checkpoints",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom cache directory for precomputed latents & embeddings",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to saved LoRA checkpoint folder to resume training from",
    )
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution for training")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank (4 is optimal for 6GB VRAM)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
