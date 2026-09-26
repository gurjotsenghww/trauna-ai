#!/usr/bin/env python3
"""
Tràuna AI — LoRA Training Pipeline
Fine-tunes a YouTube Gaming Thumbnail LoRA on local NVIDIA GPU (RTX 3050 6GB).

Optimized for 6GB VRAM:
  - 8-bit AdamW optimizer (bitsandbytes)
  - Gradient Checkpointing enabled on UNet
  - FP16 mixed precision
  - Batch size 1 with gradient accumulation
  - LoRA rank=8, alpha=16 on CrossAttention modules
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
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# Ensure UTF-8 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class ThumbnailDataset(Dataset):
    """Loads paired thumbnail images and captions from dataset/training_ready."""

    def __init__(self, data_dir: str, size: Tuple[int, int] = (512, 512), max_samples: int = None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.captions_dir = os.path.join(data_dir, "captions")

        img_files = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))
        if max_samples:
            img_files = img_files[:max_samples]

        self.samples = []
        for img_path in img_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            cap_path = os.path.join(self.captions_dir, f"{base}.txt")
            if os.path.exists(cap_path):
                self.samples.append((img_path, cap_path))

        print(f"[Dataset] Loaded {len(self.samples)} valid image-caption pairs from {data_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cap_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        with open(cap_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        return {"pixel_values": pixel_values, "caption": caption}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Tràuna AI — Gaming Thumbnail LoRA Trainer")
    print(f"  Device:       {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Base Model:   {args.base_model}")
    print(f"  Data Dir:     {args.data_dir}")
    print(f"  Output Dir:   {args.output_dir}")
    print(f"  Max Steps:    {args.max_train_steps}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  LoRA Rank:    {args.lora_rank}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Tokenizer & Text Encoder
    print("[1/5] Loading Text Encoder & Tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.base_model, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    # 2. Load VAE
    print("[2/5] Loading VAE...", flush=True)
    vae = AutoencoderKL.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)

    # 3. Load Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    # 4. Load UNet & Apply PEFT LoRA
    print("[3/5] Loading UNet & applying LoRA adapter...", flush=True)
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet", torch_dtype=torch.float16
    )

    # Enable Gradient Checkpointing for low VRAM
    unet.enable_gradient_checkpointing()

    # Configure LoRA on Cross-Attention Layers
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    unet.to(device)

    # 5. Optimizer (8-bit AdamW for low VRAM)
    print("[4/5] Initializing 8-bit AdamW Optimizer...", flush=True)
    try:
        optimizer = bnb.optim.AdamW8bit(
            unet.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )
        print("  -> Using 8-bit AdamW (saving ~75% optimizer VRAM)")
    except Exception as e:
        print(f"  -> Falling back to standard AdamW: {e}")
        optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # 6. DataLoader
    print("[5/5] Loading Dataset...", flush=True)
    dataset = ThumbnailDataset(args.data_dir, size=(args.resolution, args.resolution), max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # 7. Training Loop
    print("\n" + "=" * 70)
    print("  🚀 TRAINING STARTED!")
    print(f"  Total Steps: {args.max_train_steps} | Checkpoints every: {args.save_steps} steps")
    print("=" * 70 + "\n")

    global_step = 0
    t_start = time.time()
    scaler = torch.amp.GradScaler('cuda')

    unet.train()

    while global_step < args.max_train_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            captions = batch["caption"]

            with torch.no_grad():
                # Convert images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Tokenize and encode captions
                inputs = tokenizer(
                    captions,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                encoder_hidden_states = text_encoder(inputs.input_ids)[0]

            # Sample noise to add to the latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % 10 == 0 or global_step == 1:
                elapsed = time.time() - t_start
                it_per_sec = global_step / elapsed
                print(
                    f"[Step {global_step:4d}/{args.max_train_steps}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Speed: {it_per_sec:.2f} it/s | "
                    f"Elapsed: {elapsed:.1f}s",
                    flush=True,
                )

            # Save Checkpoint
            if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                unet.save_pretrained(ckpt_dir)
                print(f"\n[Checkpoint Saved] -> {ckpt_dir}\n", flush=True)

            if global_step >= args.max_train_steps:
                break

    # Save final LoRA model
    final_dir = os.path.join(args.output_dir, "trauna_gaming_lora")
    os.makedirs(final_dir, exist_ok=True)
    unet.save_pretrained(final_dir)

    total_time = (time.time() - t_start) / 60
    print("\n" + "=" * 70)
    print(f"  🎉 TRAINING COMPLETE in {total_time:.2f} minutes!")
    print(f"  Final LoRA Saved: {final_dir}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Tràuna AI — LoRA Training Script")
    parser.add_argument(
        "--base_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model to fine-tune (e.g. runwayml/stable-diffusion-v1-5 or stabilityai/stable-diffusion-xl-base-1.0)",
    )
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
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution (512 for SD 1.5, 768/1024 for SDXL)")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
