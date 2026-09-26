import torch
from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

print("=" * 60)
print("Testing SDXL UNet LoRA memory on RTX 3050 6GB...")
print("=" * 60)

device = torch.device("cuda")
torch.cuda.empty_cache()

# 1. Load UNet in FP16
print("Loading UNet in FP16 from local cache...")
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/sdxl-turbo",
    subfolder="unet",
    torch_dtype=torch.float16,
    variant="fp16",
)
unet.enable_gradient_checkpointing()
print(f"UNet loaded. VRAM reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")

# 2. Add LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v"],
)
unet = get_peft_model(unet, lora_config)
unet.to(device)
print(f"LoRA attached. VRAM reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
unet.print_trainable_parameters()

# 3. 8-bit Optimizer
optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=1e-4)
print(f"8-bit AdamW initialized. VRAM reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")

# 4. Dummy forward and backward pass
sample = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=device)
timestep = torch.tensor([1], device=device)
encoder_hidden_states = torch.randn(1, 77, 2048, dtype=torch.float16, device=device)
added_cond_kwargs = {
    "text_embeds": torch.randn(1, 1280, dtype=torch.float16, device=device),
    "time_ids": torch.zeros(1, 6, dtype=torch.float16, device=device),
}

with torch.amp.autocast("cuda", dtype=torch.float16):
    out = unet(sample, timestep, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample
    loss = out.mean()

print(f"Forward pass SUCCESS! VRAM: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
loss.backward()
optimizer.step()
optimizer.zero_grad()
print(f"Backward pass SUCCESS! VRAM: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
print("\nSDXL LoRA training is 100% VIABLE on RTX 3050 6GB!")
