import torch
import peft
import diffusers
import bitsandbytes as bnb

print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
print("PEFT:", peft.__version__)
print("Diffusers:", diffusers.__version__)
print("BitsAndBytes:", bnb.__version__)
print("All training modules ready!")
