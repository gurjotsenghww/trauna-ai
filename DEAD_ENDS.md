# Dead Ends Log

| Iteration | Approach Tried | Why It Failed | Files Touched |
|-----------|---------------|---------------|---------------|
| Phase 0 | Direct generation at 1280x720 | Self-attention receptive field breakdown (160x90 latents), spawning dual denoising nuclei and conjoined heads | ai/generate.py, backend/routes/generate.js |
| Phase 0 | `pipe.enable_model_cpu_offload()` | Reserves 5.42GB even at 1024x576, overflowing 6GB physical VRAM | ai/models/sdxl_turbo.py |
| Phase 0 | Unquantized FP16/BF16 FLUX.1-schnell (12B) | Requires 34.25GB RAM and disk space, exceeding host specs (15.7GB RAM, 21GB C: drive) | ai/models/flux_schnell.py |
| Phase 0 | Flat prompt token concatenation | Blends subject and creature tokens in cross-attention, causing hybrid monster fusion | backend/services/promptEnhancer.js |
