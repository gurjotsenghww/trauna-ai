# Project: Tràuna AI — Pipeline Upgrade & Dual-Head Glitch Elimination

## Architecture
- **AI Core (`ai/`)**: Python diffusion pipeline using Hugging Face `diffusers`, PyTorch, and CUDA.
  - Models (`ai/models/`): SDXL Turbo / SDXL 1.0 (`sdxl_turbo.py`), FLUX.1-schnell (`flux_schnell.py`), base interface (`base.py`).
  - Generation CLI (`ai/generate.py`): Entrypoint called by backend child process.
- **Backend (`backend/`)**: Node.js Express service.
  - Routes (`backend/routes/generate.js`): Accepts generation requests, coordinates prompt building and post-processing.
  - Services:
    - `promptEnhancer.js`: Enhances user text into optimized diffusion prompts.
    - `promptService.js`: Builds structured prompts from game templates.
    - `aiService.js`: Spawns Python generation subprocess.
    - `compositeService.js`: Sharp image processing, downscaling to 1280×720, text overlay.
- **Frontend (`frontend/`)**: Web UI for user prompt input, options, and thumbnail display.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Native Aspect-Ratio Bucketing | Generate strictly at native 16:9 buckets (1344×768 or 1024×576) divisible by 64 | M1 | Survey / R1 (DONE) |
| 2 | SDXL Micro-Conditioning | Inject `original_size`, `target_size`, and `crops_coords_top_left=(0,0)` | M1 | Survey / R1 (DONE) |
| 3 | VRAM Optimization & Offloading | `enable_sequential_cpu_offload()` + `vae.enable_tiling()` + `vae.enable_slicing()` keeping peak VRAM < 5.8 GB | M1 | Survey / R2, R4 (DONE) |
| 4 | Architecture Support & FLUX Fallback | Default to SDXL Turbo, safe fallback and quantization support for FLUX without RAM/disk crashes | M1 | Survey / R2 (DONE) |
| 5 | Full-Stack Invocation Decoupling | Decouple backend request resolution from final image size (pass native bucket to AI service) | M2 | Survey / R1 |
| 6 | Commercial Downscaling & Sizing | Downsample native master to exactly 1280×720 via Sharp Lanczos3 kernel | M2 | Survey / R1 |
| 7 | Prompt Enhancer Single-Subject Fix | Prevent unconditional streamer face injection for single-subject/non-streamer prompts | M3 | Survey / R3 |
| 8 | Three-Zone Spatial Composition Grammar | Structure prompts with Foreground Subject Zone, Depth Demarcation, Distant Background Zone | M3 | Survey / R3 |
| 9 | Anti-Hybrid & Anti-Duplication Negatives | Enforce negative prompt matrix preventing conjoined heads and human-creature feature bleeding | M3 | Survey / R3 |
| 10 | Automated E2E Benchmark Suite | Script verifying 1280×720 dimensions, single face / 0 duplicate heads, < 5.8 GB VRAM, clean skin, depth separation | M4 & E2E Track | Survey / Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Suite | Requirements-driven test suite (Tiers 1-4) & automated benchmark harness (`TEST_INFRA.md`, `TEST_READY.md`) | none | DONE (127/127 tests pass) |
| M1 | AI Generation Pipeline & VRAM Optimization | Update `sdxl_turbo.py`, `generate.py`, micro-conditioning, sequential offload, VAE tiling/slicing, peak VRAM < 5.8 GB | none | DONE (Gate Passed: Peak VRAM 3.01 GB, 0 leak, CLEAN) |
| M2 | Full-Stack Invocation & Composite Downsampling | Update `backend/routes/generate.js`, `aiService.js`, `compositeService.js` (Sharp Lanczos3 to 1280×720) | M1 | IN_PROGRESS |
| M3 | Spatial Prompt Composition & Glitch Elimination | Update `promptEnhancer.js` & `promptService.js` with Three-Zone Grammar & negative matrix | M1 | IN_PROGRESS |
| M4 | Final E2E Test Pass & Coverage Hardening | Run and pass 100% of E2E tests, run acceptance benchmarks, adversarial hardening | E2E, M1, M2, M3 | PLANNED |

## Interface Contracts
### `backend/services/aiService.js` ↔ `ai/generate.py`
- CLI Arguments:
  - `--prompt`: String (enhanced prompt with three-zone composition)
  - `--output`: String (path to output image file)
  - `--width`: Integer (native bucket: `1344` or `1024`, default `1344`)
  - `--height`: Integer (native bucket: `768` or `576`, default `768`)
  - `--steps`: Integer (e.g. `4` for turbo)
  - `--guidance_scale`: Float (default `1.5`)
  - `--negative_prompt`: String (anti-duplication negative tokens)
- Return: Process exit code 0 on success; JSON or log output indicating peak allocated VRAM.

### `ai/generate.py` ↔ `ai/models/sdxl_turbo.py`
- Function Signature: `generate(prompt, width=1344, height=768, num_inference_steps=4, guidance_scale=1.5, negative_prompt=None, seed=None) -> PIL.Image`
- Conditioning kwargs passed to `pipe()`:
  - `original_size`: `(width, height)`
  - `target_size`: `(width, height)`
  - `crops_coords_top_left`: `(0, 0)`

### `backend/routes/generate.js` ↔ `backend/services/compositeService.js`
- Function Signature: `compose({ generatedImagePath, primaryImagePath, thumbnailText, outputDir }) -> Promise<{ url, filename }>`
- Behavior:
  - Input: Native generated image at 1344×768 or 1024×576.
  - Output: High quality downsampled 1280×720 image via Sharp `fit: 'cover', position: 'centre', kernel: 'lanczos3'`.

## Code Layout
- `ai/models/sdxl_turbo.py`: SDXL Turbo wrapper with sequential CPU offload, VAE tiling/slicing, and micro-conditioning.
- `ai/models/base.py`: Base model class interface.
- `ai/generate.py`: Main CLI script for image generation.
- `ai/benchmark_e2e.py`: End-to-end benchmark and acceptance verification harness.
- `backend/routes/generate.js`: Express route handler for thumbnail generation.
- `backend/services/aiService.js`: Node service calling Python generation.
- `backend/services/promptEnhancer.js`: Context-aware prompt spatial composer.
- `backend/services/promptService.js`: Game-specific prompt builder.
- `backend/services/compositeService.js`: Sharp image resizing and compositing.
