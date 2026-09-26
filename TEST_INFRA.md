# Tràuna AI — Test Infrastructure & Methodology Specification (TEST_INFRA)

## 1. Test Philosophy: Opaque-Box & Requirement-Driven

The Tràuna AI testing architecture adheres strictly to an **opaque-box (black-box), requirement-driven verification philosophy**. Under this philosophy:
- **Contract Integrity Over Implementation**: Tests assert against observable behaviors, interface contracts, output artifacts, performance bounds, and stability invariants rather than internal code branching. Swapping an underlying model, adjusting layer offloading, or refactoring the Node.js/Python IPC bridge must not break test validity as long as the interface contracts and quality guarantees are upheld.
- **Authoritative Requirements Alignment**: Every test case traces directly to requirements defined in `PROJECT.md` and `ORIGINAL_REQUEST.md`:
  - **R1**: Native Aspect-Ratio Bucketing (1344×768 or 1024×576) & Elimination of Dual-Head / Conjoined-Subject Glitches.
  - **R2**: High-Fidelity Model Architecture Support (SDXL Turbo / FLUX) with graceful fallback and offloading.
  - **R3**: Context-Aware Prompt Spatial Composition (Foreground Subject Zone vs. Background Scene & Threat Zone).
  - **R4**: VRAM-Safe Execution Constraint (Peak allocated memory strictly < 5.8 GB on 6GB VRAM GPUs).
- **Automated Oracle & Metric-Driven Acceptance**: Tests compute explicit deterministic and statistical quality metrics:
  - Precise dimension checking (1280×720 master output, 1344×768 / 1024×576 native model output).
  - Subject count & conjoined head detection via automated facial bounding box analysis (asserting single-subject prompts yield exactly 1 primary face and 0 duplicate/conjoined heads).
  - Real-time CUDA peak allocated VRAM tracking (`torch.cuda.max_memory_allocated() < 5.8 GB`).
  - Skin & eye quality checks (absence of monster texture/decay color bleeding on human face bounding boxes).
  - Background optical separation (high-frequency sharpness differential between foreground subject and background bokeh).
  - Stability invariants (zero CUDA out-of-memory errors, zero process crashes).

---

## 2. 4-Tier Testing Methodology

The test suite is structured across four progressive testing tiers, spanning unit validation to end-to-end production thumbnail generation:

```
┌────────────────────────────────────────────────────────────────────────┐
│               TIER 4: REAL-WORLD ACCEPTANCE SCENARIOS                  │
│       5 Curated Production Benchmarks • VRAM < 5.8GB • Dual-Head 0     │
├────────────────────────────────────────────────────────────────────────┤
│               TIER 3: PAIRWISE COMBINATORIAL TESTING                  │
│       Resolution × Guidance Scale × Steps × Neg Prompt × Backend       │
├────────────────────────────────────────────────────────────────────────┤
│               TIER 2: BOUNDARY VALUE ANALYSIS (BVA)                    │
│       Limits: 64px-1920px • 1-50 Steps • 0.0-20.0 CFG • 0-5.8GB VRAM   │
├────────────────────────────────────────────────────────────────────────┤
│               TIER 1: CATEGORY-PARTITION TESTING                       │
│       Functional Equivalence Classes • >= 5 Test Cases Per Feature     │
└────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Category-Partition Testing (Functional Equivalence Classes)
- **Objective**: Divide each feature's input and operational domain into mutually exclusive equivalence classes. Verify that representative valid and invalid partitions behave strictly according to specification.
- **Coverage Threshold**: **At least 5 test cases per feature** across all 10 features in the inventory (Minimum 50 test cases total).
- **Categories Partitioned**:
  - *Prompt Syntax*: Standard gaming prompt, single-subject streamer, animal subject, horror/creature prompt, multi-entity prompt, empty prompt, whitespace-only, special/unicode characters, prompt with missing trigger tokens.
  - *Aspect-Ratio Buckets*: Native 16:9 bucket A (`1344×768`), native 16:9 bucket B (`1024×576`), non-native direct target (`1280×720`), non-standard bucket (`512×512`, `768×768`).
  - *Model Selection*: `sdxl_turbo` (default), `flux_schnell`, invalid backend name.
  - *Negative Prompting*: Anti-duplication negative matrix provided, empty string, `None`, custom user negative prompt.
  - *Compositing Modes*: Sharp Lanczos3 cover resizing, text overlay presence/absence, primary image flag.

### Tier 2: Boundary Value Analysis (BVA)
- **Objective**: Stress parameters at their exact mathematical, architectural, and memory boundaries (minimum, nominal, maximum, and out-of-bounds).
- **Coverage Threshold**: **At least 5 test cases per feature** testing upper, lower, and off-nominal boundaries.
- **Boundaries Evaluated**:
  - *Image Resolution*:
    - Lower boundary: `64×64` (minimum VAE tile dimension).
    - Speed bucket boundary: `1024×576` (standard SDXL 16:9 bucket).
    - Production master bucket: `1344×768` (high-fidelity 16:9 bucket).
    - Standard YouTube display output: `1280×720` (downscaled master).
    - Extreme upper boundary: `1920×1080` (verifying graceful rejection or memory safety).
    - Divisibility constraint: Modulo 64 alignment (`1344 % 64 == 0`, `768 % 64 == 0`, `1024 % 64 == 0`, `576 % 64 == 0`).
  - *Denoising Steps*:
    - Minimum step: `1` step (SDXL Turbo lightning pass).
    - Recommended nominal: `4` steps (SDXL Turbo standard).
    - Quality mode: `6` steps (Ultra preset).
    - High boundary: `10` steps.
    - Extreme boundary: `50` steps (asserting loop stability).
  - *Guidance Scale (CFG)*:
    - Zero guidance: `0.0` (pure speed 1-pass for Turbo / FLUX).
    - Low guidance: `1.0`.
    - Nominal guidance: `1.5` (standard SDXL Turbo with negative prompt).
    - High guidance: `7.5` (standard SD 1.5/SDXL base).
    - Extreme boundary: `20.0` (asserting dynamic range clipping prevention).
  - *VRAM Allocation*:
    - Safe operational range: `< 4.5 GB`.
    - Warning threshold: `4.5 GB - 5.5 GB`.
    - Hard ceiling: Strictly `< 5.8 GB` peak allocated memory on 6.00 GB GPU hardware.
  - *Prompt Length*:
    - Lower boundary: `0` characters (empty).
    - Minimal boundary: `1` word.
    - Nominal boundary: `50 - 150` characters.
    - Long boundary: `500` characters (CLIP token truncation test at 77 tokens).

### Tier 3: Pairwise Combinatorial Testing
- **Objective**: Systematically test interactions between key operational parameters to uncover edge-case defects that emerge only when multiple features interact.
- **Coverage Combinations**:
  - `(Model Backend)` × `(Resolution Bucket)` × `(Guidance Scale)` × `(Negative Prompt State)` × `(Step Count)`
  - Key interaction pairs evaluated:
    1. `sdxl_turbo` × `1344×768` × `guidance_scale=1.5` × `negative_prompt=Active` × `steps=4` (Primary Production Mode)
    2. `sdxl_turbo` × `1024×576` × `guidance_scale=0.0` × `negative_prompt=None` × `steps=1` (Fastest Mode)
    3. `sdxl_turbo` × `1280×720` × `guidance_scale=1.5` × `negative_prompt=Active` × `steps=4` (Non-bucket compatibility test)
    4. `sdxl_turbo` × `1344×768` × `guidance_scale=0.0` × `negative_prompt=Active` × `steps=4` (CFG 0 with negative prompt suppression)
    5. `sdxl_turbo` × `1024×576` × `guidance_scale=2.0` × `negative_prompt=Active` × `steps=6` (Ultra preset)
    6. `flux_schnell` × `1024×576` × `guidance_scale=0.0` × `negative_prompt=None` × `steps=4` (FLUX default configuration)

### Tier 4: Real-World Acceptance Scenarios
- **Objective**: Execute end-to-end commercial thumbnail generation scenarios representing the user's primary YouTube gaming production workloads, verifying all acceptance criteria.
- **Core Curated Scenarios**:
  1. **Scenario 1 — Single Streamer Subject (Dual-Head Elimination Test)**:
     - *Prompt*: `"traunathumb, professional youtube gaming thumbnail, in foreground on left, extreme close-up of a single gaming streamer with screaming reaction face, wearing headphones; in background, colorful modern gaming room; depth of field separation, 16:9"`
     - *Acceptance Invariant*: Output conforms to 1280×720; exactly 1 primary face detected; 0 secondary/conjoined heads; peak VRAM < 5.8 GB.
  2. **Scenario 2 — Single Animal Subject (Anatomy & Non-Human Glitch Test)**:
     - *Prompt*: `"traunathumb, professional youtube gaming thumbnail, in foreground, extreme close-up of a single cat with mouth wide open in shocked expression, realistic fur, sharp eyes; blurred living room background; 16:9"`
     - *Acceptance Invariant*: Single coherent animal subject; 0 duplicate conjoined heads; clean mouth anatomy; 1280×720 dimensions.
  3. **Scenario 3 — Streamer + Minecraft Threat (Spatial Separation & Anti-Bleed Test)**:
     - *Prompt*: `"traunathumb, professional youtube gaming thumbnail, in foreground on left, close-up portrait of single human streamer in terror; in far background on right, a giant blocky mutant creeper in glowing Minecraft nether; optical bokeh separation, 16:9"`
     - *Acceptance Invariant*: Exactly 1 human face; zero creeper green texture bleeding onto streamer face (`decay_ratio < 0.05`); sharp foreground vs blurred background.
  4. **Scenario 4 — Streamer + Horror Entity (Skin/Eye Quality & Anti-Fusion Test)**:
     - *Prompt*: `"traunathumb, professional youtube gaming thumbnail, in foreground on left, terrified streamer face with wide eyes and open mouth, clean skin, natural eyes; in dark distant background on right, zombie figure in foggy haunted hallway; strong depth separation, 16:9"`
     - *Acceptance Invariant*: Exactly 1 human face; coherent eye pupils; zero zombie rot on streamer face; peak VRAM < 5.8 GB.
  5. **Scenario 5 — High-Speed Action / GTA (Scene Stability & Text Space Test)**:
     - *Prompt*: `"traunathumb, professional youtube gaming thumbnail, in foreground on left, laughing streamer face; in background on right, sports car flying through massive explosion in GTA Los Santos; vibrant rim light, 16:9"`
     - *Acceptance Invariant*: Sharp foreground subject; high-contrast background without hybrid vehicle fusion; zero CUDA OOMs.

---

## 3. Feature Inventory Mapped Across Tiers 1–4

| Feature ID | Feature Description | Tier 1 (Equivalence Partition) | Tier 2 (Boundary Value Analysis) | Tier 3 (Pairwise Combinations) | Tier 4 (Acceptance Benchmark) |
|---|---|---|---|---|---|
| **F-01** | Native Aspect-Ratio Bucketing | 1344×768 (B1), 1024×576 (B2), 1280×720 (direct), 512×512 (square), 768×1344 (portrait) | Lower: 64×64; Nominal: 1024×576; Upper: 1344×768; Modulo: 64-divisible check; Off-nominal: 1920×1080 | 1344×768 × CFG 1.5; 1024×576 × CFG 0.0; 1280×720 × Sharp resize | Scenario 1, 2, 3, 4, 5 (all verify final 1280×720 sizing) |
| **F-02** | SDXL Micro-Conditioning | `original_size`, `target_size`, `crops_coords_top_left` injected correctly vs omitted | Zero crop coords `(0,0)`; Target matching bucket size; Rescaling coordinates | Injected across all native bucket runs with SDXL Turbo | Micro-conditioning active in Scenarios 1–5 preventing dual-head drift |
| **F-03** | VRAM Optimization & Offload | Peak VRAM under normal inference, sequential CPU offload enabled, VAE tiling/slicing | Threshold: 4.5 GB nominal, 5.5 GB peak, 5.8 GB hard ceiling, 6.0 GB hardware limit | High steps (6) + native 1344×768 + CFG 1.5 + VAE decode | Monitored dynamically across all 5 acceptance runs (`< 5.8 GB`) |
| **F-04** | Architecture Support & Fallback | `sdxl_turbo` loaded, `flux_schnell` fallback, unknown backend error handling | Memory check under 1 step vs 4 steps vs 6 steps; disk cache validation | Switching `TRAUNA_MODEL` environment variable and verifying fallback | Verified on local RTX 3050 Laptop GPU hardware |
| **F-05** | Full-Stack Invocation Decoupling | CLI argument parsing: `--prompt`, `--output`, `--width`, `--height`, `--steps`, `--guidance_scale`, `--negative_prompt`, `--seed` | Empty prompt handling; negative steps rejected; negative width rejected; seed boundary: `0` to `2^32-1` | Node.js subprocess invocation contract matching Python CLI arguments | Invocation of Python CLI directly and via benchmark harness |
| **F-06** | Commercial Downscaling & Sizing | Sharp Lanczos3 cover resizing; aspect ratio preservation; JPG/PNG export | 1344×768 downscaled to 1280×720; 1024×576 upscaled/fitted to 1280×720 | Native master bucket paired with composite pipeline | Final output verified strictly at 1280×720 |
| **F-07** | Prompt Enhancer Single-Subject | Creator face prompt; non-streamer prompt; animal prompt; solo keyword enforcement | Single-subject keywords ("single solo human", "one person") vs multi-entity | Single subject + game world combinations | Scenario 1 (single streamer) and Scenario 2 (single cat) |
| **F-08** | Three-Zone Spatial Composition | Foreground zone, Depth demarcation zone, Distant background zone | Zone 1 character tokens; Zone 2 depth tokens; Zone 3 threat tokens | Spatial layout allocator tested against all game styles | Scenario 3 (Minecraft creeper), Scenario 4 (Zombie horror), Scenario 5 (GTA) |
| **F-09** | Anti-Hybrid & Anti-Duplication Negatives | Enforcing negative prompt tokens: conjoined twins, duplicate heads, hybrid monster | Default negative prompt vs enhanced negative prompt vs empty negative prompt | Negative prompt active with CFG > 0 vs CFG = 0 | Evaluated in Scenarios 1, 3, 4 (skin decay check and face count) |
| **F-10** | Automated E2E Benchmark Runner | CLI arguments, JSON summary export, Markdown report generation, face bounding box metrics | 0 faces detected (warning/error); 1 face (pass); >1 face (fail); VRAM ceiling exceedance (fail) | Benchmark execution across mock mode and live GPU mode | `ai/benchmark_e2e.py` executing all 5 scenarios |

---

## 4. Test Architecture & Directory Layout

The test suite is organized under `ai/tests/` and project root:

```
trauna-ai/
├── TEST_INFRA.md                   # This document: Test philosophy & methodology
├── TEST_READY.md                   # Execution report & readiness certificate
├── ai/
│   ├── benchmark_e2e.py            # Automated E2E benchmark & acceptance runner
│   ├── tests/
│   │   ├── __init__.py             # Test package initializer
│   │   ├── test_tier1_category.py  # Tier 1: Functional Equivalence Classes (>= 5 per feature)
│   │   ├── test_tier2_boundary.py  # Tier 2: Boundary Value Analysis (>= 5 per feature)
│   │   ├── test_tier3_pairwise.py  # Tier 3: Combinatorial & Contract Tests
│   │   ├── test_tier4_acceptance.py# Tier 4: Real-World Scenario Acceptance Runner
│   │   ├── test_face_detector.py   # Specialized unit tests for face/duplicate detection
│   │   └── helpers.py              # Test utilities, VRAM measurement, synthetic images
│   └── outputs/
│       └── benchmarks/             # Benchmark reports, JSON summaries, artifacts
```

---

## 5. Execution Commands & CLI Interfaces

### 5.1 Running the Python Unit & Contract Test Suite
All unit, boundary, and pairwise tests can be executed via Python's standard `unittest` framework using the dedicated virtual environment:

```bash
# Run all tests in the suite
& "D:/Tràuna AI/trauna-ai/ai/venv/Scripts/python.exe" -m unittest discover -s ai/tests -p "test_*.py" -v

# Run individual tiers
& "D:/Tràuna AI/trauna-ai/ai/venv/Scripts/python.exe" -m unittest ai/tests/test_tier1_category.py -v
& "D:/Tràuna AI/trauna-ai/ai/venv/Scripts/python.exe" -m unittest ai/tests/test_tier2_boundary.py -v
& "D:/Tràuna AI/trauna-ai/ai/venv/Scripts/python.exe" -m unittest ai/tests/test_tier3_pairwise.py -v
& "D:/Tràuna AI/trauna-ai/ai/tests/test_tier4_acceptance.py" -v
```

### 5.2 Running the E2E Acceptance Benchmark Runner (`ai/benchmark_e2e.py`)

The automated benchmark runner evaluates the acceptance criteria on the target hardware:

```bash
# Full live GPU acceptance benchmark on RTX 3050 Laptop GPU:
& "D:/Tràuna AI/trauna-ai/ai/venv/Scripts/python.exe" ai/benchmark_e2e.py \
    --model sdxl_turbo \
    --device cuda \
    --vram_ceiling_gb 5.8 \
    --output_dir ai/outputs/benchmarks

# Fast validation / dry-run mode (verifies metrics, face detector, and pipeline without full generation):
& "D:/Tràuna AI/trauna-ai/ai/venv/Scripts/python.exe" ai/benchmark_e2e.py \
    --mock \
    --output_dir ai/outputs/benchmarks

# Custom scenario evaluation:
& "D:/Tràuna AI/trauna-ai/ai/venv/Scripts/python.exe" ai/benchmark_e2e.py \
    --scenario 1 \
    --model sdxl_turbo \
    --device cuda
```

---

## 6. Acceptance Criteria & Pass/Fail Thresholds

| Metric | Measurement Technique | Target Threshold | Hard Failure Condition |
|---|---|---|---|
| **Dimension Check** | `image.size == (W, H)` | `1280×720` (master) or native `1344×768`/`1024×576` | Image width ≠ 1280 or height ≠ 720 on final thumb |
| **Subject Count (Faces)** | Face detector (Haar Cascade / Contours / DNN) | Exactly `1` face for single-subject prompts | `face_count == 0` (no subject) or `face_count >= 2` (duplicate/conjoined heads) |
| **Conjoined Head Anomaly** | Bounding box spatial proximity & aspect ratio | `conjoined_heads == 0` | Two overlapping or adjacent head bounding boxes |
| **Peak VRAM Allocation** | `torch.cuda.max_memory_allocated() / (1024**3)` | `< 5.0 GB` nominal | Peak allocated memory `≥ 5.8 GB` |
| **Peak VRAM Reserved** | `torch.cuda.max_memory_reserved() / (1024**3)` | `< 5.8 GB` | Peak reserved memory `≥ 6.0 GB` (exceeds card physical VRAM) |
| **Skin / Hybrid Decay Ratio** | Ratio of abnormal green/decay tint in face ROI: `mean(G > 1.15 * R)` | `< 0.05` (Clean skin rendering) | `decay_ratio ≥ 0.05` (creeper/zombie texture on human face) |
| **Depth Separation Ratio** | Sharpness of foreground face ROI vs background ROI: `Laplacian_var(FG) / Laplacian_var(BG)` | `> 1.2` (Bokeh / blur separation) | Sharpness ratio `≤ 1.0` (flat focus, no depth separation) |
| **Process Stability** | Subprocess exit code & CUDA exception catch | `Exit Code 0`, 0 exceptions | `torch.cuda.OutOfMemoryError` or process crash |

---

## 7. Reporting & Deliverables

Every benchmark execution outputs standardized machine-readable and human-readable reports in `ai/outputs/benchmarks/`:
1. `benchmark_summary.json`: Detailed JSON object containing hardware specs, CUDA version, per-test latency, peak VRAM allocated, peak VRAM reserved, face count, conjoined count, dimension check, and test status.
2. `benchmark_report.md`: Formatted Markdown report summarizing overall pass/fail status, acceptance criteria compliance matrix, and diagnostic logs.
3. Annotated test images (`scenario_1_annotated.png`, etc.) highlighting detected subject bounding boxes, facial regions, and background zones for visual auditing.
