# Tràuna AI

**Local AI-Powered Gaming YouTube Thumbnail Generator**

> Tràuna AI is a local-first tool that uses pretrained image-generation models to create professional YouTube gaming thumbnails on your own hardware.

---

## Quick Start

### 1. Check Prerequisites

- Node.js 18+
- Python 3.10+
- NVIDIA GPU with CUDA (RTX 3050 6 GB minimum tested)
- CUDA Toolkit 12.x

### 2. Set Up Python AI Environment

```bash
cd trauna-ai/ai
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Verify GPU and Environment

```bash
python check_gpu.py
```

Expected output:
```
PyTorch Version: 2.x.x
CUDA Available : True
CUDA Version   : 12.x
GPU [0]
  Name         : NVIDIA GeForce RTX 3050 ...
  VRAM Total   : 6.00 GB
```

### 4. Test Image Generation

```bash
python test_generation.py
```

This will download the model on first run (~25 GB for FLUX.1-schnell) and generate one test image.

> If FLUX OOMs your GPU, use the SDXL Turbo fallback:
> ```bash
> set TRAUNA_MODEL=sdxl_turbo
> python test_generation.py
> ```

### 5. Install Backend Dependencies

```bash
cd trauna-ai/backend
npm install
```

### 6. Start the Backend

```bash
node server.js
```

The server starts at `http://localhost:3001`.

### 7. Open the Frontend

The frontend is served automatically by the backend at `http://localhost:3001`.

Or open `frontend/index.html` directly in a browser for development.

---

## Project Structure

```
trauna-ai/
├── frontend/          # HTML/CSS/JS UI
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── backend/           # Node.js / Express API
│   ├── server.js
│   ├── routes/
│   │   └── generate.js
│   ├── services/
│   │   ├── promptService.js    # User input → AI prompt
│   │   ├── aiService.js        # Calls Python AI subprocess
│   │   └── compositeService.js # Image resize + text overlay
│   ├── uploads/               # Temporary uploaded files
│   └── outputs/               # Generated thumbnails
│
├── ai/                # Python AI layer
│   ├── generate.py             # CLI entry point
│   ├── check_gpu.py            # Environment diagnostic
│   ├── test_generation.py      # End-to-end model test
│   ├── requirements.txt
│   └── models/
│       ├── base_model.py       # Abstract interface
│       ├── flux_schnell.py     # FLUX.1-schnell backend
│       └── sdxl_turbo.py       # SDXL Turbo fallback
│
├── dataset/           # Future training data
│   ├── images/
│   ├── captions/
│   └── metadata/
│
└── assets/
    └── examples/
```

---

## Model Abstraction

All model backends implement `BaseModel` from `ai/models/base_model.py`:

```python
class BaseModel(ABC):
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        seed: int | None,
    ) -> PIL.Image.Image:
        ...
```

To switch models, set the environment variable:

```bash
# Use FLUX.1-schnell (default)
set TRAUNA_MODEL=flux_schnell

# Use SDXL Turbo (lighter, for low VRAM)
set TRAUNA_MODEL=sdxl_turbo
```

No other code changes are required.

---

## API Reference

### `POST /api/generate`

Accepts `multipart/form-data`:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `primaryImage` | File | No | Main subject (face, avatar, character) |
| `additionalImages` | File[] | No | Additional images (up to 10) |
| `gameplayScreenshot` | File | No | In-game screenshot for context |
| `game` | String | **Yes** | Game ID (see GAMES in app.js) |
| `videoDescription` | String | No | What the video is about |
| `thumbnailText` | String | No | Text to overlay on the thumbnail |

Response:
```json
{
  "success": true,
  "prompt": "<generated AI prompt>",
  "results": [
    { "url": "/outputs/thumbnail_<id>.png", "filename": "thumbnail_<id>.png" }
  ],
  "elapsed": "45.2s"
}
```

---

## VRAM Optimisations (RTX 3050 6 GB)

| Technique | Applied | Effect |
|-----------|---------|--------|
| `float16` | ✅ | Halves VRAM usage |
| Sequential CPU offload | ✅ | Loads layers one at a time |
| Smaller output resolution | ✅ | 1024×576 AI gen, upscaled to 1280×720 |
| Low step count | ✅ | 4 steps for schnell/turbo models |

---

## Supported Games

| ID | Label |
|----|-------|
| `minecraft` | Minecraft |
| `minecraft_hardcore` | Minecraft Hardcore |
| `fortnite` | Fortnite |
| `gta` | GTA |
| `valorant` | Valorant |
| `pubg` | PUBG |
| `call_of_duty` | Call of Duty |
| `apex_legends` | Apex Legends |
| `roblox` | Roblox |
| `among_us` | Among Us |
| `horror` | Horror Games |
| `other` | Other |

---

## Future Roadmap

- [ ] Multiple concurrent generation (queue system)
- [ ] Tràuna Gaming LoRA training on curated dataset
- [ ] Subject reference image inpainting
- [ ] Cloud inference option
- [ ] More game categories
- [ ] Batch generation (4 variations)
- [ ] Style presets

---

## Dataset Preparation

See `dataset/README.md` for the expected format when curating training data for a future Tràuna Gaming LoRA.

> ⚠️ Do not scrape copyrighted thumbnails. Use only legally usable, properly licensed images.

---

## Success Checklist

- [ ] `check_gpu.py` detects RTX 3050 with CUDA
- [ ] `test_generation.py` generates one image
- [ ] `npm install` succeeds in `backend/`
- [ ] `node server.js` starts at port 3001
- [ ] Frontend loads at `http://localhost:3001`
- [ ] Upload + game select + generate produces a 1280×720 PNG
