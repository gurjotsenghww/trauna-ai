# Tràuna AI — Dataset Format

This directory is reserved for future training data for the **Tràuna Gaming LoRA**.

> ⚠️ Do NOT train anything yet. This is documentation only.

---

## Directory Structure

```
dataset/
├── images/          # Source thumbnail images (PNG or JPG, 1280×720)
├── captions/        # Plain-text caption files, one per image
│   └── <image_id>.txt
└── metadata/        # Structured JSON metadata, one per image
    └── <image_id>.json
```

---

## Image Requirements

- Format: PNG or JPG
- Resolution: 1280×720 (16:9)
- Filename: `<uuid>.png` or `<uuid>.jpg`
- Only use legally usable, properly licensed images
- No copyrighted third-party thumbnails without permission

---

## Caption Format

Each image in `images/` must have a corresponding `.txt` in `captions/` with the same base filename.

**Example** — `captions/abc123.txt`:
```
YouTube gaming thumbnail, excited streamer face close-up, Minecraft survival world background, dramatic lighting, golden hour, blocky pixelated environment, bold white text reading "100 DAYS", strong visual hierarchy, high contrast, vibrant colours
```

Caption guidelines:
- Describe the image factually and specifically
- Include subject, environment, lighting, atmosphere, text, composition
- Write in a style suitable for image-generation training (FLUX / SDXL)
- 50–200 words recommended

---

## Metadata Format

Each image should have a corresponding `.json` in `metadata/` with the same base filename.

**Schema** — `metadata/abc123.json`:
```json
{
  "id":              "abc123",
  "filename":        "abc123.png",
  "game":            "minecraft",
  "genre":           "survival",
  "subject":         "streamer_face",
  "composition":     "close_up_face_left_background_right",
  "camera_angle":    "low_angle_looking_up",
  "lighting":        "golden_hour_dramatic",
  "emotion":         "excited",
  "text_position":   "bottom_centre",
  "thumbnail_style": "reaction_face",
  "thumbnail_text":  "100 DAYS",
  "resolution":      "1280x720",
  "source":          "original",
  "license":         "cc0",
  "created_at":      "2026-09-24"
}
```

---

## Metadata Field Definitions

| Field | Type | Values |
|-------|------|--------|
| `game` | string | `minecraft`, `fortnite`, `gta`, `valorant`, `pubg`, `call_of_duty`, `apex_legends`, `roblox`, `among_us`, `horror`, `other` |
| `genre` | string | `survival`, `battle_royale`, `horror`, `open_world`, `fps`, `simulation`, `adventure`, `other` |
| `subject` | string | `streamer_face`, `avatar`, `character`, `scene_only`, `split_face_scene` |
| `composition` | string | `face_left_scene_right`, `face_right_scene_left`, `face_centre`, `full_scene`, `split_screen` |
| `camera_angle` | string | `low_angle`, `eye_level`, `overhead`, `close_up`, `extreme_close_up` |
| `lighting` | string | `dramatic`, `golden_hour`, `neon`, `horror_dark`, `bright_daylight`, `fire_glow`, `storm` |
| `emotion` | string | `excited`, `scared`, `shocked`, `confident`, `angry`, `happy`, `intense`, `neutral` |
| `text_position` | string | `top_left`, `top_centre`, `top_right`, `bottom_left`, `bottom_centre`, `bottom_right`, `none` |
| `thumbnail_style` | string | `reaction_face`, `cinematic_scene`, `split_screen`, `meme_style`, `minimal` |
| `source` | string | `original`, `commissioned`, `cc0`, `fair_use` |
| `license` | string | `cc0`, `cc_by`, `original`, `proprietary` |

---

## Training Notes (Future)

When training data is ready:

1. Minimum recommended dataset: 200+ high-quality images per style
2. Target model: LoRA on FLUX.1-dev or SDXL
3. Training framework: `diffusers` + `peft` or `kohya_ss`
4. Trigger word (proposed): `TRAUNA_GAMING_THUMB`
5. Training hardware: RTX 3050 with gradient checkpointing + xformers

> Do NOT begin training until the local prototype is validated and a sufficient dataset has been legally curated.
