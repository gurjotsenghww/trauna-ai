"""
Tràuna AI — ai/tests/helpers.py

Testing helpers and synthetic image generation for E2E testing framework.
Provides deterministic image oracles, skin analysis, Laplacian focus estimation,
and VRAM measurement utilities without external OpenCV dependencies.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def rgb_to_ycbcr(img_np: np.ndarray) -> np.ndarray:
    """
    Convert RGB uint8 numpy array [H, W, 3] to YCbCr [H, W, 3].
    ITU-R BT.601 conversion formula.
    """
    img_f = img_np.astype(np.float32)
    r = img_f[:, :, 0]
    g = img_f[:, :, 1]
    b = img_f[:, :, 2]

    y  =  0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b + 128.0
    cr =  0.500000 * r - 0.418688 * g - 0.081312 * b + 128.0

    return np.stack([y, cb, cr], axis=-1)


def detect_skin_mask(img_np: np.ndarray) -> np.ndarray:
    """
    Extract boolean mask of human skin pixels in image.
    Uses standard chromatic bounds in YCbCr:
    Cb in [77, 127] and Cr in [133, 173]
    Plus basic RGB relative constraints: R > G > B.
    """
    ycbcr = rgb_to_ycbcr(img_np)
    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    r = img_np[:, :, 0].astype(np.int32)
    g = img_np[:, :, 1].astype(np.int32)
    b = img_np[:, :, 2].astype(np.int32)

    skin = (
        (cr >= 130) & (cr <= 175) &
        (cb >= 75) & (cb <= 130) &
        (r > g) & (g > b) &
        (r - g >= 10) &
        (r > 60)
    )
    return skin


def calculate_laplacian_sharpness(img_gray_tensor: torch.Tensor) -> float:
    """
    Calculate the Laplacian variance of a 2D grayscale tensor [H, W] or [1, 1, H, W].
    Higher variance indicates sharper, higher-frequency edges.
    """
    if img_gray_tensor.ndim == 2:
        img_gray_tensor = img_gray_tensor.unsqueeze(0).unsqueeze(0)
    elif img_gray_tensor.ndim == 3:
        img_gray_tensor = img_gray_tensor.unsqueeze(0)

    # 3x3 discrete Laplacian operator
    kernel = torch.tensor([
        [0.0,  1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0,  1.0, 0.0]
    ], dtype=torch.float32, device=img_gray_tensor.device).view(1, 1, 3, 3)

    lap = F.conv2d(img_gray_tensor.float(), kernel, padding=1)
    variance = float(lap.var().item())
    return variance


def get_current_vram_gb() -> dict:
    """
    Returns current CUDA memory stats in gigabytes.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0, "max_reserved": 0.0, "device": "cpu"}

    return {
        "allocated": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved": torch.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "max_reserved": torch.cuda.max_memory_reserved() / (1024 ** 3),
        "device": torch.cuda.get_device_name(0),
    }


def create_synthetic_thumbnail(
    width: int = 1280,
    height: int = 720,
    face_count: int = 1,
    conjoined: bool = False,
    green_decay: bool = False,
    background_bokeh: bool = True,
    noise_level: float = 0.02,
) -> Image.Image:
    """
    Generates a deterministic synthetic thumbnail image for test oracle verification.
    """
    img = Image.new("RGB", (width, height), color=(30, 30, 45))
    draw = ImageDraw.Draw(img)

    # 1. Background scene
    if background_bokeh:
        # Smooth background gradient with slight noise (low high-frequency variance)
        for y in range(height):
            c = int(30 + 40 * (y / height))
            draw.line([(0, y), (width, y)], fill=(c, c // 2, c + 20))
    else:
        # High frequency noisy pattern
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

    # 2. Draw Face(s)
    # Human skin default: RGB (225, 165, 135)
    # Zombie/mutant decay default: RGB (90, 185, 75)
    skin_color = (90, 185, 75) if green_decay else (225, 165, 135)
    eye_white = (245, 245, 245)
    pupil_color = (20, 20, 25)
    mouth_color = (160, 40, 40)

    face_w = int(width * 0.22)
    face_h = int(height * 0.50)
    top_y = int(height * 0.22)

    if face_count == 1 and not conjoined:
        # Single foreground face on left third
        left_x = int(width * 0.12)
        right_x = left_x + face_w
        bottom_y = top_y + face_h

        # Head oval
        draw.ellipse([left_x, top_y, right_x, bottom_y], fill=skin_color, outline=(40, 20, 15), width=3)

        # Eyes with sharp pupils
        eye_y = top_y + int(face_h * 0.35)
        eye_w = int(face_w * 0.16)
        eye_h = int(face_h * 0.12)
        # Left eye
        draw.ellipse([left_x + int(face_w * 0.20), eye_y, left_x + int(face_w * 0.20) + eye_w, eye_y + eye_h], fill=eye_white)
        draw.ellipse([left_x + int(face_w * 0.25), eye_y + 2, left_x + int(face_w * 0.25) + eye_w // 2, eye_y + eye_h - 2], fill=pupil_color)
        # Right eye
        draw.ellipse([right_x - int(face_w * 0.20) - eye_w, eye_y, right_x - int(face_w * 0.20), eye_y + eye_h], fill=eye_white)
        draw.ellipse([right_x - int(face_w * 0.25) - eye_w // 2, eye_y + 2, right_x - int(face_w * 0.25), eye_y + eye_h - 2], fill=pupil_color)

        # Open mouth (reaction expression)
        mouth_y = top_y + int(face_h * 0.65)
        mouth_w = int(face_w * 0.35)
        mouth_h = int(face_h * 0.20)
        draw.ellipse([left_x + (face_w - mouth_w) // 2, mouth_y, left_x + (face_w + mouth_w) // 2, mouth_y + mouth_h], fill=mouth_color)

    elif conjoined:
        # Two conjoined heads sharing neck and torso
        left_x1 = int(width * 0.10)
        left_x2 = int(width * 0.22)
        # Overlapping ellipses
        draw.ellipse([left_x1, top_y, left_x1 + face_w, top_y + face_h], fill=skin_color, outline=(40, 20, 15), width=2)
        draw.ellipse([left_x2, top_y, left_x2 + face_w, top_y + face_h], fill=skin_color, outline=(40, 20, 15), width=2)

    elif face_count == 2:
        # Two distinct separate faces (e.g. left and center-right)
        pos1 = int(width * 0.10)
        pos2 = int(width * 0.55)
        for pos in [pos1, pos2]:
            draw.ellipse([pos, top_y, pos + face_w, top_y + face_h], fill=skin_color, outline=(40, 20, 15), width=3)
            # Eyes
            eye_y = top_y + int(face_h * 0.35)
            draw.ellipse([pos + int(face_w * 0.2), eye_y, pos + int(face_w * 0.4), eye_y + 20], fill=eye_white)
            draw.ellipse([pos + int(face_w * 0.6), eye_y, pos + int(face_w * 0.8), eye_y + 20], fill=eye_white)

    # 3. Add subtle texture
    if noise_level > 0:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, noise_level * 255, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img
