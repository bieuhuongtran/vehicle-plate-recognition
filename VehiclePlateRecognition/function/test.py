"""
VehiclePlateRecognition.function.fake
-----------------------------------
Small utilities to generate synthetic license plate text and images and to
anonymize (mask/pixelate/overlay) plate regions in images.

USAGE & SAFETY
- This module is intended for benign testing, dataset augmentation, and
  privacy-preserving anonymization only.
- By default operations are performed in-memory and do NOT write files to disk
  unless you explicitly pass a `save_dir` to `generate_synthetic_dataset`.
- Do NOT use this module to facilitate illegal activity (forgery, evading
  law enforcement, etc.).

Dependencies: Pillow, numpy
"""

from __future__ import annotations

import io
import os
import random
import string
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

__all__ = [
    "generate_plate_text",
    "render_plate_image",
    "pixelate_region",
    "blur_region",
    "overlay_text_region",
    "anonymize_plate",
    "generate_synthetic_dataset",
]

# Common plate format tokens:
# L = letter, D = digit, A = alphanumeric
DEFAULT_FORMATS = ["LLL-DDDD", "LL-DDDD", "L-DDDD", "LLL-DDD"]


def _fill_token(tok: str) -> str:
    if tok == "L":
        return random.choice(string.ascii_uppercase)
    if tok == "D":
        return random.choice(string.digits)
    if tok == "A":
        return random.choice(string.ascii_uppercase + string.digits)
    return tok


def generate_plate_text(fmt: str = "LLL-DDDD") -> str:
    """Generate a synthetic license plate string from a simple format.

    Tokens:
      L = uppercase letter
      D = digit
      A = alphanumeric
      any other character is copied verbatim

    Example: "LL-DDDD" -> "AB-1234"
    """
    out = []
    for c in fmt:
        if c in {"L", "D", "A"}:
            out.append(_fill_token(c))
        else:
            out.append(c)
    return "".join(out)


def render_plate_image(
    text: str,
    size: Tuple[int, int] = (320, 96),
    font_path: Optional[str] = None,
    font_size: Optional[int] = None,
    bgcolor: Tuple[int, int, int] = (255, 255, 255),
    fg: Tuple[int, int, int] = (0, 0, 0),
    border: bool = True,
    noise: bool = False,
) -> Image.Image:
    """Render a simple rectangular image containing `text` (plate-like look).

    The function uses PIL. If `font_path` is not provided, the default PIL font
    is used. This is intended for synthetic data / testing only.
    """
    w, h = size
    img = Image.new("RGB", (w, h), color=bgcolor)
    draw = ImageDraw.Draw(img)

    # Choose font size heuristically if not provided
    if font_size is None:
        font_size = int(h * 0.5)

    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Measure text and center it
    tw, th = draw.textsize(text, font=font)
    x = (w - tw) // 2
    y = (h - th) // 2

    # Draw a subtle border to emulate plate frame
    if border:
        pad = int(h * 0.06)
        draw.rectangle((pad, pad, w - pad - 1, h - pad - 1), outline=fg)

    draw.text((x, y), text, font=font, fill=fg)

    if noise:
        arr = np.array(img).astype(np.int16)
        noise_level = int(4 + random.random() * 8)
        arr += np.random.randint(-noise_level, noise_level, size=arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def _clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def pixelate_region(image: Image.Image, bbox: Tuple[int, int, int, int], pixel_size: int = 10) -> Image.Image:
    """Pixelate a rectangular region in the image.

    bbox: (x1, y1, x2, y2)
    """
    img = image.copy()
    w, h = img.size
    x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
    region = img.crop((x1, y1, x2, y2))
    rw, rh = region.size
    if rw == 0 or rh == 0:
        return img
    # Downscale then upscale to pixelate
    region = region.resize((max(1, rw // pixel_size), max(1, rh // pixel_size)), resample=Image.NEAREST)
    region = region.resize((rw, rh), Image.NEAREST)
    img.paste(region, (x1, y1))
    return img


def blur_region(image: Image.Image, bbox: Tuple[int, int, int, int], radius: int = 8) -> Image.Image:
    img = image.copy()
    w, h = img.size
    x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
    region = img.crop((x1, y1, x2, y2))
    region = region.filter(ImageFilter.GaussianBlur(radius))
    img.paste(region, (x1, y1))
    return img


def overlay_text_region(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    text: str = "ANON",
    bgcolor: Tuple[int, int, int] = (0, 0, 0),
    fg: Tuple[int, int, int] = (255, 255, 255),
    font_path: Optional[str] = None,
) -> Image.Image:
    """Overlay a filled rectangle and place `text` centered inside it."""
    img = image.copy()
    w, h = img.size
    x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
    draw = ImageDraw.Draw(img)
    draw.rectangle((x1, y1, x2, y2), fill=bgcolor)

    # Choose font size based on bbox height
    font_size = max(10, int((y2 - y1) * 0.6))
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    tw, th = draw.textsize(text, font=font)
    tx = x1 + ((x2 - x1) - tw) // 2
    ty = y1 + ((y2 - y1) - th) // 2
    draw.text((tx, ty), text, fill=fg, font=font)
    return img


def anonymize_plate(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    method: str = "overlay",
    overlay_text: Optional[str] = "ANON",
    **method_kwargs,
) -> Image.Image:
    """Anonymize a plate in `image` at `bbox`.

    method: one of ['overlay', 'blur', 'pixelate']
    overlay_text: optional text to place when method='overlay'
    Additional kwargs are forwarded to the specific method.
    """
    method = method.lower()
    if method == "overlay":
        return overlay_text_region(image, bbox, text=overlay_text or "ANON", **method_kwargs)
    if method == "blur":
        return blur_region(image, bbox, **method_kwargs)
    if method == "pixelate":
        return pixelate_region(image, bbox, **method_kwargs)
    raise ValueError("Unknown anonymization method: %s" % method)


def generate_synthetic_dataset(
    n: int = 10,
    formats: Optional[List[str]] = None,
    size: Tuple[int, int] = (320, 96),
    save_dir: Optional[str] = None,
    noise: bool = False,
) -> List[Tuple[str, Image.Image]]:
    """Generate `n` synthetic plate images (in-memory). Optionally save to disk.

    Returns a list of (text, PIL.Image).
    If `save_dir` is provided, images are written as PNG files named by index.
    """
    if formats is None:
        formats = DEFAULT_FORMATS
    out = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for i in range(n):
        fmt = random.choice(formats)
        txt = generate_plate_text(fmt)
        img = render_plate_image(txt, size=size, noise=noise)
        out.append((txt, img))
        if save_dir:
            filename = os.path.join(save_dir, f"plate_{i:04d}_{txt}.png")
            img.save(filename)
    return out


if __name__ == "__main__":
    # Quick demo: generate 5 plates, anonymize half of them in a sample bbox
    samples = generate_synthetic_dataset(5, noise=True)
    print("Generated plates:")
    for i, (txt, img) in enumerate(samples):
        print(i, txt, img.size)
    # Example anonymization using a sample bbox covering center-bottom region
    txt0, img0 = samples[0]
    w, h = img0.size
    bbox = (int(w * 0.25), int(h * 0.45), int(w * 0.75), int(h * 0.85))
    anonym = anonymize_plate(img0, bbox, method="pixelate", pixel_size=8)
    # Save to a temporary file only for demonstration if user runs this script directly
    out_path = os.path.join(os.path.dirname(__file__), "_demo_plate.png")
    anonym.save(out_path)
    print("Demo anonymized image saved to:", out_path)
