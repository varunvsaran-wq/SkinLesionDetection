"""Phase 2 — real dermoscopy image preprocessing.

Pure, testable functions operating on RGB ``uint8`` numpy arrays:

    load_image -> remove_hair -> shades_of_gray -> resize -> save_image

- **Shades-of-Gray color constancy** normalizes the illuminant so lesions imaged
  under different light/scopes look comparable (Minkowski p-norm, p=6 default).
- **DullRazor-style hair removal** detects dark hair shafts via a morphological
  blackhat, then inpaints them away so they don't confuse the classifier.

These functions take/return plain numpy arrays so they can be unit-tested without
the graph. ``preprocess_image`` ties them together for the ``preprocess`` node.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Default classifier input size (EfficientNet-B0 / ViT-B/16 expect 224×224).
DEFAULT_SIZE: tuple[int, int] = (224, 224)
DEFAULT_SOG_POWER: int = 6


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an RGB ``uint8`` array of shape (H, W, 3).

    PIL is used so we get a predictable RGB channel order (OpenCV loads BGR).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as im:
        # np.array (not asarray) returns a writable copy — avoids a non-writable
        # tensor warning when the array is later handed to torch/OpenCV.
        return np.array(im.convert("RGB"), dtype=np.uint8)


def save_image(img: np.ndarray, path: str | Path) -> Path:
    """Save an RGB ``uint8`` array to ``path`` (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(img)).save(path)
    return path


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Clip to [0, 255] and cast to uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)


def shades_of_gray(img: np.ndarray, power: int = DEFAULT_SOG_POWER) -> np.ndarray:
    """Shades-of-Gray color constancy.

    Estimates the per-channel illuminant as the Minkowski p-norm of each channel,
    normalizes that illuminant to unit length, and rescales the channels so the
    estimated light source becomes neutral gray. ``power=1`` is Gray-World,
    ``power→∞`` approaches max-RGB; ``power=6`` is the common dermoscopy default.
    """
    arr = img.astype(np.float32)
    # Per-channel Minkowski mean -> illuminant estimate, shape (3,).
    illum = np.power(np.mean(np.power(arr, power), axis=(0, 1)), 1.0 / power)
    illum = np.maximum(illum, 1e-6)
    # Normalize illuminant to unit vector, then distribute across 3 channels so
    # overall brightness is roughly preserved.
    illum = illum / np.sqrt(np.sum(illum**2))
    scale = 1.0 / (illum * np.sqrt(3.0))
    return _to_uint8(arr * scale)


def hair_mask(
    img: np.ndarray,
    kernel_size: int = 17,
    threshold: int = 10,
) -> np.ndarray:
    """Binary mask (uint8 0/255) of detected hair shafts via morphological blackhat.

    Blackhat = closing(img) - img highlights thin dark structures (hairs) against a
    lighter lesion/skin background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def remove_hair(
    img: np.ndarray,
    kernel_size: int = 17,
    threshold: int = 10,
    inpaint_radius: int = 1,
) -> np.ndarray:
    """DullRazor-style hair removal: detect hair, then inpaint it away.

    Returns the cleaned RGB ``uint8`` image (same shape as input).
    """
    mask = hair_mask(img, kernel_size=kernel_size, threshold=threshold)
    # cv2.inpaint is channel-order agnostic for the fill; keep RGB throughout.
    cleaned = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned


def resize_image(img: np.ndarray, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Resize to (width, height). INTER_AREA for downscale, INTER_CUBIC for upscale."""
    w, h = size
    cur_h, cur_w = img.shape[:2]
    interp = cv2.INTER_AREA if (w * h) < (cur_w * cur_h) else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def preprocess_array(
    img: np.ndarray,
    size: tuple[int, int] = DEFAULT_SIZE,
    do_hair_removal: bool = True,
    sog_power: int = DEFAULT_SOG_POWER,
) -> np.ndarray:
    """Full preprocessing on an in-memory array: hair removal → color constancy → resize."""
    if do_hair_removal:
        img = remove_hair(img)
    img = shades_of_gray(img, power=sog_power)
    img = resize_image(img, size=size)
    return img


def preprocess_image(
    input_path: str | Path,
    output_path: str | Path,
    size: tuple[int, int] = DEFAULT_SIZE,
    do_hair_removal: bool = True,
    sog_power: int = DEFAULT_SOG_POWER,
) -> Path:
    """Load an image, run the full preprocessing pipeline, and save the result.

    Returns the path the preprocessed image was written to.
    """
    img = load_image(input_path)
    out = preprocess_array(
        img, size=size, do_hair_removal=do_hair_removal, sog_power=sog_power
    )
    return save_image(out, output_path)


__all__ = [
    "DEFAULT_SIZE",
    "DEFAULT_SOG_POWER",
    "load_image",
    "save_image",
    "shades_of_gray",
    "hair_mask",
    "remove_hair",
    "resize_image",
    "preprocess_array",
    "preprocess_image",
]
