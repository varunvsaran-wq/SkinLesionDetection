"""Phase 2 acceptance tests for the preprocessing pipeline.

Uses synthetic fixtures with known properties so the algorithms are verified
deterministically without committing dataset images. A final test runs against a
handful of real HAM10000 samples *if* a dataset is present locally, and is
skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dermassist import preprocessing as pp
from dermassist.config import get_settings


# --------------------------- synthetic fixtures ---------------------------- #


def _color_cast_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Uniform image with a strong red cast (R dominant, G/B equal)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 200  # R
    img[..., 1] = 100  # G
    img[..., 2] = 100  # B
    return img


def _haired_image(h: int = 80, w: int = 80) -> tuple[np.ndarray, slice, slice]:
    """Light gray lesion with a thin dark horizontal 'hair' across the middle."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    rows = slice(h // 2 - 1, h // 2 + 1)  # 2px thick hair
    cols = slice(5, w - 5)
    img[rows, cols, :] = 20  # dark hair
    return img, rows, cols


# ------------------------------- tests ------------------------------------- #


def test_shades_of_gray_reduces_color_cast():
    img = _color_cast_image()
    out = pp.shades_of_gray(img, power=6)

    assert out.shape == img.shape
    assert out.dtype == np.uint8

    before = img.reshape(-1, 3).mean(axis=0)
    after = out.reshape(-1, 3).mean(axis=0)
    # The spread across channel means should shrink as the cast is neutralized.
    assert after.std() < before.std()


def test_hair_mask_detects_the_hair():
    img, rows, cols = _haired_image()
    mask = pp.hair_mask(img)

    assert mask.shape == img.shape[:2]
    # Hair region should be flagged; the clean corners should not.
    assert mask[rows, cols].mean() > 0
    assert mask[:5, :5].sum() == 0


def test_remove_hair_inpaints_dark_shaft():
    img, rows, cols = _haired_image()
    cleaned = pp.remove_hair(img)

    assert cleaned.shape == img.shape
    before = img[rows, cols].mean()
    after = cleaned[rows, cols].mean()
    # Dark hair pixels should be filled toward the surrounding light value.
    assert after > before + 50


def test_resize_changes_shape():
    img = _color_cast_image(100, 120)
    out = pp.resize_image(img, size=(224, 224))
    assert out.shape == (224, 224, 3)


def test_load_save_round_trip(tmp_path: Path):
    img = _color_cast_image()
    path = tmp_path / "round.png"
    pp.save_image(img, path)
    assert path.exists()
    loaded = pp.load_image(path)
    assert loaded.shape == img.shape
    # PNG is lossless: pixels are identical.
    assert np.array_equal(loaded, img)


def test_load_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        pp.load_image(tmp_path / "nope.png")


def test_preprocess_image_end_to_end(tmp_path: Path):
    img, _, _ = _haired_image()
    src = tmp_path / "lesion.png"
    pp.save_image(img, src)
    out = tmp_path / "out" / "lesion__preprocessed.png"

    written = pp.preprocess_image(src, out, size=(224, 224))
    assert written == out
    assert out.exists()

    result = pp.load_image(out)
    assert result.shape == (224, 224, 3)


def _find_real_samples(limit: int = 3) -> list[Path]:
    root = get_settings().dataset_path
    if not root.exists():
        return []
    samples: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        samples.extend(sorted(root.rglob(ext)))
        if len(samples) >= limit:
            break
    return samples[:limit]


@pytest.mark.parametrize("sample", _find_real_samples() or [None])
def test_preprocess_real_ham10000_samples(sample, tmp_path: Path):
    if sample is None:
        pytest.skip("No local HAM10000/ISIC dataset found (set DATASET_PATH).")
    out = tmp_path / f"{sample.stem}__pre.png"
    pp.preprocess_image(sample, out, size=(224, 224))
    assert pp.load_image(out).shape == (224, 224, 3)
