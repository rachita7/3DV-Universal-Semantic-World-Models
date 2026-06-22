"""Image preprocessing: pad to multiple of 16, ImageNet normalize."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import numpy as np
import torch
from PIL import Image

from config import settings as cfg

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pad_image_rgb(img: np.ndarray) -> np.ndarray:
    """Pad HxWx3 uint8 RGB bottom to PAD_HEIGHT."""
    h, w = img.shape[:2]
    if h == cfg.PAD_HEIGHT:
        return img
    pad_h = cfg.PAD_HEIGHT - h
    return np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode="constant", constant_values=0)


def rgb_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 -> 1x3xHxW float tensor, ImageNet normalized."""
    img = pad_image_rgb(img).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return t


def load_rgb_tensor_from_array(img: np.ndarray, device: str | torch.device) -> torch.Tensor:
    return rgb_to_tensor(img).to(device)


def main():
    dummy = np.zeros((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3), dtype=np.uint8)
    t = rgb_to_tensor(dummy)
    assert t.shape == (1, 3, cfg.PAD_HEIGHT, cfg.IMAGE_WIDTH)
    print(f"Tensor shape: {tuple(t.shape)}")


if __name__ == "__main__":
    main()
