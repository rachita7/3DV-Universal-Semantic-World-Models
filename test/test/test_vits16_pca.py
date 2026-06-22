#!/usr/bin/env python3
"""
Step 3 ViT-S/16 feature test — PCA false-color visualization for one frame.

Run from repo root:
    python test/test_vits16_pca.py

Edit FRAME_IDX below. Outputs go to test/test_outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- config (edit these) ---
FRAME_IDX = 200
EXTRACT_IF_MISSING = True  # run ViT-S on this frame if .npy not cached
DEVICE = "cuda"  # "cpu" if no GPU
# Optional: override feature cache dir (None = outputs/room2/features_vits16)
FEATURES_DIR: str | None = None
# ---------------------------

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_DIR = TEST_DIR / "test_outputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from config import settings as cfg
from src.data.replica import frame_path, load_rgb
from src.features.dino_backbone import (
    extract_patch_features,
    features_vits16_dir,
    load_frame_features,
)
from src.features.model_loaders import load_vits16
from src.features.preprocess import pad_image_rgb, rgb_to_tensor


def resolve_features_dir() -> Path:
    if FEATURES_DIR:
        path = Path(FEATURES_DIR)
        return path if path.is_absolute() else REPO_ROOT / path
    return features_vits16_dir()


def load_or_extract_features(frame_idx: int) -> np.ndarray:
    """Load cached ViT-S features or extract one frame using src step-3 functions."""
    feat_dir = resolve_features_dir()
    feat_path = feat_dir / f"{frame_idx:06d}.npy"

    if feat_path.exists():
        print(f"Loading cached features: {feat_path}")
        return load_frame_features(frame_idx, feature_dir=feat_dir)

    if not EXTRACT_IF_MISSING:
        raise FileNotFoundError(
            f"No features at {feat_path}. Set EXTRACT_IF_MISSING=True or run pipeline step 3."
        )

    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable — using CPU")

    print(f"Extracting ViT-S features for frame {frame_idx} ({device})...")
    model = load_vits16(device)
    rgb = load_rgb(frame_idx)
    tensor = rgb_to_tensor(rgb).to(device)
    feat = extract_patch_features(model, tensor)
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, feat)
    print(f"Saved features to {feat_path}")
    return feat


def pca_to_rgb(feat: np.ndarray) -> np.ndarray:
    """
    Reduce patch features (H, W, D) to RGB via PCA (3 components).

    Uses SVD on mean-centered patch vectors; per-channel percentile stretch to [0, 255].
    """
    h, w, d = feat.shape
    x = feat.reshape(-1, d).astype(np.float64)
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pcs = (x @ vt[:3].T).reshape(h, w, 3)

    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        ch = pcs[:, :, c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        rgb[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def upsample_patch_image(patch_img: np.ndarray, patch_size: int = cfg.PATCH_SIZE) -> np.ndarray:
    """Nearest-neighbor upsample (H_p, W_p, 3) to full padded resolution."""
    return np.repeat(np.repeat(patch_img, patch_size, axis=0), patch_size, axis=1)


def save_comparison_figure(
    rgb: np.ndarray,
    pca_full: np.ndarray,
    out_path: Path,
    frame_idx: int,
) -> None:
    rgb_pad = pad_image_rgb(rgb)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(rgb_pad)
    axes[0].set_title(f"Frame {frame_idx:06d} — RGB")
    axes[0].axis("off")
    axes[1].imshow(pca_full)
    axes[1].set_title(f"Frame {frame_idx:06d} — ViT-S PCA (3 components)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = OUTPUT_DIR / f"vits16_pca_frame_{FRAME_IDX:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = load_or_extract_features(FRAME_IDX)
    expected = (cfg.PATCH_GRID_H, cfg.PATCH_GRID_W, cfg.VITS16_FEATURE_DIM)
    if feat.shape != expected:
        raise ValueError(f"Expected feature shape {expected}, got {feat.shape}")

    pca_patch = pca_to_rgb(feat)
    pca_full = upsample_patch_image(pca_patch)

    pca_patch_path = out_dir / f"vits16_pca_frame_{FRAME_IDX:06d}_patchgrid.png"
    pca_full_path = out_dir / f"vits16_pca_frame_{FRAME_IDX:06d}.png"
    compare_path = out_dir / f"vits16_pca_frame_{FRAME_IDX:06d}_comparison.png"

    Image.fromarray(pca_patch).save(pca_patch_path)
    Image.fromarray(pca_full).save(pca_full_path)

    rgb = load_rgb(FRAME_IDX)
    save_comparison_figure(rgb, pca_full, compare_path, FRAME_IDX)

    norms = np.linalg.norm(feat.reshape(-1, feat.shape[-1]), axis=1)
    summary = {
        "frame_idx": FRAME_IDX,
        "feature_shape": list(feat.shape),
        "patch_grid": [cfg.PATCH_GRID_H, cfg.PATCH_GRID_W],
        "feature_dim": cfg.VITS16_FEATURE_DIM,
        "mean_patch_norm": float(norms.mean()),
        "rgb_source": str(frame_path(FRAME_IDX)),
        "outputs": {
            "pca_fullres": str(pca_full_path),
            "pca_patchgrid": str(pca_patch_path),
            "comparison": str(compare_path),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFrame {FRAME_IDX} ViT-S features: {feat.shape}")
    print(f"  mean patch L2 norm: {norms.mean():.4f}")
    print(f"Saved to {out_dir}/")
    print(f"  {pca_full_path.name}          — PCA RGB upsampled to {pca_full.shape[1]}x{pca_full.shape[0]}")
    print(f"  {pca_patch_path.name}  — native patch grid {pca_patch.shape[1]}x{pca_patch.shape[0]}")
    print(f"  {compare_path.name}   — RGB vs PCA side-by-side")


if __name__ == "__main__":
    main()
