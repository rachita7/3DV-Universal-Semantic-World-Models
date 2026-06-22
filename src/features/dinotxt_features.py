"""DINOtxt aligned patch token extraction."""

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
from src.utils.progress import tqdm

from config import settings as cfg
from src.data.mesh import output_dir
from src.data.replica import load_rgb, load_trajectory
from src.features.preprocess import rgb_to_tensor
from src.utils.cache import should_skip
from src.utils.io import ensure_dir, save_npy

from src.features.model_loaders import load_dinotxt


def features_dinotxt_dir() -> Path:
    return ensure_dir(output_dir() / "features_dinotxt")


@torch.inference_mode()
def extract_aligned_patches(model, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Returns L2-normalized aligned patch tokens (H_p, W_p, D) from DINOtxt vision head.

    Uses image_patch_tokens from encode_image_with_patch_tokens (1024-d per patch).
    Note: normalize=True on that API only L2-normalizes the global image_features,
    not per-patch tokens — we normalize patches explicitly here.
    """
    _, patch_tokens, _ = model.encode_image_with_patch_tokens(img_tensor, normalize=False)
    _, n, d = patch_tokens.shape
    h_p = cfg.PATCH_GRID_H
    w_p = cfg.PATCH_GRID_W
    assert n == h_p * w_p, f"Expected {h_p * w_p} patch tokens, got {n}"
    feat = patch_tokens.reshape(1, h_p, w_p, d).squeeze(0)
    feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
    return feat.cpu().numpy().astype(np.float32)


def extract_all_frames(force: bool = False, device: str | None = None) -> Path:
    """Step 6: extract DINOtxt aligned patch features for all frames."""
    out_dir = features_dinotxt_dir()
    poses = load_trajectory()
    n_frames = len(poses)
    device = device or cfg.DEVICE
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"

    if not force:
        all_exist = all(should_skip(out_dir / f"{fi:06d}.npy", False) for fi in range(n_frames))
        if all_exist:
            print(f"All {n_frames} DINOtxt feature files exist, skipping.")
            return out_dir

    model, _ = load_dinotxt(device)
    for fi in tqdm(range(n_frames), desc="DINOtxt features"):
        out_path = out_dir / f"{fi:06d}.npy"
        if should_skip(out_path, force):
            continue
        rgb = load_rgb(fi)
        tensor = rgb_to_tensor(rgb).to(device)
        feat = extract_aligned_patches(model, tensor)
        save_npy(out_path, feat)

    print(f"Saved DINOtxt features to {out_dir}")
    return out_dir


def load_frame_features(
    fi: int,
    feature_dir: Path | None = None,
    *,
    mmap: bool = False,
) -> np.ndarray:
    feature_dir = feature_dir or features_dinotxt_dir()
    path = feature_dir / f"{fi:06d}.npy"
    if mmap:
        return np.load(path, mmap_mode="r")
    return np.load(path)


def main():
    import config.settings as settings

    settings.NUM_FRAMES = 1
    extract_all_frames(force=True, device="cpu")
    feat = load_frame_features(0)
    print(f"Frame 0 DINOtxt features: {feat.shape}")


if __name__ == "__main__":
    main()
