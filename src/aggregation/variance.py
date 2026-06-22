"""Semantic variance metrics: cosine dispersion and pairwise dissimilarity."""

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
from src.utils.progress import tqdm

from config import settings as cfg
from src.data.mesh import output_dir, points_path
from src.data.projection import load_visibility
from src.features.dino_backbone import features_vits16_dir, load_frame_features
from src.utils.geometry import normalize_vectors
from src.utils.io import ensure_dir, save_npy


def variance_dir() -> Path:
    return ensure_dir(output_dir() / "variance")


def dispersion_path() -> Path:
    return variance_dir() / "dispersion.npy"


def gather_point_features(
    vis: dict[str, np.ndarray],
    point_idx: int,
    feature_dir: Path | None = None,
) -> np.ndarray:
    """Collect ViT-S features for all observations of a point. Returns (K, D)."""
    feature_dir = feature_dir or features_vits16_dir()
    start = int(vis["point_obs_offsets"][point_idx])
    end = int(vis["point_obs_offsets"][point_idx + 1])
    if start == end:
        return np.zeros((0, cfg.VITS16_FEATURE_DIM), dtype=np.float32)

    frames = vis["obs_frame_idx"][start:end]
    px = vis["obs_patch_x"][start:end]
    py = vis["obs_patch_y"][start:end]

    feats = []
    frame_cache: dict[int, np.ndarray] = {}
    for fi, pxi, pyi in zip(frames, px, py):
        fi = int(fi)
        if fi not in frame_cache:
            frame_cache[fi] = load_frame_features(fi, feature_dir)
        feats.append(frame_cache[fi][int(pyi), int(pxi)])
    return np.stack(feats, axis=0)


def cosine_dispersion(feats: np.ndarray) -> float:
    """
    Primary flicker metric: 1 - ||mean(normalized feats)||_2.
    Returns 0 for empty or single observation.
    """
    if len(feats) <= 1:
        return 0.0
    normed = normalize_vectors(feats)
    mean_vec = normed.mean(axis=0)
    return float(1.0 - np.linalg.norm(mean_vec))


def mean_pairwise_dissimilarity(feats: np.ndarray) -> float:
    """
    Mean of (1 - cosine similarity) over pairs, excluding self-pairs.

    Uses the closed form from accumulated unit vectors:
    (n / (n - 1)) * (1 - ||mean(normalized feats)||^2).
    """
    if len(feats) <= 1:
        return 0.0
    normed = normalize_vectors(feats.astype(np.float64))
    n = len(normed)
    mean_vec = normed.sum(axis=0) / n
    r_squared = float(np.sum(mean_vec**2))
    return float(np.clip((n / (n - 1)) * (1.0 - r_squared), 0.0, 2.0))


def pairwise_dissimilarity_from_sums(
    running_sums: np.ndarray,
    num_obs: np.ndarray,
) -> np.ndarray:
    """Vectorized pairwise dissimilarity from per-point sums of normalized features."""
    n_points = len(num_obs)
    pairwise = np.zeros(n_points, dtype=np.float32)
    valid = num_obs >= 2
    if not valid.any():
        return pairwise

    n = num_obs[valid].astype(np.float64)
    mean_vectors = running_sums[valid] / n[:, None]
    r_squared = np.sum(mean_vectors**2, axis=1)
    pairwise[valid] = np.clip((n / (n - 1)) * (1.0 - r_squared), 0.0, 2.0).astype(np.float32)
    return pairwise


def _observations_sorted_by_frame(vis: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Sort flat CSR observations by frame index for frame-outer iteration."""
    order = np.argsort(vis["obs_frame_idx"], kind="stable")
    return (
        vis["obs_frame_idx"][order],
        vis["obs_point_idx"][order],
        vis["obs_patch_x"][order],
        vis["obs_patch_y"][order],
    )


def compute_dispersion_vectorized(
    vis: dict[str, np.ndarray],
    feature_dir: Path | None = None,
    compute_pairwise: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Frame-outer vectorized cosine dispersion.

    Loads each frame's ViT-S features once, gathers all visible point features
    for that frame in batch, and accumulates L2-normalized vectors per point.
    Dispersion = 1 - ||sum(normalized feats) / count|| (same as mean-of-normalized).
    Pairwise uses (n/(n-1)) * (1 - ||mean||^2) from the same accumulators.
    """
    feature_dir = feature_dir or features_vits16_dir()
    n_points = int(vis["num_points"][0])
    dim = cfg.VITS16_FEATURE_DIM
    w_patches = cfg.PATCH_GRID_W

    sorted_fi, sorted_pi, sorted_px, sorted_py = _observations_sorted_by_frame(vis)

    running_sums = np.zeros((n_points, dim), dtype=np.float64)
    num_obs = np.zeros(n_points, dtype=np.int32)

    if len(sorted_fi) > 0:
        unique_frames, frame_starts = np.unique(sorted_fi, return_index=True)
        frame_ends = np.append(frame_starts[1:], len(sorted_fi))

        for fi, start, end in tqdm(
            zip(unique_frames, frame_starts, frame_ends),
            total=len(unique_frames),
            desc="Variance",
        ):
            fi = int(fi)
            feat_path = feature_dir / f"{fi:06d}.npy"
            if not feat_path.exists():
                continue

            pi = sorted_pi[start:end]
            px = sorted_px[start:end].astype(np.int64)
            py = sorted_py[start:end].astype(np.int64)

            img_features = load_frame_features(fi, feature_dir)
            flat = img_features.reshape(-1, dim)
            patch_idx = py * w_patches + px
            extracted = flat[patch_idx]

            normalized = normalize_vectors(extracted.astype(np.float64))
            running_sums[pi] += normalized
            num_obs[pi] += 1

    dispersion = np.zeros(n_points, dtype=np.float32)
    valid = num_obs >= 2
    if valid.any():
        mean_vectors = running_sums[valid] / num_obs[valid, None]
        dispersion[valid] = (1.0 - np.linalg.norm(mean_vectors, axis=1)).astype(np.float32)

    pairwise = (
        pairwise_dissimilarity_from_sums(running_sums, num_obs)
        if compute_pairwise
        else np.zeros(n_points, dtype=np.float32)
    )

    return dispersion, pairwise, num_obs


def compute_all_dispersion(
    vis: dict[str, np.ndarray] | None = None,
    force: bool = False,
    feature_dir: Path | None = None,
    compute_pairwise: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 4: compute dispersion for all points."""
    out_disp = dispersion_path()
    out_pairwise = variance_dir() / "pairwise.npy"
    out_num_obs = variance_dir() / "num_observations.npy"

    if out_disp.exists() and not force:
        return np.load(out_disp), np.load(out_pairwise), np.load(out_num_obs)

    vis = vis or load_visibility()
    if compute_pairwise is None:
        compute_pairwise = getattr(cfg, "COMPUTE_PAIRWISE", False)

    dispersion, pairwise, num_obs = compute_dispersion_vectorized(
        vis,
        feature_dir=feature_dir,
        compute_pairwise=compute_pairwise,
    )

    save_npy(out_disp, dispersion)
    save_npy(out_pairwise, pairwise)
    save_npy(out_num_obs, num_obs)
    print(f"Dispersion: mean={dispersion.mean():.4f}, max={dispersion.max():.4f}")
    return dispersion, pairwise, num_obs


def main():
    disp, pw, nobs = compute_all_dispersion(force=True)
    print(f"Computed dispersion for {len(disp)} points, avg obs={nobs.mean():.1f}")


if __name__ == "__main__":
    main()
