#!/usr/bin/env python3
"""
Step 7 mean aggregation test — memory-safe vectorized DINOtxt mean.

Loads saved DINOtxt patch features and averages them for the first NUM_POINTS
3D points using vectorized ops, while keeping at most ONE frame (~13 MB) in
memory at a time. Saves aggregated vectors to test/test_outputs/.

Why this is both fast and memory-safe:
- In visibility.npz, observations are stored sorted by point index with a CSR
  offset array (point_obs_offsets). The first NUM_POINTS points occupy a
  contiguous slice, so we read only their observations.
- We then load each *relevant* frame exactly once, gather only the needed
  patches, accumulate into a tiny (NUM_POINTS, D) running sum, and drop the
  frame. We never hold all frames (26 GB) in memory.

Run from repo root:
    python test/test_mean.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# --- config (edit these) ---
NUM_POINTS = 100
SPOT_CHECK = 10  # re-aggregate first K points with an independent loop (0 = skip)
VISIBILITY_NPZ = "outputs/room2/visibility.npz"
FEATURES_DIR = "outputs/room2/features_dinotxt"
POINTS_NPY = "outputs/room2/points_100000.npy"  # optional, summary only
# ---------------------------

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
OUT_DIR = TEST_DIR / "test_outputs" / f"mean_{NUM_POINTS}pts"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings as cfg
from src.utils.geometry import normalize_vectors
from src.utils.io import load_npy, save_npy


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def load_first_n_observations(
    vis_path: Path, n_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Return (point_idx, frame_idx, patch_idx_flat) for the first n_points only.

    Reads each full obs array once, slices the contiguous first-n_points range,
    then drops the full array (peak ~one obs array, a few hundred MB).
    """
    w_patches = cfg.PATCH_GRID_W
    with np.load(vis_path) as npz:
        offsets = npz["point_obs_offsets"]
        n_orig = int(npz["num_points"][0])
        if n_points > n_orig:
            raise ValueError(f"NUM_POINTS={n_points} exceeds visibility ({n_orig})")
        start = int(offsets[0])
        end = int(offsets[n_points])

        obs_pi = npz["obs_point_idx"][start:end].astype(np.int64)
        obs_fi = npz["obs_frame_idx"][start:end].astype(np.int64)
        px = npz["obs_patch_x"][start:end].astype(np.int64)
        py = npz["obs_patch_y"][start:end].astype(np.int64)

    patch_flat = py * w_patches + px
    return obs_pi, obs_fi, patch_flat, n_orig


def aggregate_mean_first_n(
    obs_pi: np.ndarray,
    obs_fi: np.ndarray,
    patch_flat: np.ndarray,
    n_points: int,
    feature_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Frame-outer accumulation; one frame in memory at a time."""
    dim = cfg.DINOTXT_FEATURE_DIM
    running = np.zeros((n_points, dim), dtype=np.float64)
    counts = np.zeros(n_points, dtype=np.int64)

    unique_frames = np.unique(obs_fi)
    for fi in unique_frames:
        feat_path = feature_dir / f"{int(fi):06d}.npy"
        if not feat_path.exists():
            continue
        mask = obs_fi == fi
        slots = obs_pi[mask]
        idx = patch_flat[mask]

        img = np.load(feat_path)  # (H, W, D), ~13 MB
        flat = img.reshape(-1, dim)
        vecs = normalize_vectors(flat[idx].astype(np.float64))
        del img, flat

        np.add.at(running, slots, vecs)
        np.add.at(counts, slots, 1)

    aggregated = np.zeros((n_points, dim), dtype=np.float32)
    valid = counts > 0
    if valid.any():
        means = running[valid] / counts[valid, None]
        aggregated[valid] = normalize_vectors(means).astype(np.float32)
    return aggregated, counts


def spot_check_loop(
    obs_pi: np.ndarray,
    obs_fi: np.ndarray,
    patch_flat: np.ndarray,
    k: int,
    feature_dir: Path,
) -> np.ndarray:
    """Independent per-point reaggregation for the first k points (caches frames)."""
    dim = cfg.DINOTXT_FEATURE_DIM
    out = np.zeros((k, dim), dtype=np.float32)
    frame_cache: dict[int, np.ndarray] = {}
    for pi in range(k):
        sel = obs_pi == pi
        if not sel.any():
            continue
        fis = obs_fi[sel]
        idxs = patch_flat[sel]
        feats = []
        for fi, idx in zip(fis, idxs):
            fi = int(fi)
            if fi not in frame_cache:
                frame_cache[fi] = np.load(feature_dir / f"{fi:06d}.npy").reshape(-1, dim)
            feats.append(frame_cache[fi][int(idx)])
        feats = normalize_vectors(np.stack(feats).astype(np.float64))
        mean = feats.mean(axis=0)
        out[pi] = normalize_vectors(mean.reshape(1, -1)).squeeze(0).astype(np.float32)
    return out


def main() -> None:
    vis_path = resolve_path(VISIBILITY_NPZ)
    feat_dir = resolve_path(FEATURES_DIR)
    points_path = resolve_path(POINTS_NPY)

    if not vis_path.exists():
        raise FileNotFoundError(f"Missing visibility: {vis_path}")
    if not feat_dir.is_dir():
        raise FileNotFoundError(f"Missing DINOtxt features dir: {feat_dir}")

    n_frames_cached = len(list(feat_dir.glob("*.npy")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Visibility: {vis_path}")
    print(f"DINOtxt features: {feat_dir} ({n_frames_cached} frames cached)")
    print(f"Loading observations for first {NUM_POINTS} points...")

    obs_pi, obs_fi, patch_flat, n_orig = load_first_n_observations(vis_path, NUM_POINTS)
    unique_frames = np.unique(obs_fi)
    print(
        f"  {len(obs_pi):,} observations across {len(unique_frames):,} unique frames "
        f"(of {n_orig:,} total points)"
    )

    print("Aggregating (vectorized, one frame at a time)...")
    t0 = time.perf_counter()
    aggregated, counts = aggregate_mean_first_n(
        obs_pi, obs_fi, patch_flat, NUM_POINTS, feat_dir
    )
    elapsed = time.perf_counter() - t0

    out_path = OUT_DIR / "mean.npy"
    save_npy(out_path, aggregated)

    norms = np.linalg.norm(aggregated, axis=1)

    comparison: dict[str, float | int] = {}
    if SPOT_CHECK > 0:
        k = min(SPOT_CHECK, NUM_POINTS)
        print(f"Spot-checking against independent per-point loop on {k} points...")
        loop = spot_check_loop(obs_pi, obs_fi, patch_flat, k, feat_dir)
        max_diff = float(np.max(np.abs(aggregated[:k] - loop)))
        comparison = {"spot_check_points": k, "max_abs_diff": max_diff}
        print(f"  max |vectorized - loop| = {max_diff:.2e}")
        if max_diff > 1e-4:
            print("  WARNING: large discrepancy — check aggregation logic")

    if points_path.exists():
        save_npy(OUT_DIR / f"points_{NUM_POINTS}.npy", load_npy(points_path)[:NUM_POINTS])

    summary = {
        "num_points": NUM_POINTS,
        "num_orig_points": n_orig,
        "feature_dim": cfg.DINOTXT_FEATURE_DIM,
        "num_observations": int(len(obs_pi)),
        "num_unique_frames": int(len(unique_frames)),
        "observations_per_point_mean": float(counts.mean()),
        "observations_per_point_max": int(counts.max()),
        "points_with_zero_obs": int((counts == 0).sum()),
        "aggregated_norm_mean": float(norms.mean()),
        "vectorized_seconds": elapsed,
        "ms_per_point": elapsed / NUM_POINTS * 1000,
        "comparison": comparison,
        "inputs": {"visibility": str(vis_path), "features_dir": str(feat_dir)},
        "outputs": {"mean_npy": str(out_path)},
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {elapsed:.2f}s ({elapsed / NUM_POINTS * 1000:.2f} ms/point)")
    print(f"  mean L2 norm: {norms.mean():.4f}")
    print(f"  avg observations/point: {counts.mean():.1f} (max {counts.max()})")
    print(f"Saved to {OUT_DIR}/")
    print(f"  {out_path.name}")


if __name__ == "__main__":
    main()
