#!/usr/bin/env python3
"""
Validate the sam_slerp aggregation logic with a synthetic SAM cache (no GPU/SAM).

Writes a few fake per-frame SAM artifacts (seg_map + mask_emb + mask_area) into an
isolated output dir, builds a synthetic visibility dict, runs the production
sam_slerp aggregation, and checks the result against a manual area-weighted SLERP.

Run from repo root:
    conda run -n 3dvision python test/test_sam_slerp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings as cfg

# Isolate outputs so we never touch production outputs/room2/.
cfg.OUTPUT_ROOT = str(TEST_DIR / "test_outputs")
cfg.ROOM = "sam_slerp_test"

from src.aggregation.slerp_agg import _slerp_from_weights, sam_slerp_aggregate_all
from src.features.sam_masks import sam_frame_path, sam_masks_dir
from src.utils.geometry import normalize_vectors

DIM = cfg.DINOTXT_FEATURE_DIM
H, W = cfg.PATCH_GRID_H, cfg.PATCH_GRID_W


def write_synthetic_frame(fi: int, n_masks: int, rng: np.random.Generator) -> dict:
    """Create one fake SAM frame: random normalized mask embeddings + a seg_map."""
    mask_emb = normalize_vectors(rng.standard_normal((n_masks, DIM))).astype(np.float32)
    mask_area = rng.integers(500, 50_000, size=n_masks).astype(np.int32)
    seg_map = np.full((H, W), -1, dtype=np.int16)
    # Tile patches across masks deterministically so lookups are predictable.
    for m in range(n_masks):
        seg_map[:, m::n_masks] = m
    np.savez_compressed(sam_frame_path(fi), seg_map=seg_map, mask_emb=mask_emb, mask_area=mask_area)
    return {"seg_map": seg_map, "mask_emb": mask_emb, "mask_area": mask_area}


def build_synthetic_vis(observations: list[tuple[int, int, int, int]], n_points: int) -> dict:
    """observations: list of (point_idx, frame, patch_x, patch_y), sorted by point."""
    observations = sorted(observations, key=lambda o: o[0])
    pi = np.array([o[0] for o in observations], dtype=np.int32)
    fr = np.array([o[1] for o in observations], dtype=np.int32)
    px = np.array([o[2] for o in observations], dtype=np.int16)
    py = np.array([o[3] for o in observations], dtype=np.int16)
    counts = np.bincount(pi, minlength=n_points)
    offsets = np.zeros(n_points + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    n_frames = int(fr.max()) + 1 if len(fr) else 0
    return {
        "obs_point_idx": pi,
        "obs_frame_idx": fr,
        "obs_patch_x": px,
        "obs_patch_y": py,
        "point_obs_offsets": offsets,
        "num_points": np.array([n_points], dtype=np.int32),
        "num_frames": np.array([n_frames], dtype=np.int32),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    sam_masks_dir()  # ensure isolated dir exists

    # 3 frames with 2/3/2 masks.
    frames = {0: write_synthetic_frame(0, 2, rng), 1: write_synthetic_frame(1, 3, rng), 2: write_synthetic_frame(2, 2, rng)}

    # Point 0: seen in all 3 frames. Point 1: frames 0,2. Point 2: no observations.
    observations = [
        (0, 0, 0, 0),  # frame0 patch(0,0) -> mask 0%2=0
        (0, 1, 1, 0),  # frame1 patch x=1 -> 1%3=1
        (0, 2, 1, 0),  # frame2 x=1 -> 1%2=1
        (1, 0, 1, 5),  # frame0 x=1 -> 1
        (1, 2, 0, 5),  # frame2 x=0 -> 0
    ]
    n_points = 3
    vis = build_synthetic_vis(observations, n_points)

    aggregated = sam_slerp_aggregate_all(n_points, point_indices=None, vis=vis)

    # ---- manual expected for point 0 ----
    def lookup(fi, px, py):
        e = frames[fi]
        mid = int(e["seg_map"][py, px])
        return e["mask_emb"][mid], float(e["mask_area"][mid])

    feats0, w0 = zip(*[lookup(0, 0, 0), lookup(1, 1, 0), lookup(2, 1, 0)])
    expected0 = _slerp_from_weights(np.stack(feats0), np.asarray(w0, dtype=np.float64))

    diff0 = float(np.max(np.abs(aggregated[0] - expected0)))
    norms = np.linalg.norm(aggregated, axis=1)

    print(f"point 0 max abs diff vs manual: {diff0:.2e}")
    print(f"point 0 norm: {norms[0]:.4f} (expect ~1)")
    print(f"point 1 norm: {norms[1]:.4f} (expect ~1)")
    print(f"point 2 norm: {norms[2]:.4f} (expect 0 — no observations)")

    ok = diff0 < 1e-6 and abs(norms[0] - 1) < 1e-4 and abs(norms[1] - 1) < 1e-4 and norms[2] == 0.0
    print("\nRESULT:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
