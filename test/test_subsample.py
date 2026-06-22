#!/usr/bin/env python3
"""
Step 5 subsampling test — variance-weighted subsample from cached 20k run.

Loads pairwise variance from test/test_outputs/variance_20000pts_2000frames/,
subsamples 10k points with probability proportional to variance, and visualizes.

Run from repo root:
    python test/test_subsample.py

Edit RUN_DIR / NUM_SUBSAMPLE / SHOW_VIEWER below.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- config (edit these) ---
RUN_DIR = "variance_20000pts_2000frames"
NUM_INITIAL = 20_000
NUM_SUBSAMPLE = 10_000
RANDOM_SEED = 42  # subsample uses RANDOM_SEED + 1 (matches production step 5)
SHOW_VIEWER = True  # set False on headless/SSH machines
WEIGHT_METRIC = "pairwise"  # "pairwise" | "dispersion" (production step 5 uses dispersion)
# ---------------------------

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_ROOT = TEST_DIR / "test_outputs"
DATA_DIR = OUTPUT_ROOT / RUN_DIR
OUT_DIR = DATA_DIR / f"subsample_{NUM_SUBSAMPLE}"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mesh import subsample_by_variance
from src.utils.io import load_npy, save_npy
from src.viz.variance_heatmap import save_variance_ply


def resolve_weights(data_dir: Path, metric: str) -> tuple[np.ndarray, Path]:
    """Load variance weights; fall back to recompute_jet/ if main pairwise cache is empty."""
    if metric not in ("pairwise", "dispersion"):
        raise ValueError(f"WEIGHT_METRIC must be 'pairwise' or 'dispersion', got {metric!r}")

    primary = data_dir / f"{metric}.npy"
    if not primary.exists():
        raise FileNotFoundError(f"Missing weight file: {primary}")

    weights = load_npy(primary)
    if metric == "pairwise" and np.all(weights == 0):
        fallback = data_dir / "recompute_jet" / "pairwise.npy"
        if fallback.exists():
            print(
                f"  note: {primary.name} is all zeros (COMPUTE_PAIRWISE=False during variance run); "
                f"using {fallback.relative_to(data_dir.parent)}"
            )
            return load_npy(fallback), fallback
        raise ValueError(
            f"{primary.name} is all zeros. Re-run test/test_variance_recompute.py first, "
            "or set WEIGHT_METRIC='dispersion'."
        )
    return weights, primary


def save_scatter_png(
    points: np.ndarray,
    values: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    cmap = plt.colormaps["hot"]
    norm = np.clip(values.astype(np.float64), 0.0, 1.0)
    rgb = cmap(norm)[:, :3]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=rgb,
        s=0.4,
        alpha=0.85,
        marker=".",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def show_open3d(ply_path: Path) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — skip viewer")
        return
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"  Open3D: {ply_path.name} ({len(pcd.points)} points) — close window to exit")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Variance-weighted subsample",
            width=1280,
            height=720,
        )
    except Exception as exc:
        print(f"  Viewer failed ({exc}). Open PLY manually: {ply_path}")


def main() -> None:
    points_path = DATA_DIR / f"points_{NUM_INITIAL}.npy"
    dispersion_path = DATA_DIR / "dispersion.npy"

    if not points_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {points_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    points = load_npy(points_path)
    weights, weights_path = resolve_weights(DATA_DIR, WEIGHT_METRIC)
    dispersion = load_npy(dispersion_path) if dispersion_path.exists() else None

    if len(points) != len(weights):
        raise ValueError(
            f"Points ({len(points)}) and weights ({len(weights)}) length mismatch"
        )

    print(f"Loaded {len(points)} points from {points_path.name}")
    print(f"Loaded {WEIGHT_METRIC} weights from {weights_path.name} (mean={weights.mean():.4f})")

    idx = subsample_by_variance(weights, NUM_SUBSAMPLE, RANDOM_SEED + 1)
    subsampled_points = points[idx]
    subsampled_weights = weights[idx]

    idx_path = OUT_DIR / f"points_{NUM_SUBSAMPLE}_idx.npy"
    pts_path = OUT_DIR / f"points_{NUM_SUBSAMPLE}.npy"
    ply_path = OUT_DIR / "subsample_heatmap.ply"
    png_path = OUT_DIR / "subsample_scatter.png"

    save_npy(idx_path, idx.astype(np.int32))
    save_npy(pts_path, subsampled_points)
    save_variance_ply(subsampled_points, subsampled_weights, ply_path)
    save_scatter_png(
        subsampled_points,
        subsampled_weights,
        png_path,
        f"Variance-weighted subsample (N={len(subsampled_points)})",
    )

    orig_mean = float(weights.mean())
    sub_mean = float(subsampled_weights.mean())
    uplift = sub_mean / orig_mean if orig_mean > 0 else float("nan")

    print(f"\n{WEIGHT_METRIC.capitalize()} variance (subsampling weights from {weights_path.name}):")
    print(f"  original {len(points):,} points:  mean = {orig_mean:.6f}")
    print(f"  subsampled {len(idx):,} points: mean = {sub_mean:.6f}")
    print(f"  uplift (sub / orig):            {uplift:.3f}x")

    if dispersion is not None and WEIGHT_METRIC != "dispersion":
        orig_disp = float(dispersion.mean())
        sub_disp = float(dispersion[idx].mean())
        print(f"\nCosine dispersion (reference, not used for weights):")
        print(f"  original {len(points):,} points:  mean = {orig_disp:.6f}")
        print(f"  subsampled {len(idx):,} points: mean = {sub_disp:.6f}")

    summary = {
        "run_dir": str(DATA_DIR),
        "weight_metric": WEIGHT_METRIC,
        "weight_file": str(weights_path),
        "num_initial": int(len(points)),
        "num_subsample": int(len(idx)),
        "random_seed": RANDOM_SEED + 1,
        f"{WEIGHT_METRIC}_mean_original": orig_mean,
        f"{WEIGHT_METRIC}_mean_subsampled": sub_mean,
        f"{WEIGHT_METRIC}_uplift_ratio": uplift,
        "outputs": {
            "indices": str(idx_path),
            "points": str(pts_path),
            "heatmap_ply": str(ply_path),
            "scatter_png": str(png_path),
        },
    }
    if dispersion is not None:
        summary["dispersion_mean_original"] = orig_disp
        summary["dispersion_mean_subsampled"] = sub_disp

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  {idx_path.name}")
    print(f"  {pts_path.name}")
    print(f"  {ply_path.name}")
    print(f"  {png_path.name}")

    if SHOW_VIEWER:
        show_open3d(ply_path)


if __name__ == "__main__":
    main()
