#!/usr/bin/env python3
"""
Recompute variance (dispersion + pairwise) from cached test artifacts and visualize with jet.

Run from repo root:
    python test/test_variance_recompute.py

Edit RUN_DIR below. Set SHOW_VIEWER=False on headless/SSH machines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- config (edit these) ---
RUN_DIR = "variance_20000pts_2000frames"  # subfolder under test/test_outputs/
SHOW_VIEWER = True  # Open3D windows for dispersion + pairwise PLY
# ---------------------------

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_ROOT = TEST_DIR / "test_outputs"
DATA_DIR = OUTPUT_ROOT / RUN_DIR
OUT_DIR = DATA_DIR / "recompute_jet"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.aggregation.variance import compute_dispersion_vectorized
from src.data.projection import load_visibility
from src.utils.io import load_npy, save_npy
from src.viz.gt_labels import save_colored_ply


def values_to_jet_colors(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map scalar values to RGB using matplotlib jet colormap."""
    cmap = plt.colormaps["jet"]
    v = values.astype(np.float64)
    if vmin is None:
        pos = v[v > 0]
        vmin = float(np.percentile(pos, 2)) if len(pos) else float(v.min())
    if vmax is None:
        vmax = float(np.percentile(v, 98)) if len(v) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-8
    norm = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = cmap(norm)[:, :3]
    return (rgb * 255).astype(np.uint8)


def save_jet_scatter_png(
    points: np.ndarray,
    values: np.ndarray,
    out_path: Path,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    colors = values_to_jet_colors(values, vmin=vmin, vmax=vmax)
    rgb = colors.astype(np.float32) / 255.0
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=rgb,
        s=0.3,
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


def save_histogram(values: np.ndarray, out_path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=80, color="steelblue", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def show_open3d(ply_path: Path, title: str | None = None) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — skip PLY viewer")
        return
    if not ply_path.exists():
        print(f"  skip viewer (missing): {ply_path}")
        return
    pcd = o3d.io.read_point_cloud(str(ply_path))
    window_name = title or ply_path.name
    print(f"  Open3D: {window_name} ({len(pcd.points)} points) — close window for next")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=1280,
            height=720,
        )
    except Exception as exc:
        print(f"  Viewer failed ({exc}). Open manually: {ply_path}")


def main() -> None:
    vis_path = DATA_DIR / "visibility.npz"
    feat_dir = DATA_DIR / "features_vits16"
    points_files = sorted(DATA_DIR.glob("points_*.npy"))

    for p in (vis_path, feat_dir):
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")
    if not points_files:
        raise FileNotFoundError(f"No points_*.npy in {DATA_DIR}")

    points_path = points_files[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    points = load_npy(points_path)
    n_points = len(points)

    vis = load_visibility(vis_path)
    if int(vis["num_points"][0]) != n_points:
        raise ValueError(
            f"visibility has {vis['num_points'][0]} points, points file has {n_points}"
        )

    print(f"Recomputing variance for {n_points} points...")
    print(f"  visibility: {vis_path}")
    print(f"  features:   {feat_dir}")

    dispersion, pairwise, num_obs = compute_dispersion_vectorized(
        vis,
        feature_dir=feat_dir,
        compute_pairwise=True,
    )

    save_npy(OUT_DIR / "dispersion.npy", dispersion)
    save_npy(OUT_DIR / "pairwise.npy", pairwise)
    save_npy(OUT_DIR / "num_observations.npy", num_obs)

    # Shared jet scale per metric (2–98 percentile on positive values)
    disp_pos = dispersion[dispersion > 0]
    disp_vmin = float(np.percentile(disp_pos, 2)) if len(disp_pos) else 0.0
    disp_vmax = float(np.percentile(dispersion, 98)) if len(dispersion) else 1.0

    pw_pos = pairwise[pairwise > 0]
    pw_vmin = float(np.percentile(pw_pos, 2)) if len(pw_pos) else 0.0
    pw_vmax = float(np.percentile(pairwise, 98)) if len(pairwise) else 1.0

    disp_ply = OUT_DIR / "dispersion_jet.ply"
    pw_ply = OUT_DIR / "pairwise_jet.ply"
    save_colored_ply(points, values_to_jet_colors(dispersion, disp_vmin, disp_vmax), disp_ply)
    save_colored_ply(points, values_to_jet_colors(pairwise, pw_vmin, pw_vmax), pw_ply)

    disp_png = OUT_DIR / "dispersion_jet_scatter.png"
    pw_png = OUT_DIR / "pairwise_jet_scatter.png"
    save_jet_scatter_png(
        points,
        dispersion,
        disp_png,
        f"Cosine dispersion (jet) — N={n_points}",
        vmin=disp_vmin,
        vmax=disp_vmax,
    )
    save_jet_scatter_png(
        points,
        pairwise,
        pw_png,
        f"Pairwise dissimilarity (jet) — N={n_points}",
        vmin=pw_vmin,
        vmax=pw_vmax,
    )

    save_histogram(
        dispersion,
        OUT_DIR / "dispersion_histogram.png",
        "Cosine dispersion",
        "Cosine dispersion",
    )
    save_histogram(
        pairwise,
        OUT_DIR / "pairwise_histogram.png",
        "Pairwise dissimilarity",
        "Pairwise dissimilarity",
    )

    cached_disp = DATA_DIR / "dispersion.npy"
    cached_pw = DATA_DIR / "pairwise.npy"
    comparison = {}
    if cached_disp.exists():
        old_d = load_npy(cached_disp)
        comparison["dispersion_max_abs_diff"] = float(np.max(np.abs(old_d - dispersion)))
    if cached_pw.exists():
        old_p = load_npy(cached_pw)
        comparison["pairwise_max_abs_diff"] = float(np.max(np.abs(old_p - pairwise)))

    summary = {
        "run_dir": str(DATA_DIR),
        "num_points": n_points,
        "num_obs_mean": float(num_obs.mean()),
        "dispersion_mean": float(dispersion.mean()),
        "dispersion_max": float(dispersion.max()),
        "pairwise_mean": float(pairwise.mean()),
        "pairwise_max": float(pairwise.max()),
        "jet_scale": {
            "dispersion": [disp_vmin, disp_vmax],
            "pairwise": [pw_vmin, pw_vmax],
        },
        "comparison_to_cached": comparison,
        "outputs": {
            "dispersion_ply": str(disp_ply),
            "pairwise_ply": str(pw_ply),
            "dispersion_scatter_png": str(disp_png),
            "pairwise_scatter_png": str(pw_png),
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDispersion: mean={dispersion.mean():.4f}, max={dispersion.max():.4f}")
    print(f"Pairwise:   mean={pairwise.mean():.4f}, max={pairwise.max():.4f}")
    if comparison:
        print(f"Cached diff: {comparison}")
    print(f"\nSaved jet visualizations to {OUT_DIR}/")
    print(f"  {disp_ply.name}")
    print(f"  {pw_ply.name}")

    if SHOW_VIEWER:
        print("\nPLY viewer — close each window to advance.")
        show_open3d(disp_ply, title="Cosine dispersion (jet)")
        show_open3d(pw_ply, title="Pairwise dissimilarity (jet)")


if __name__ == "__main__":
    main()
