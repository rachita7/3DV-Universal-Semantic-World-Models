"""Visualize semantic variance as a colored point cloud."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import matplotlib.pyplot as plt
import numpy as np

from config import settings as cfg
from src.aggregation.variance import dispersion_path, variance_dir
from src.data.mesh import output_dir, points_path
from src.utils.io import load_npy


def dispersion_to_colors(dispersion: np.ndarray) -> np.ndarray:
    """Map dispersion to RGB using a percentile-scaled matplotlib jet colormap."""
    d = dispersion.astype(np.float64)
    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    vmin = float(np.percentile(d, 2))
    vmax = float(np.percentile(d, 98))
    if vmax <= vmin:
        vmax = float(d.max()) if d.max() > vmin else vmin + 1e-8
    norm = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = plt.colormaps["jet"]
    colors = cmap(norm)[:, :3]
    return (colors * 255).astype(np.uint8)


def save_variance_ply(points: np.ndarray, dispersion: np.ndarray, out_path: Path) -> None:
    """Write ASCII PLY with dispersion-colored vertices."""
    colors = dispersion_to_colors(dispersion)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def save_variance_histogram(dispersion: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dispersion, bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Cosine dispersion")
    ax.set_ylabel("Count")
    ax.set_title("Semantic flickering distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_variance_viz(force: bool = False) -> Path:
    """Step 4 viz: heatmap PLY + histogram."""
    out_ply = variance_dir() / "heatmap.ply"
    out_hist = variance_dir() / "histogram.png"

    if out_ply.exists() and out_hist.exists() and not force:
        print(f"Variance viz exists at {out_ply}")
        return out_ply

    points = load_npy(points_path(cfg.NUM_POINTS_INITIAL))
    dispersion = load_npy(dispersion_path())
    save_variance_ply(points, dispersion, out_ply)
    save_variance_histogram(dispersion, out_hist)
    print(f"Saved variance heatmap to {out_ply}")
    return out_ply


def main():
    run_variance_viz(force=True)


if __name__ == "__main__":
    main()
