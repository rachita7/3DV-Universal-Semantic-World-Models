#!/usr/bin/env python3
"""
Single-frame projection test — color points by visibility in one camera frame.

Run from repo root:
    python test/test_projection.py

Edit FRAME_IDX and NUM_POINTS below. Outputs go to test/test_outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# --- config (edit these) ---
FRAME_IDX = 1999
NUM_POINTS = 1_000_000
RANDOM_SEED = 42
SHOW_VIEWER = True
# Optional: path to existing points .npy (set to None to sample fresh)
POINTS_NPY: str | None = None
# ---------------------------

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_DIR = TEST_DIR / "test_outputs"

# Colors: visible / occluded-in-frustum / out of frustum
COLOR_VISIBLE = (0, 220, 0)
COLOR_OCCLUDED = (220, 40, 40)
COLOR_OUT_OF_FRUSTUM = (140, 140, 140)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings as cfg
from src.data.mesh import sample_mesh_points
from src.data.projection import (
    STATUS_OCCLUDED,
    STATUS_OUT_OF_FRUSTUM,
    STATUS_VISIBLE,
    project_frame_visibility,
)
from src.data.replica import frame_path, load_camera_params, load_trajectory
from src.utils.io import load_npy
from src.viz.gt_labels import save_colored_ply


def load_points() -> np.ndarray:
    if POINTS_NPY:
        path = Path(POINTS_NPY)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return load_npy(path)
    points, _, _ = sample_mesh_points(NUM_POINTS, RANDOM_SEED)
    return points


def status_to_colors(status: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(status), 3), dtype=np.uint8)
    colors[status == STATUS_OUT_OF_FRUSTUM] = COLOR_OUT_OF_FRUSTUM
    colors[status == STATUS_OCCLUDED] = COLOR_OCCLUDED
    colors[status == STATUS_VISIBLE] = COLOR_VISIBLE
    return colors


def show_open3d(ply_path: Path) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — skip viewer")
        return
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"  Open3D: {ply_path.name} — close window to exit")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"frame {FRAME_IDX:06d} visibility",
            width=1280,
            height=720,
        )
    except Exception as exc:
        print(f"  Viewer failed ({exc}). Open PLY in MeshLab: {ply_path}")


def main() -> None:
    out_dir = OUTPUT_DIR / f"projection_frame_{FRAME_IDX:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    poses = load_trajectory()
    if FRAME_IDX < 0 or FRAME_IDX >= len(poses):
        raise ValueError(f"FRAME_IDX={FRAME_IDX} out of range [0, {len(poses) - 1}]")

    cam = load_camera_params()
    points = load_points()
    print(f"Projecting {len(points)} points into frame {FRAME_IDX}...")
    print(f"  RGB: {frame_path(FRAME_IDX)}")
    print(f"  depth tolerance: {cfg.DEPTH_TOLERANCE} m")

    result = project_frame_visibility(points, FRAME_IDX, poses, cam)
    status = result["status"]
    n_vis = int((status == STATUS_VISIBLE).sum())
    n_occ = int((status == STATUS_OCCLUDED).sum())
    n_out = int((status == STATUS_OUT_OF_FRUSTUM).sum())

    colors = status_to_colors(status)
    ply_path = out_dir / "visibility_colored.ply"
    save_colored_ply(points, colors, ply_path)

    # Visible-only subset (green cloud)
    vis_idx = result["visible_indices"]
    if len(vis_idx) > 0:
        vis_colors = np.tile(np.array(COLOR_VISIBLE, dtype=np.uint8), (len(vis_idx), 1))
        save_colored_ply(points[vis_idx], vis_colors, out_dir / "visible_only.ply")

    np.save(out_dir / "visibility_status.npy", status)
    np.save(out_dir / f"points_{len(points)}.npy", points)

    summary = {
        "frame_idx": FRAME_IDX,
        "num_points": int(len(points)),
        "num_visible": n_vis,
        "num_occluded_in_frustum": n_occ,
        "num_out_of_frustum": n_out,
        "visible_fraction": float(n_vis / len(points)),
        "depth_tolerance_m": cfg.DEPTH_TOLERANCE,
        "camera": {"w": cam.w, "h": cam.h, "fx": cam.fx, "fy": cam.fy},
        "rgb_frame": str(frame_path(FRAME_IDX)),
        "legend": {
            "green": "visible (passed frustum + depth test)",
            "red": "in image frustum but occluded / depth mismatch",
            "gray": "behind camera or outside image / patch grid",
        },
        "sample_visible_uv": [
            {"u": float(u), "v": float(v), "z_cam": float(z)}
            for u, v, z in zip(
                result["u"][:5],
                result["v"][:5],
                result["z_cam"][:5],
            )
        ],
        "outputs": {
            "colored_ply": str(ply_path),
            "visible_only_ply": str(out_dir / "visible_only.ply"),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFrame {FRAME_IDX} visibility:")
    print(f"  visible:  {n_vis:6d} ({100 * n_vis / len(points):.1f}%)")
    print(f"  occluded: {n_occ:6d} ({100 * n_occ / len(points):.1f}%)")
    print(f"  outside:  {n_out:6d} ({100 * n_out / len(points):.1f}%)")
    print(f"\nSaved to {out_dir}/")
    print(f"  {ply_path.name}  (green / red / gray)")
    print(f"  visible_only.ply")
    print(f"  summary.json")

    if SHOW_VIEWER:
        show_open3d(ply_path)


if __name__ == "__main__":
    main()
