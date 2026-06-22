"""3D point projection, frustum culling, and occlusion testing."""

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

from config import settings as cfg
from src.data.mesh import output_dir, points_path, run_sample_points
from src.data.replica import load_camera_params, load_depth, load_trajectory
from src.utils.geometry import invert_pose, project_to_pixels, world_to_camera
from src.utils.io import ensure_dir, load_npy


def visibility_path() -> Path:
    return output_dir() / "visibility.npz"


def _build_point_offsets(obs_point_idx: np.ndarray, n_points: int) -> np.ndarray:
    """CSR offsets from flat observation point indices."""
    counts = np.bincount(obs_point_idx, minlength=n_points)
    offsets = np.zeros(n_points + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return offsets


# Per-point status for single-frame projection (used by tests and build_visibility)
STATUS_OUT_OF_FRUSTUM = 0
STATUS_OCCLUDED = 1
STATUS_VISIBLE = 2


def project_frame_visibility(
    points: np.ndarray,
    frame_idx: int,
    poses: np.ndarray,
    cam,
    depth_tolerance: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Project all points into one frame and classify visibility.

    Returns dict with:
        status: (N,) int8 — OUT_OF_FRUSTUM | OCCLUDED | VISIBLE
        visible_indices: indices of visible points
        u, v, z_cam, patch_x, patch_y: arrays aligned to visible_indices
    """
    depth_tolerance = depth_tolerance if depth_tolerance is not None else cfg.DEPTH_TOLERANCE
    n_points = len(points)
    status = np.zeros(n_points, dtype=np.int8)
    empty = np.array([], dtype=np.int32)

    w2c = invert_pose(poses[frame_idx])
    cam_pts = world_to_camera(points.astype(np.float64), w2c)
    z = cam_pts[:, 2]
    u, v, _ = project_to_pixels(cam_pts, cam.fx, cam.fy, cam.cx, cam.cy)

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    px = (u // cfg.PATCH_SIZE).astype(np.int32)
    py = (v // cfg.PATCH_SIZE).astype(np.int32)

    mask = (
        (z > 0)
        & (u >= 0)
        & (v >= 0)
        & (u < cam.w)
        & (v < cam.h)
        & (px >= 0)
        & (px < cfg.PATCH_GRID_W)
        & (py >= 0)
        & (py < cfg.PATCH_GRID_H)
    )
    if not mask.any():
        return {
            "status": status,
            "visible_indices": empty,
            "u": np.array([], dtype=np.float32),
            "v": np.array([], dtype=np.float32),
            "z_cam": np.array([], dtype=np.float32),
            "patch_x": np.array([], dtype=np.int16),
            "patch_y": np.array([], dtype=np.int16),
        }

    idx = np.nonzero(mask)[0]
    status[idx] = STATUS_OCCLUDED

    ui_c = np.clip(ui[idx], 0, cam.w - 1)
    vi_c = np.clip(vi[idx], 0, cam.h - 1)
    depth_map = load_depth(frame_idx, cam.scale)
    depth_at = depth_map[vi_c, ui_c]
    z_vis = z[idx]

    occ = (depth_at > 0) & (np.abs(z_vis - depth_at) <= depth_tolerance)
    if not occ.any():
        return {
            "status": status,
            "visible_indices": empty,
            "u": np.array([], dtype=np.float32),
            "v": np.array([], dtype=np.float32),
            "z_cam": np.array([], dtype=np.float32),
            "patch_x": np.array([], dtype=np.int16),
            "patch_y": np.array([], dtype=np.int16),
        }

    keep = idx[occ]
    status[keep] = STATUS_VISIBLE
    return {
        "status": status,
        "visible_indices": keep.astype(np.int32),
        "u": u[keep].astype(np.float32),
        "v": v[keep].astype(np.float32),
        "z_cam": z_vis[occ].astype(np.float32),
        "patch_x": px[keep].astype(np.int16),
        "patch_y": py[keep].astype(np.int16),
    }


def build_visibility(
    points: np.ndarray,
    poses: np.ndarray,
    cam,
    depth_tolerance: float | None = None,
) -> dict[str, np.ndarray]:
    """
    For each point, find all visible (frame, patch) observations.

    Frame-outer vectorized loop: one depth load per frame, all points projected at once.
    """
    depth_tolerance = depth_tolerance if depth_tolerance is not None else cfg.DEPTH_TOLERANCE
    n_points = len(points)
    n_frames = len(poses)

    obs_point_idx: list[np.ndarray] = []
    obs_frame_idx: list[np.ndarray] = []
    obs_patch_x: list[np.ndarray] = []
    obs_patch_y: list[np.ndarray] = []
    obs_u: list[np.ndarray] = []
    obs_v: list[np.ndarray] = []
    obs_z_cam: list[np.ndarray] = []

    for fi in range(n_frames):
        frame_vis = project_frame_visibility(points, fi, poses, cam, depth_tolerance)
        keep = frame_vis["visible_indices"]
        if len(keep) == 0:
            continue

        obs_point_idx.append(keep)
        obs_frame_idx.append(np.full(len(keep), fi, dtype=np.int32))
        obs_patch_x.append(frame_vis["patch_x"])
        obs_patch_y.append(frame_vis["patch_y"])
        obs_u.append(frame_vis["u"])
        obs_v.append(frame_vis["v"])
        obs_z_cam.append(frame_vis["z_cam"])

        if (fi + 1) % 100 == 0 or fi + 1 == n_frames:
            print(f"  visibility: {fi + 1}/{n_frames} frames processed")

    if obs_point_idx:
        all_pi = np.concatenate(obs_point_idx)
        order = np.argsort(all_pi, kind="stable")
        obs_point_idx_arr = all_pi[order]
        obs_frame_idx_arr = np.concatenate(obs_frame_idx)[order]
        obs_patch_x_arr = np.concatenate(obs_patch_x)[order]
        obs_patch_y_arr = np.concatenate(obs_patch_y)[order]
        obs_u_arr = np.concatenate(obs_u)[order]
        obs_v_arr = np.concatenate(obs_v)[order]
        obs_z_cam_arr = np.concatenate(obs_z_cam)[order]
    else:
        obs_point_idx_arr = np.array([], dtype=np.int32)
        obs_frame_idx_arr = np.array([], dtype=np.int32)
        obs_patch_x_arr = np.array([], dtype=np.int16)
        obs_patch_y_arr = np.array([], dtype=np.int16)
        obs_u_arr = np.array([], dtype=np.float32)
        obs_v_arr = np.array([], dtype=np.float32)
        obs_z_cam_arr = np.array([], dtype=np.float32)

    point_obs_offsets = _build_point_offsets(obs_point_idx_arr, n_points)

    return {
        "obs_point_idx": obs_point_idx_arr,
        "obs_frame_idx": obs_frame_idx_arr,
        "obs_patch_x": obs_patch_x_arr,
        "obs_patch_y": obs_patch_y_arr,
        "obs_u": obs_u_arr,
        "obs_v": obs_v_arr,
        "obs_z_cam": obs_z_cam_arr,
        "point_obs_offsets": point_obs_offsets,
        "num_points": np.array([n_points], dtype=np.int32),
        "num_frames": np.array([n_frames], dtype=np.int32),
    }


def save_visibility(data: dict[str, np.ndarray], path: Path | None = None) -> Path:
    path = path or visibility_path()
    ensure_dir(path.parent)
    np.savez_compressed(path, **data)
    return path


def load_visibility(path: Path | None = None) -> dict[str, np.ndarray]:
    path = path or visibility_path()
    return dict(np.load(path))


def get_point_observations(vis: dict[str, np.ndarray], point_idx: int) -> dict[str, np.ndarray]:
    """Slice observations for a single point."""
    start = int(vis["point_obs_offsets"][point_idx])
    end = int(vis["point_obs_offsets"][point_idx + 1])
    return {
        "frame_idx": vis["obs_frame_idx"][start:end],
        "patch_x": vis["obs_patch_x"][start:end],
        "patch_y": vis["obs_patch_y"][start:end],
        "u": vis["obs_u"][start:end],
        "v": vis["obs_v"][start:end],
        "z_cam": vis["obs_z_cam"][start:end],
    }


def run_build_visibility(force: bool = False) -> dict[str, np.ndarray]:
    """Steps 1-2 combined: ensure points exist, build and save visibility."""
    out = visibility_path()
    if out.exists() and not force:
        print(f"Loading existing visibility from {out}")
        return load_visibility(out)

    pts_path = points_path(cfg.NUM_POINTS_INITIAL)
    if pts_path.exists() and not force:
        points = load_npy(pts_path)
    else:
        points, _ = run_sample_points(force=force)

    cam = load_camera_params()
    poses = load_trajectory()
    print(f"Building visibility for {len(points)} points x {len(poses)} frames...")
    vis = build_visibility(points, poses, cam)
    save_visibility(vis, out)
    total_obs = len(vis["obs_frame_idx"])
    avg_obs = total_obs / len(points)
    print(f"Saved visibility: {total_obs} observations, avg {avg_obs:.1f} per point")
    return vis


def main():
    import config.settings as settings

    settings.NUM_POINTS_INITIAL = 100
    settings.NUM_FRAMES = 100
    vis = run_build_visibility(force=True)
    obs0 = get_point_observations(vis, 0)
    print(f"Point 0: {len(obs0['frame_idx'])} visible observations")


if __name__ == "__main__":
    main()
