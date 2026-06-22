"""SE(3), intrinsics, and patch mapping utilities."""

from __future__ import annotations

import numpy as np


def load_poses(traj_path: str) -> np.ndarray:
    """Load camera-to-world poses. Returns (N, 4, 4) float64."""
    poses = []
    with open(traj_path) as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            if len(vals) != 16:
                continue
            poses.append(np.array(vals, dtype=np.float64).reshape(4, 4))
    return np.stack(poses, axis=0)


def invert_pose(c2w: np.ndarray) -> np.ndarray:
    """Invert a 4x4 SE(3) matrix."""
    w2c = np.eye(4, dtype=np.float64)
    r = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = r.T
    w2c[:3, 3] = -r.T @ t
    return w2c


def world_to_camera(points_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Transform Nx3 world points to camera space."""
    ones = np.ones((points_world.shape[0], 1), dtype=np.float64)
    homo = np.hstack([points_world.astype(np.float64), ones])
    cam = (w2c @ homo.T).T[:, :3]
    return cam


def project_to_pixels(cam_pts: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-space points to pixel coords. Returns u, v, z."""
    z = cam_pts[:, 2]
    u = fx * cam_pts[:, 0] / z + cx
    v = fy * cam_pts[:, 1] / z + cy
    return u, v, z


def pixel_to_patch(u: float, v: float, patch_size: int = 16) -> tuple[int, int]:
    """Convert pixel coordinate to patch index."""
    px = int(u // patch_size)
    py = int(v // patch_size)
    return px, py


def normalize_vectors(vecs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / np.maximum(norms, eps)
