"""Replica dataset loaders: camera params, RGB, depth."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import cv2
import numpy as np

from config import settings as cfg
from src.utils.geometry import load_poses


@dataclass
class CameraParams:
    w: int
    h: int
    fx: float
    fy: float
    cx: float
    cy: float
    scale: float


def load_camera_params(path: str | None = None) -> CameraParams:
    path = path or cfg.CAM_PARAMS_PATH
    with open(path) as f:
        data = json.load(f)["camera"]
    return CameraParams(
        w=int(data["w"]),
        h=int(data["h"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        scale=float(data["scale"]),
    )


def load_trajectory(traj_path: str | None = None, num_frames: int | None = None) -> np.ndarray:
    traj_path = traj_path or cfg.TRAJ_PATH
    poses = load_poses(traj_path)
    n = num_frames if num_frames is not None else cfg.NUM_FRAMES
    if n is not None:
        poses = poses[:n]
    return poses


def frame_path(frame_idx: int) -> Path:
    return Path(cfg.FRAMES_DIR) / f"frame{frame_idx:06d}.jpg"


def depth_path(frame_idx: int) -> Path:
    return Path(cfg.FRAMES_DIR) / f"depth{frame_idx:06d}.png"


def load_rgb(frame_idx: int) -> np.ndarray:
    """Load RGB as HxWx3 uint8."""
    img = cv2.imread(str(frame_path(frame_idx)), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(frame_path(frame_idx))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(frame_idx: int, scale: float | None = None) -> np.ndarray:
    """Load depth in meters as HxW float32."""
    scale = scale if scale is not None else cfg.DEPTH_SCALE
    depth_raw = cv2.imread(str(depth_path(frame_idx)), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(depth_path(frame_idx))
    depth_m = depth_raw.astype(np.float32) / scale
    depth_m[depth_raw == 0] = 0.0
    return depth_m


def num_available_frames() -> int:
    return len(load_poses(cfg.TRAJ_PATH))


def main():
    cam = load_camera_params()
    poses = load_trajectory(num_frames=2)
    rgb = load_rgb(0)
    depth = load_depth(0, cam.scale)
    print(f"Camera: {cam.w}x{cam.h}, fx={cam.fx}")
    print(f"Poses: {poses.shape}, RGB: {rgb.shape}, depth range: {depth[depth > 0].min():.3f}-{depth[depth > 0].max():.3f}m")


if __name__ == "__main__":
    main()
