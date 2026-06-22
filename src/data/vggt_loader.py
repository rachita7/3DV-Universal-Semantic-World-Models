"""VGGT / COLMAP geometry source: read SfM binaries, build visibility.

Alternative to the Replica depth projection (step 2). Reads COLMAP-style
``images.bin`` / ``points3D.bin`` produced by VGGT, exposes the point -> views
and camera lookups, and converts them into the project's ``visibility.npz``
schema so the downstream feature / aggregation steps run unchanged.

VGGT mode supplies geometry only (no mesh GT labels), so the GT-dependent steps
(1, 4, 5, 9) remain Replica-specific.
"""

from __future__ import annotations

import importlib.util
import struct
from dataclasses import dataclass
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import numpy as np

from config import settings as cfg
from src.data.mesh import output_dir
from src.data.projection import save_visibility, visibility_path
from src.utils.io import save_npy


# --- COLMAP binary readers (self-contained; ported from inter_image_distance) ---

@dataclass
class Image:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray


@dataclass
class Point3D:
    point3d_id: int
    xyz: np.ndarray


def _read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected EOF while reading binary model file.")
    return struct.unpack(fmt, data)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) -> world->camera rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def camera_center_world(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Camera centre in world coords: C = -R^T t (pose is world->camera)."""
    r_wc = qvec_to_rotmat(qvec)
    return -r_wc.T @ tvec


def read_images_binary(path: str | Path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "<Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "<idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]

            name_bytes = bytearray()
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                if char == b"":
                    raise EOFError("Unexpected EOF while reading image name.")
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8")

            num_points2d = _read_next_bytes(fid, 8, "<Q")[0]
            xyid = _read_next_bytes(fid, 24 * num_points2d, "<" + "ddq" * num_points2d)
            xs = np.array(xyid[0::3], dtype=np.float64)
            ys = np.array(xyid[1::3], dtype=np.float64)
            point3d_ids = np.array(xyid[2::3], dtype=np.int64)
            xys = np.column_stack([xs, ys]) if num_points2d else np.empty((0, 2), dtype=np.float64)

            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, point3d_ids)
    return images


def read_points3d_binary(path: str | Path) -> dict[int, Point3D]:
    points3d: dict[int, Point3D] = {}
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "<Q")[0]
        for _ in range(num_points):
            props = _read_next_bytes(fid, 43, "<QdddBBBd")
            point3d_id = props[0]
            xyz = np.array(props[1:4], dtype=np.float64)
            track_length = _read_next_bytes(fid, 8, "<Q")[0]
            fid.read(8 * track_length)  # (image_id:int32, point2D_idx:int32) per element
            points3d[point3d_id] = Point3D(point3d_id, xyz)
    return points3d


class VGGTLoader:
    """Reads VGGT / COLMAP binary outputs and builds structured lookups."""

    def __init__(self, images_bin_path: str | None = None, points3d_bin_path: str | None = None):
        images_bin_path = images_bin_path or cfg.VGGT_IMAGES_BIN
        points3d_bin_path = points3d_bin_path or cfg.VGGT_POINTS3D_BIN
        self._images = read_images_binary(images_bin_path)
        self._points3d = read_points3d_binary(points3d_bin_path)
        self._rotation_cache = {img.name: qvec_to_rotmat(img.qvec) for img in self._images.values()}

    def get_point_to_views(self) -> dict[tuple[float, float, float], dict[str, list]]:
        """Map each reconstructed 3D point to its 2D observations per image.

        Returns ``{(x, y, z): {image_name: [np.array([u, v])]}}``.
        """
        result: dict[tuple[float, float, float], dict[str, list]] = {}
        for image in self._images.values():
            for idx, point3d_id in enumerate(image.point3d_ids):
                if point3d_id < 0 or point3d_id not in self._points3d:
                    continue
                key = tuple(self._points3d[point3d_id].xyz.tolist())
                result.setdefault(key, {})[image.name] = [image.xys[idx].copy()]
        return result

    def get_camera_centers(self) -> dict[str, np.ndarray]:
        """World-space camera centre per image: ``{image_name: [cx, cy, cz]}``."""
        return {img.name: camera_center_world(img.qvec, img.tvec) for img in self._images.values()}

    def get_camera_rotations(self) -> dict[str, np.ndarray]:
        """World->camera rotation per image: ``{image_name: R (3x3)}``."""
        return dict(self._rotation_cache)


def _build_point_offsets(obs_point_idx: np.ndarray, n_points: int) -> np.ndarray:
    """CSR offsets from (point-sorted) observation point indices."""
    counts = np.bincount(obs_point_idx, minlength=n_points)
    offsets = np.zeros(n_points + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return offsets


def build_vggt_visibility(
    loader: VGGTLoader,
    patch_size: int | None = None,
    grid_w: int | None = None,
    grid_h: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert VGGT lookups into ``(points, visibility_dict)``.

    Observations are kept only when their pixel falls inside the patch grid and
    the point lies in front of the camera (z > 0). Image names are mapped to
    frame indices by sorted order, matching the per-frame feature file naming.
    """
    patch_size = patch_size or cfg.PATCH_SIZE
    grid_w = grid_w or cfg.PATCH_GRID_W
    grid_h = grid_h or cfg.PATCH_GRID_H

    point_to_views = loader.get_point_to_views()
    centers = loader.get_camera_centers()
    rotations = loader.get_camera_rotations()
    name_to_frame = {name: fi for fi, name in enumerate(sorted(centers))}

    points: list[list[float]] = []
    obs_pt, obs_fr, obs_px, obs_py = [], [], [], []
    obs_u, obs_v, obs_z = [], [], []

    for pi, (xyz, views) in enumerate(point_to_views.items()):
        points.append(list(xyz))
        p = np.asarray(xyz, dtype=np.float64)
        for name, uv_list in views.items():
            if name not in name_to_frame:
                continue
            u, v = float(uv_list[0][0]), float(uv_list[0][1])
            px, py = int(u // patch_size), int(v // patch_size)
            if not (0 <= px < grid_w and 0 <= py < grid_h):
                continue
            z = float((rotations[name] @ (p - centers[name]))[2])
            if z <= 0:
                continue
            obs_pt.append(pi)
            obs_fr.append(name_to_frame[name])
            obs_px.append(px)
            obs_py.append(py)
            obs_u.append(u)
            obs_v.append(v)
            obs_z.append(z)

    n_points = len(points)
    n_frames = len(name_to_frame)
    points_arr = np.asarray(points, dtype=np.float32).reshape(n_points, 3)
    obs_point_idx = np.asarray(obs_pt, dtype=np.int32)

    vis = {
        "obs_point_idx": obs_point_idx,
        "obs_frame_idx": np.asarray(obs_fr, dtype=np.int32),
        "obs_patch_x": np.asarray(obs_px, dtype=np.int16),
        "obs_patch_y": np.asarray(obs_py, dtype=np.int16),
        "obs_u": np.asarray(obs_u, dtype=np.float32),
        "obs_v": np.asarray(obs_v, dtype=np.float32),
        "obs_z_cam": np.asarray(obs_z, dtype=np.float32),
        "point_obs_offsets": _build_point_offsets(obs_point_idx, n_points),
        "num_points": np.array([n_points], dtype=np.int32),
        "num_frames": np.array([n_frames], dtype=np.int32),
    }
    return points_arr, vis


def vggt_points_path() -> Path:
    return output_dir() / "vggt_points.npy"


def run_build_vggt_visibility(force: bool = False) -> dict[str, np.ndarray]:
    """Step 2 (VGGT source): build + save visibility from COLMAP binaries."""
    out = visibility_path()
    if out.exists() and not force:
        print(f"Loading existing visibility from {out}")
        return dict(np.load(out))

    loader = VGGTLoader()
    print(f"Building VGGT visibility from {cfg.VGGT_IMAGES_BIN} / {cfg.VGGT_POINTS3D_BIN} ...")
    points, vis = build_vggt_visibility(loader)
    save_visibility(vis, out)
    save_npy(vggt_points_path(), points)
    total_obs = len(vis["obs_frame_idx"])
    n_points = int(vis["num_points"][0])
    avg = total_obs / n_points if n_points else 0.0
    print(
        f"Saved VGGT visibility: {n_points} points x {int(vis['num_frames'][0])} frames, "
        f"{total_obs} observations (avg {avg:.1f}/point)"
    )
    return vis


def _write_synthetic_colmap(images_path: Path, points_path: Path) -> None:
    """Write tiny synthetic COLMAP binaries (for the smoke test)."""
    pts = {1: (0.0, 0.0, 5.0), 2: (1.0, 0.5, 6.0), 3: (-1.0, 0.0, 4.0)}
    with open(points_path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for pid, xyz in pts.items():
            f.write(struct.pack("<Q", pid))
            f.write(struct.pack("<ddd", *xyz))
            f.write(struct.pack("<BBB", 100, 100, 100))
            f.write(struct.pack("<d", 0.5))
            f.write(struct.pack("<Q", 0))  # empty track

    images = {
        1: ("frame000000.jpg", [(10.0, 12.0, 1), (40.0, 20.0, 2), (5.0, 30.0, 3)]),
        2: ("frame000001.jpg", [(15.0, 18.0, 1), (50.0, 22.0, 2)]),
    }
    with open(images_path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img_id, (name, obs) in images.items():
            f.write(struct.pack("<i", img_id))
            f.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))  # identity quaternion
            f.write(struct.pack("<ddd", 0.0, 0.0, 0.0))  # zero translation
            f.write(struct.pack("<i", 1))  # camera_id
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(obs)))
            for u, v, pid in obs:
                f.write(struct.pack("<ddq", u, v, pid))


def main() -> None:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="vggt_loader_test_"))
    images_bin = tmp / "images.bin"
    points3d_bin = tmp / "points3D.bin"
    _write_synthetic_colmap(images_bin, points3d_bin)

    loader = VGGTLoader(str(images_bin), str(points3d_bin))
    ptv = loader.get_point_to_views()
    centers = loader.get_camera_centers()
    print(f"Loaded {len(ptv)} points across {len(centers)} images")

    points, vis = build_vggt_visibility(loader)
    print(
        f"Visibility: points={points.shape}, observations={len(vis['obs_frame_idx'])}, "
        f"frames={int(vis['num_frames'][0])}"
    )
    assert points.shape[0] == int(vis["num_points"][0])
    assert vis["point_obs_offsets"][-1] == len(vis["obs_frame_idx"])
    print("vggt_loader smoke test OK")


if __name__ == "__main__":
    main()
