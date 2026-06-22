"""Load Replica mesh, sample points, assign GT labels."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import numpy as np
import trimesh

from config import settings as cfg
from src.utils.io import ensure_dir, load_npy, save_npy


def load_semantic_mesh(mesh_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mesh_semantic.ply with per-face object_id.

    Returns:
        vertices (V, 3), faces (F, 4) quad indices, face_object_ids (F,)
    """
    with open(mesh_path, "rb") as f:
        mesh = trimesh.exchange.ply.load_ply(f)

    vertices = np.asarray(mesh["vertices"], dtype=np.float64)

    ply_raw = mesh.get("metadata", {}).get("_ply_raw")
    if ply_raw is None or "face" not in ply_raw:
        raise ValueError("mesh_semantic.ply missing raw face data")

    face_data = ply_raw["face"]["data"]
    face_object_ids = np.asarray(face_data["object_id"], dtype=np.int32)

    # vertex_indices: variable-length quads stored as (count, [i0,i1,i2,i3])
    face_indices = []
    for row in face_data["vertex_indices"]:
        count = int(row[0])
        indices = [int(row[1][j]) for j in range(count)]
        face_indices.append(indices)

    max_len = max(len(f) for f in face_indices)
    faces = np.full((len(face_indices), max_len), -1, dtype=np.int64)
    for i, fi in enumerate(face_indices):
        faces[i, : len(fi)] = fi

    return vertices, faces, face_object_ids


def vertex_labels_from_faces(faces: np.ndarray, face_object_ids: np.ndarray, num_vertices: int) -> np.ndarray:
    """Majority vote of incident face object_ids per vertex."""
    vertex_votes: list[list[int]] = [[] for _ in range(num_vertices)]
    for fi in range(len(faces)):
        oid = int(face_object_ids[fi])
        face = faces[fi]
        for vi in face:
            if vi >= 0:
                vertex_votes[int(vi)].append(oid)

    labels = np.zeros(num_vertices, dtype=np.int32)
    for i, votes in enumerate(vertex_votes):
        if votes:
            counts = Counter(votes)
            max_count = max(counts.values())
            candidates = [k for k, v in counts.items() if v == max_count]
            labels[i] = min(candidates)
    return labels


def load_info_semantic(path: str) -> tuple[dict[int, str], dict[int, str]]:
    """
    Returns:
        object_id -> class_name
        class_id -> class_name (from classes list)
    """
    with open(path) as f:
        info = json.load(f)

    obj_to_class: dict[int, str] = {}
    for obj in info["objects"]:
        obj_to_class[int(obj["id"])] = obj["class_name"]

    class_id_to_name: dict[int, str] = {}
    for cls in info["classes"]:
        class_id_to_name[int(cls["id"])] = cls["name"]

    return obj_to_class, class_id_to_name


def sample_mesh_points(
    num_points: int,
    seed: int,
    mesh_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uniformly sample vertices (without replacement if num_points < V).

    Returns:
        points (N, 3), vertex_indices (N,), gt_object_ids (N,)
    """
    mesh_path = mesh_path or cfg.MESH_PATH
    vertices, faces, face_object_ids = load_semantic_mesh(mesh_path)
    v_labels = vertex_labels_from_faces(faces, face_object_ids, len(vertices))

    rng = np.random.default_rng(seed)
    n = min(num_points, len(vertices))
    indices = rng.choice(len(vertices), size=n, replace=False)
    points = vertices[indices].astype(np.float32)
    gt_object_ids = v_labels[indices]
    return points, indices.astype(np.int64), gt_object_ids


def object_ids_to_class_names(object_ids: np.ndarray, obj_to_class: dict[int, str]) -> np.ndarray:
    names = np.array([obj_to_class.get(int(oid), "unknown") for oid in object_ids], dtype=object)
    return names


def subsample_by_variance(
    dispersion: np.ndarray,
    num_subsample: int,
    seed: int,
) -> np.ndarray:
    """Variance-weighted subsample without replacement. Returns indices into original array."""
    rng = np.random.default_rng(seed)
    weights = dispersion.astype(np.float64)
    weights = np.maximum(weights, 1e-8)
    weights /= weights.sum()
    n = min(num_subsample, len(dispersion))
    return rng.choice(len(dispersion), size=n, replace=False, p=weights)


def output_dir() -> Path:
    return ensure_dir(Path(cfg.OUTPUT_ROOT) / cfg.ROOM)


def points_path(num: int) -> Path:
    return output_dir() / f"points_{num}.npy"


def gt_labels_path(num: int) -> Path:
    return output_dir() / f"gt_labels_{num}.npy"


def run_sample_points(force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Step 1: sample points and save."""
    out_pts = points_path(cfg.NUM_POINTS_INITIAL)
    out_gt = gt_labels_path(cfg.NUM_POINTS_INITIAL)
    if out_pts.exists() and out_gt.exists() and not force:
        return load_npy(out_pts), load_npy(out_gt)

    points, _, gt_ids = sample_mesh_points(cfg.NUM_POINTS_INITIAL, cfg.RANDOM_SEED)
    save_npy(out_pts, points)
    save_npy(out_gt, gt_ids)
    print(f"Saved {len(points)} points to {out_pts}")
    return points, gt_ids


def run_subsample_points(dispersion_path: Path, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Step 5: variance-weighted subsample."""
    out_idx = output_dir() / f"points_{cfg.NUM_POINTS_SUBSAMPLE}_idx.npy"
    out_pts = points_path(cfg.NUM_POINTS_SUBSAMPLE)
    if out_idx.exists() and out_pts.exists() and not force:
        idx = load_npy(out_idx)
        return idx, load_npy(out_pts)

    dispersion = load_npy(dispersion_path)
    all_points = load_npy(points_path(cfg.NUM_POINTS_INITIAL))
    idx = subsample_by_variance(dispersion, cfg.NUM_POINTS_SUBSAMPLE, cfg.RANDOM_SEED + 1)
    save_npy(out_idx, idx.astype(np.int32))
    save_npy(out_pts, all_points[idx])
    print(f"Saved {len(idx)} subsampled points to {out_pts}")
    return idx, all_points[idx]


def main():
    points, gt = run_sample_points(force=True)
    obj_to_class, _ = load_info_semantic(cfg.INFO_SEMANTIC_PATH)
    names = object_ids_to_class_names(gt[:10], obj_to_class)
    print(f"Sampled {len(points)} points, first 10 classes: {names}")


if __name__ == "__main__":
    main()
