"""Mean aggregation of multi-view feature vectors."""

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
from src.utils.progress import tqdm

from config import settings as cfg
from src.data.mesh import output_dir
from src.data.projection import load_visibility
from src.data.replica import CameraParams, load_camera_params
from src.features.dinotxt_features import features_dinotxt_dir, load_frame_features
from src.utils.geometry import normalize_vectors
from src.utils.io import ensure_dir, save_npy


def aggregated_dir() -> Path:
    return ensure_dir(output_dir() / "aggregated")


def build_subsample_slot_map(
    n_orig: int,
    point_indices: np.ndarray | None,
    n_out: int,
) -> np.ndarray:
    """Map original visibility point index -> output slot (-1 if excluded)."""
    if point_indices is None:
        return np.arange(n_orig, dtype=np.int32)
    inv = np.full(n_orig, -1, dtype=np.int32)
    inv[point_indices.astype(np.int64)] = np.arange(n_out, dtype=np.int32)
    return inv


def _selected_observations_by_frame(
    vis: dict[str, np.ndarray],
    slot_map: np.ndarray,
    w_patches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gather only the observations whose point maps to an output slot, sorted by frame.

    Returns (frames, slots, patch_flat) for the selected observations. Sorting is
    done only over the selected subset (not all observations), keeping memory bounded
    when aggregating a subsample.
    """
    slots_all = slot_map[vis["obs_point_idx"].astype(np.int64)]
    keep = slots_all >= 0
    slots = slots_all[keep].astype(np.int64)
    del slots_all

    frames = vis["obs_frame_idx"][keep].astype(np.int64)
    px = vis["obs_patch_x"][keep].astype(np.int64)
    py = vis["obs_patch_y"][keep].astype(np.int64)
    patch_flat = py * w_patches + px
    del px, py

    order = np.argsort(frames, kind="stable")
    return frames[order], slots[order], patch_flat[order]


def _get_frame_features(fi: int, feature_dir: Path, frame_cache: dict[int, np.ndarray]) -> np.ndarray:
    if fi not in frame_cache:
        frame_cache[fi] = load_frame_features(fi, feature_dir)
    return frame_cache[fi]


def gather_point_features_dinotxt(
    vis: dict[str, np.ndarray],
    point_idx: int,
    point_indices: np.ndarray | None = None,
    feature_dir: Path | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Collect DINOtxt features for observations of a point.
    If point_indices is provided, map point_idx to original visibility index.
    Pass frame_cache to reuse loaded frame maps across many points (one load per frame).
    """
    feature_dir = feature_dir or features_dinotxt_dir()
    cache = frame_cache if frame_cache is not None else {}
    orig_idx = int(point_indices[point_idx]) if point_indices is not None else point_idx

    start = int(vis["point_obs_offsets"][orig_idx])
    end = int(vis["point_obs_offsets"][orig_idx + 1])
    if start == end:
        return np.zeros((0, cfg.DINOTXT_FEATURE_DIM), dtype=np.float32)

    frames = vis["obs_frame_idx"][start:end]
    px = vis["obs_patch_x"][start:end]
    py = vis["obs_patch_y"][start:end]

    feats = []
    for fi, pxi, pyi in zip(frames, px, py):
        fi = int(fi)
        img = _get_frame_features(fi, feature_dir, cache)
        feats.append(img[int(pyi), int(pxi)])
    return np.stack(feats, axis=0)


def observation_distance_and_angle(
    u: np.ndarray,
    v: np.ndarray,
    z_cam: np.ndarray,
    cam: CameraParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera-centre distance and viewing angle for each observation."""
    cam_x = (u.astype(np.float64) - cam.cx) * z_cam.astype(np.float64) / cam.fx
    cam_y = (v.astype(np.float64) - cam.cy) * z_cam.astype(np.float64) / cam.fy
    z = z_cam.astype(np.float64)
    dist = np.sqrt(cam_x * cam_x + cam_y * cam_y + z * z)
    cos_angle = np.clip(z / np.maximum(dist, 1e-12), -1.0, 1.0)
    return dist.astype(np.float32), np.arccos(cos_angle).astype(np.float32)


def compute_mean_observation_distance(vis: dict[str, np.ndarray], cam: CameraParams) -> float:
    """Global mean camera-centre distance across all stored observations."""
    if len(vis["obs_z_cam"]) == 0:
        return 1.0
    dist, _ = observation_distance_and_angle(vis["obs_u"], vis["obs_v"], vis["obs_z_cam"], cam)
    return float(dist.mean())


def gather_point_features_dinotxt_with_geometry(
    vis: dict[str, np.ndarray],
    point_idx: int,
    cam: CameraParams,
    point_indices: np.ndarray | None = None,
    feature_dir: Path | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect DINOtxt features plus per-view distance and viewing angle."""
    feature_dir = feature_dir or features_dinotxt_dir()
    cache = frame_cache if frame_cache is not None else {}
    orig_idx = int(point_indices[point_idx]) if point_indices is not None else point_idx

    start = int(vis["point_obs_offsets"][orig_idx])
    end = int(vis["point_obs_offsets"][orig_idx + 1])
    if start == end:
        empty = np.zeros((0, cfg.DINOTXT_FEATURE_DIM), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    frames = vis["obs_frame_idx"][start:end]
    px = vis["obs_patch_x"][start:end]
    py = vis["obs_patch_y"][start:end]
    u = vis["obs_u"][start:end]
    v = vis["obs_v"][start:end]
    z_cam = vis["obs_z_cam"][start:end]
    distances, angles = observation_distance_and_angle(u, v, z_cam, cam)

    feats = []
    for fi, pxi, pyi in zip(frames, px, py):
        fi = int(fi)
        img = _get_frame_features(fi, feature_dir, cache)
        feats.append(img[int(pyi), int(pxi)])
    return np.stack(feats, axis=0), distances, angles


def mean_aggregate(feats: np.ndarray) -> np.ndarray:
    """L2-normalize each view, mean, re-normalize."""
    if len(feats) == 0:
        return np.zeros(cfg.DINOTXT_FEATURE_DIM, dtype=np.float32)
    normed = normalize_vectors(feats)
    mean = normed.mean(axis=0)
    return normalize_vectors(mean.reshape(1, -1)).squeeze(0).astype(np.float32)


def aggregate_mean_vectorized(
    vis: dict[str, np.ndarray],
    n_points: int,
    point_indices: np.ndarray | None = None,
    feature_dir: Path | None = None,
) -> np.ndarray:
    """
    Frame-outer vectorized mean aggregation.

    Reads each DINOtxt frame at most once (one frame in memory at a time),
    batch-gathers only the needed patches, and accumulates L2-normalized vectors
    per point (same math as mean_aggregate). Cost scales with the number of
    frames, not the number of points.
    """
    feature_dir = feature_dir or features_dinotxt_dir()
    n_orig = int(vis["num_points"][0])
    dim = cfg.DINOTXT_FEATURE_DIM
    w_patches = cfg.PATCH_GRID_W
    slot_map = build_subsample_slot_map(n_orig, point_indices, n_points)

    running_sums = np.zeros((n_points, dim), dtype=np.float64)
    num_obs = np.zeros(n_points, dtype=np.int32)

    frames, slots, patch_flat = _selected_observations_by_frame(vis, slot_map, w_patches)
    if len(frames) == 0:
        return np.zeros((n_points, dim), dtype=np.float32)

    unique_frames, frame_starts = np.unique(frames, return_index=True)
    frame_ends = np.append(frame_starts[1:], len(frames))

    for fi, start, end in tqdm(
        zip(unique_frames, frame_starts, frame_ends),
        total=len(unique_frames),
        desc="Aggregate (mean)",
    ):
        fi = int(fi)
        feat_path = feature_dir / f"{fi:06d}.npy"
        if not feat_path.exists():
            continue

        img_features = load_frame_features(fi, feature_dir)
        flat = img_features.reshape(-1, dim)
        extracted = flat[patch_flat[start:end]]
        normalized = normalize_vectors(extracted.astype(np.float64))
        del img_features, flat, extracted

        np.add.at(running_sums, slots[start:end], normalized)
        np.add.at(num_obs, slots[start:end], 1)

    aggregated = np.zeros((n_points, dim), dtype=np.float32)
    valid = num_obs > 0
    if valid.any():
        mean_vectors = running_sums[valid] / num_obs[valid, None]
        aggregated[valid] = normalize_vectors(mean_vectors).astype(np.float32)
    return aggregated


def iter_point_features_chunked(
    vis: dict[str, np.ndarray],
    n_points: int,
    point_indices: np.ndarray | None = None,
    cam: CameraParams | None = None,
    feature_dir: Path | None = None,
    max_obs_per_chunk: int | None = None,
):
    """
    Yield ``(slot, feats, distances, angles)`` per output point using a
    memory-bounded frame-streaming gather.

    Points are processed in chunks capped at ``max_obs_per_chunk`` observations.
    For each chunk every relevant frame is read at most once and only one frame
    plus the chunk's feature buffer (<= max_obs_per_chunk x dim) live in memory.
    ``distances`` and ``angles`` are ``None`` when ``cam`` is not provided.

    Slots are yielded in increasing order 0..n_points-1, so callers may wrap the
    generator with a progress bar (``tqdm(gen, total=n_points)``).

    **Why progress looks bursty:** each chunk first loads ~2000 frame files from
    disk (slow, no per-point tqdm ticks), then runs weighted SLERP on ~600 points
    (fast). Larger ``max_obs_per_chunk`` → fewer chunks → fewer full disk passes.
    """
    feature_dir = feature_dir or features_dinotxt_dir()
    max_obs_per_chunk = max_obs_per_chunk or cfg.AGGREGATION_MAX_OBS_PER_CHUNK
    use_mmap = cfg.AGGREGATION_FRAME_MMAP
    dim = cfg.DINOTXT_FEATURE_DIM
    w_patches = cfg.PATCH_GRID_W
    offsets = vis["point_obs_offsets"]

    def orig_index(slot: int) -> int:
        return int(point_indices[slot]) if point_indices is not None else slot

    chunk_idx = 0

    def flush(members: list[tuple[int, int, int]]):
        nonlocal chunk_idx
        if not members:
            return
        chunk_idx += 1
        n_obs = sum(e - s for _, s, e in members)
        n_pts = len(members)
        glob_idx = np.concatenate([np.arange(s, e) for _, s, e in members])
        frames = vis["obs_frame_idx"][glob_idx].astype(np.int64)
        px = vis["obs_patch_x"][glob_idx].astype(np.int64)
        py = vis["obs_patch_y"][glob_idx].astype(np.int64)
        patch_flat = py * w_patches + px

        if cam is not None:
            distances, angles = observation_distance_and_angle(
                vis["obs_u"][glob_idx], vis["obs_v"][glob_idx], vis["obs_z_cam"][glob_idx], cam
            )
        else:
            distances = angles = None

        feats_flat = np.zeros((len(glob_idx), dim), dtype=np.float32)
        order = np.argsort(frames, kind="stable")
        unique_frames, fstart = np.unique(frames[order], return_index=True)
        fend = np.append(fstart[1:], len(order))
        print(
            f"  chunk {chunk_idx}: gathering {n_obs:,} obs for {n_pts:,} points "
            f"from {len(unique_frames):,} frames..."
        )
        for fi, a, b in tqdm(
            zip(unique_frames, fstart, fend),
            total=len(unique_frames),
            desc=f"  chunk {chunk_idx} frames",
            leave=False,
        ):
            fi = int(fi)
            feat_path = feature_dir / f"{fi:06d}.npy"
            if not feat_path.exists():
                continue
            img = load_frame_features(fi, feature_dir, mmap=use_mmap)
            flat = img.reshape(-1, dim)
            pos = order[a:b]
            feats_flat[pos] = np.asarray(flat[patch_flat[pos]], dtype=np.float32)
            del img, flat

        cursor = 0
        for slot, s, e in members:
            length = e - s
            sl = slice(cursor, cursor + length)
            if cam is not None:
                yield slot, feats_flat[sl], distances[sl], angles[sl]
            else:
                yield slot, feats_flat[sl], None, None
            cursor += length

    members: list[tuple[int, int, int]] = []
    obs_in_chunk = 0
    for slot in range(n_points):
        orig = orig_index(slot)
        start = int(offsets[orig])
        end = int(offsets[orig + 1])
        if start == end:
            # No observations: emit empty so caller can write a zero vector.
            empty_g = np.zeros(0, dtype=np.float32) if cam is not None else None
            yield slot, np.zeros((0, dim), dtype=np.float32), empty_g, empty_g
            continue

        members.append((slot, start, end))
        obs_in_chunk += end - start
        if obs_in_chunk >= max_obs_per_chunk:
            yield from flush(members)
            members = []
            obs_in_chunk = 0

    yield from flush(members)


def aggregate_all_points(
    n_points: int,
    point_indices: np.ndarray | None = None,
    vis: dict | None = None,
    method_name: str = "mean",
    force: bool = False,
) -> Path:
    """Step 7 (mean): aggregate features for each point."""
    out_path = aggregated_dir() / f"{method_name}.npy"
    if out_path.exists() and not force:
        print(f"Aggregated features exist at {out_path}")
        return out_path

    vis = vis or load_visibility()
    aggregated = aggregate_mean_vectorized(vis, n_points, point_indices)
    save_npy(out_path, aggregated)
    print(f"Saved aggregated features to {out_path}")
    return out_path


def main():
    from src.utils.io import load_npy

    vis = load_visibility()
    aggregated = np.zeros((10, cfg.DINOTXT_FEATURE_DIM), dtype=np.float32)
    for pi in range(10):
        feats = gather_point_features_dinotxt(vis, pi)
        aggregated[pi] = mean_aggregate(feats)
    print(f"Mean aggregated {len(aggregated)} points, norms={np.linalg.norm(aggregated, axis=1)[:3]}")


if __name__ == "__main__":
    main()
