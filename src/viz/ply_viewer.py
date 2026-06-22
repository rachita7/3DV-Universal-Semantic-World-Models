"""Interactive viewer for pipeline PLY point clouds (Open3D)."""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
import sys
from pathlib import Path

_bspec = importlib.util.spec_from_file_location(
    "project_bootstrap",
    Path(__file__).resolve().parents[1] / "bootstrap.py",
)
_bmod = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bmod)

import re

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from config import settings as cfg
from src.aggregation.variance import dispersion_path, variance_dir
from src.data.mesh import (
    load_info_semantic,
    load_semantic_mesh as load_replica_semantic_mesh,
    object_ids_to_class_names,
    output_dir,
    points_path,
    vertex_labels_from_faces,
)
from src.segmentation.classify import results_dir
from src.utils.io import load_npy
from src.viz.gt_labels import class_to_color
from src.viz.variance_heatmap import dispersion_to_colors
from src.viz.segmentation_viz import COLOR_FN, COLOR_FP, COLOR_TP, COLOR_UNKNOWN

# Fixed room camera used by both the interactive viewer and PNG export.
# This gives a stable isometric-style view of Replica room2 for comparisons.
ISO_FRONT = np.asarray([0.58, -0.58, 0.57], dtype=np.float64)
ISO_UP = np.asarray([-0.35, 0.35, 0.87], dtype=np.float64)
ISO_ZOOM = 0.72
ISO_FALLBACK_ELEV = 32.0
ISO_FALLBACK_AZIM = -45.0

# Neutral mesh backdrop for sparse segmentation-error overlays (not sampled-point unknown).
MESH_BACKGROUND_GRAY = (160, 160, 160)
_VERTEX_COORD_DECIMALS = 6
# Open3D point_size for TP/FP markers overlaid on the grey error mesh (10× default 3.0).
SEGMENTATION_ERROR_TP_FP_POINT_SIZE = 10.0


def collect_ply_paths(steps: list[int] | None = None) -> list[tuple[str, Path]]:
    """Return (label, path) pairs for PLY files produced by the given pipeline steps."""
    steps = steps or list(range(1, 10))
    out: list[tuple[str, Path]] = []
    root = output_dir()

    if 1 in steps:
        for name in ("gt_chair_table.ply", "gt_multiclass.ply"):
            path = root / "viz" / name
            if path.exists():
                out.append((f"GT labels ({name})", path))

    if 4 in steps:
        path = variance_dir() / "heatmap.ply"
        if path.exists():
            out.append(("Variance heatmap", path))

    if 9 in steps:
        for method in cfg.ABLATION_METHODS:
            path = results_dir() / f"segmentation_errors_{method}.ply"
            if path.exists():
                out.append((f"Segmentation errors ({method})", path))
        # Fallback if ablation not run
        if not any(m in label for label, _ in out for m in cfg.ABLATION_METHODS):
            path = results_dir() / f"segmentation_errors_{cfg.AGGREGATION_METHOD}.ply"
            if path.exists():
                out.append((f"Segmentation errors ({cfg.AGGREGATION_METHOD})", path))

    return out


def collect_mesh_items(steps: list[int] | None = None) -> list[tuple[str, Path]]:
    """Return semantic mesh items to visualize in addition to sampled PLY clouds."""
    steps = steps or list(range(1, 10))
    if 1 not in steps:
        return []
    mesh_path = Path(cfg.MESH_PATH)
    if mesh_path.exists():
        return [("Semantic mesh (GT labels)", mesh_path)]
    return []


def viz_png_dir() -> Path:
    """PNG exports of pipeline PLY point clouds (same sources as the interactive viewer)."""
    out = output_dir() / "viz" / "png"
    out.mkdir(parents=True, exist_ok=True)
    return out


def png_name_for_ply(ply_path: Path, label: str | None = None) -> str:
    """Stable PNG filename from PLY path or viewer label."""
    if label and "segmentation errors" in label.lower():
        m = re.search(r"\((\w+)\)", label)
        method = m.group(1) if m else ply_path.stem.replace("segmentation_errors_", "")
        return f"segmentation_errors_{method}.png"
    return f"{ply_path.stem}.png"


def load_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (N, 3) points and (N, 3) uint8 RGB from an ASCII or binary PLY."""
    path = Path(path)
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points, dtype=np.float32)
        if len(points) == 0:
            return points, np.zeros((0, 3), dtype=np.uint8)
        if pcd.has_colors():
            colors = _colors_to_uint8(np.asarray(pcd.colors))
        else:
            colors = np.full((len(points), 3), 128, dtype=np.uint8)
        return points, colors
    except ImportError:
        pass

    return _load_ascii_colored_ply(path)


def load_export_points_colors(label: str, ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load point/color data for PNG export, applying export-specific palettes."""
    if "variance heatmap" in label.lower():
        # Force the variance PNG to use the current jet color scheme even if an
        # older cached heatmap.ply was written with a different colormap.
        points = load_npy(points_path(cfg.NUM_POINTS_INITIAL)).astype(np.float32)
        dispersion = load_npy(dispersion_path())
        return points, dispersion_to_colors(dispersion)
    return load_colored_ply(ply_path)


def _load_ascii_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal ASCII PLY reader for our colored point clouds (no Open3D)."""
    with open(path) as f:
        line = f.readline().strip()
        if line != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        while True:
            header = f.readline().strip()
            if header.startswith("element vertex"):
                n = int(header.split()[-1])
                break
            if header == "end_header":
                raise ValueError(f"PLY missing vertex count: {path}")
        while f.readline().strip() != "end_header":
            pass
        data = np.loadtxt(f, max_rows=n)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    points = data[:, :3].astype(np.float32)
    colors = np.clip(data[:, 3:6], 0, 255).astype(np.uint8)
    return points, colors


def _triangulate_faces(faces: np.ndarray) -> np.ndarray:
    """Fan-triangulate variable-length Replica mesh faces."""
    tris: list[list[int]] = []
    for face in faces:
        valid = [int(v) for v in face if int(v) >= 0]
        if len(valid) < 3:
            continue
        for i in range(1, len(valid) - 1):
            tris.append([valid[0], valid[i], valid[i + 1]])
    return np.asarray(tris, dtype=np.int32)


@lru_cache(maxsize=1)
def load_semantic_mesh_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load mesh vertices, triangles, and per-vertex GT class colors."""
    vertices, faces, face_object_ids = load_replica_semantic_mesh(cfg.MESH_PATH)
    triangles = _triangulate_faces(faces)
    vertex_object_ids = vertex_labels_from_faces(faces, face_object_ids, len(vertices))
    obj_to_class, _ = load_info_semantic(cfg.INFO_SEMANTIC_PATH)
    class_names = object_ids_to_class_names(vertex_object_ids, obj_to_class)
    colors = np.asarray([class_to_color(str(cn), highlight=None) for cn in class_names], dtype=np.uint8)
    return vertices.astype(np.float32), triangles, colors


def _vertex_coord_keys(points: np.ndarray) -> np.ndarray:
    """Round vertex coordinates for exact mesh-vertex lookup."""
    return np.round(points.astype(np.float64), _VERTEX_COORD_DECIMALS)


def _point_color_lookup(
    source_points: np.ndarray,
    source_colors: np.ndarray,
) -> dict[tuple[float, float, float], np.ndarray]:
    return {
        tuple(key): color
        for key, color in zip(_vertex_coord_keys(source_points), source_colors)
    }


def is_segmentation_errors_viz(label: str | None = None, path: Path | str | None = None) -> bool:
    """True when a PLY/label refers to step-9 segmentation error visualization."""
    if label and "segmentation error" in label.lower():
        return True
    if path is not None:
        stem = Path(path).stem.lower()
        if "segmentation_error" in stem or stem in {"errors", "error"}:
            return True
        if stem.startswith("errors_"):
            return True
    return False


def transfer_point_colors_to_mesh_vertices(
    mesh_vertices: np.ndarray,
    source_points: np.ndarray,
    source_colors: np.ndarray,
) -> np.ndarray:
    """
    Transfer a colored point-cloud visualization onto the full mesh vertices.

    The pipeline point clouds are sampled mesh vertices (100k or 50k subsets).
    Nearest-neighbor transfer fills every mesh vertex, producing a dense surface
    visualization for variance / GT heatmaps instead of sparse dots.
    """
    if len(source_points) == 0:
        return np.full((len(mesh_vertices), 3), 128, dtype=np.uint8)

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return transfer_point_colors_to_mesh_vertices_sparse(
            mesh_vertices,
            source_points,
            source_colors,
        )

    tree = cKDTree(source_points.astype(np.float64))
    _dist, idx = tree.query(mesh_vertices.astype(np.float64), k=1, workers=-1)
    return source_colors[idx].astype(np.uint8)


def transfer_point_colors_to_mesh_vertices_sparse(
    mesh_vertices: np.ndarray,
    source_points: np.ndarray,
    source_colors: np.ndarray,
    *,
    base_color: tuple[int, int, int] = MESH_BACKGROUND_GRAY,
) -> np.ndarray:
    """
    Color only mesh vertices that exactly match sampled pipeline points.

    Unsampled mesh vertices keep ``base_color``, giving a grey room with
    colored dots for segmentation-error visualization.
    """
    colors = np.full((len(mesh_vertices), 3), base_color, dtype=np.uint8)
    if len(source_points) == 0:
        return colors

    lookup = _point_color_lookup(source_points, source_colors)
    for i, key in enumerate(_vertex_coord_keys(mesh_vertices)):
        matched = lookup.get(tuple(key))
        if matched is not None:
            colors[i] = matched
    return colors


def load_mesh_arrays_from_point_colors(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    sparse_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the full semantic mesh, colored from input point-cloud data."""
    vertices, triangles, _semantic_colors = load_semantic_mesh_arrays()
    if sparse_only:
        mesh_colors = transfer_point_colors_to_mesh_vertices_sparse(vertices, points, colors)
    else:
        mesh_colors = transfer_point_colors_to_mesh_vertices(vertices, points, colors)
    return vertices, triangles, mesh_colors


def load_semantic_open3d_mesh():
    """Build an Open3D TriangleMesh for the Replica semantic mesh."""
    import open3d as o3d

    vertices, triangles, colors = load_semantic_mesh_arrays()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    mesh.compute_vertex_normals()
    return mesh


def load_open3d_mesh_from_point_colors(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    sparse_only: bool = False,
):
    """Build an Open3D mesh with colors transferred from a colored point cloud."""
    import open3d as o3d

    vertices, triangles, mesh_colors = load_mesh_arrays_from_point_colors(
        points,
        colors,
        sparse_only=sparse_only,
    )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors.astype(np.float64) / 255.0)
    mesh.compute_vertex_normals()
    return mesh


def _segmentation_error_tp_fp_mask(colors_u8: np.ndarray) -> np.ndarray:
    """True for true-positive (green) and false-positive (red) sampled points."""
    colors_u8 = np.asarray(colors_u8, dtype=np.uint8)
    tp = np.all(colors_u8 == np.asarray(COLOR_TP, dtype=np.uint8), axis=1)
    fp = np.all(colors_u8 == np.asarray(COLOR_FP, dtype=np.uint8), axis=1)
    return tp | fp


def _open3d_point_cloud(points: np.ndarray, colors_u8: np.ndarray):
    """Build an Open3D point cloud from numpy points and uint8 RGB."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors_u8, dtype=np.float64) / 255.0)
    return pcd


def build_segmentation_error_geometries(
    points: np.ndarray,
    colors_u8: np.ndarray,
) -> tuple[list, float | None]:
    """
    Grey mesh with sparse vertex colors plus enlarged TP/FP point markers.

    Yellow FN and gray unknown stay mesh-vertex sized; green/red render as
    ``SEGMENTATION_ERROR_TP_FP_POINT_SIZE`` point sprites on top.
    """
    mesh = load_open3d_mesh_from_point_colors(points, colors_u8, sparse_only=True)
    geometries: list = [mesh]
    marker_mask = _segmentation_error_tp_fp_mask(colors_u8)
    point_size = None
    if marker_mask.any():
        geometries.append(_open3d_point_cloud(points[marker_mask], colors_u8[marker_mask]))
        point_size = SEGMENTATION_ERROR_TP_FP_POINT_SIZE
    return geometries, point_size


def _show_iso_geometries(
    geometries: list,
    *,
    window_name: str,
    lookat: np.ndarray,
    point_size: float | None = None,
) -> None:
    """Interactive isometric view with optional enlarged point-cloud markers."""
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name=window_name, width=1280, height=720):
        raise RuntimeError("Could not create Open3D window")

    try:
        for geometry in geometries:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        if point_size is not None:
            opt.point_size = point_size
        ctr = vis.get_view_control()
        ctr.set_front(ISO_FRONT)
        ctr.set_up(ISO_UP)
        ctr.set_lookat(np.asarray(lookat, dtype=np.float64))
        ctr.set_zoom(ISO_ZOOM)
        vis.run()
    finally:
        vis.destroy_window()


def _subsample_for_png(
    points: np.ndarray,
    colors_u8: np.ndarray,
    max_points: int | None = 80_000,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Optionally subsample for rendering and return a suffix for the title."""
    n_orig = len(points)
    if max_points is not None and n_orig > max_points:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        idx = rng.choice(n_orig, max_points, replace=False)
        return points[idx], colors_u8[idx], f" ({max_points:,} / {n_orig:,} pts shown)"
    return points, colors_u8, ""


def _save_open3d_png(
    points: np.ndarray,
    colors_u8: np.ndarray,
    out_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    point_size: float = 3.0,
) -> bool:
    """Render with Open3D's visualizer, matching the interactive PLY viewer."""
    try:
        import open3d as o3d
    except ImportError:
        return False

    if len(points) == 0:
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_u8.astype(np.float64) / 255.0)

    vis = o3d.visualization.Visualizer()
    try:
        if not vis.create_window(
            window_name="PLY PNG export",
            width=width,
            height=height,
            visible=False,
        ):
            return False
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.asarray([0.0, 0.0, 0.0])

        ctr = vis.get_view_control()
        ctr.set_front(ISO_FRONT)
        ctr.set_up(ISO_UP)
        ctr.set_lookat(points.mean(axis=0).astype(np.float64))
        ctr.set_zoom(ISO_ZOOM)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out_path), do_render=True)
        return True
    except Exception as exc:
        print(f"  Open3D PNG render failed ({exc}); falling back to matplotlib.")
        return False
    finally:
        vis.destroy_window()


def _project_points_for_png(
    points: np.ndarray,
    *,
    elev_deg: float = ISO_FALLBACK_ELEV,
    azim_deg: float = ISO_FALLBACK_AZIM,
) -> np.ndarray:
    """Simple orthographic projection used only when Open3D screenshots fail."""
    p = points.astype(np.float64)
    p = p - p.mean(axis=0, keepdims=True)
    scale = np.max(np.ptp(p, axis=0))
    if scale > 0:
        p = p / scale

    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    # Rotate around z, then x, and keep image-plane x/y.
    rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(el), -np.sin(el)],
            [0.0, np.sin(el), np.cos(el)],
        ]
    )
    return (p @ rz.T @ rx.T)[:, :2]


def _project_points_for_png_with_depth(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projection plus depth for painter's-algorithm mesh fallback rendering."""
    p = points.astype(np.float64)
    p = p - p.mean(axis=0, keepdims=True)
    scale = np.max(np.ptp(p, axis=0))
    if scale > 0:
        p = p / scale

    az = np.deg2rad(ISO_FALLBACK_AZIM)
    el = np.deg2rad(ISO_FALLBACK_ELEV)
    rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(el), -np.sin(el)],
            [0.0, np.sin(el), np.cos(el)],
        ]
    )
    rotated = p @ rz.T @ rx.T
    return rotated[:, :2], rotated[:, 2]


def _save_open3d_mesh_png(
    vertices: np.ndarray,
    triangles: np.ndarray,
    colors_u8: np.ndarray,
    out_path: Path,
    *,
    overlay_points: np.ndarray | None = None,
    overlay_colors: np.ndarray | None = None,
    overlay_point_size: float | None = None,
) -> bool:
    """Render semantic mesh with Open3D when an OpenGL context is available."""
    try:
        import open3d as o3d
    except ImportError:
        return False

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_u8.astype(np.float64) / 255.0)
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    try:
        if not vis.create_window("Mesh PNG export", width=1280, height=720, visible=False):
            return False
        vis.add_geometry(mesh)
        if overlay_points is not None and overlay_colors is not None and len(overlay_points) > 0:
            vis.add_geometry(_open3d_point_cloud(overlay_points, overlay_colors))
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0])
        opt.mesh_show_back_face = True
        if overlay_point_size is not None:
            opt.point_size = overlay_point_size
        ctr = vis.get_view_control()
        ctr.set_front(ISO_FRONT)
        ctr.set_up(ISO_UP)
        ctr.set_lookat(vertices.mean(axis=0).astype(np.float64))
        ctr.set_zoom(ISO_ZOOM)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out_path), do_render=True)
        return True
    except Exception as exc:
        print(f"  Open3D mesh PNG render failed ({exc}); falling back to matplotlib.")
        return False
    finally:
        vis.destroy_window()


def save_semantic_mesh_png(out_path: Path, *, dpi: int = 150) -> None:
    """Save semantic GT mesh as a triangle rendering, not a point-cloud scatter."""
    vertices, triangles, colors = load_semantic_mesh_arrays()
    save_mesh_png(vertices, triangles, colors, out_path, dpi=dpi)


def save_mesh_png(
    vertices: np.ndarray,
    triangles: np.ndarray,
    colors: np.ndarray,
    out_path: Path,
    *,
    dpi: int = 150,
    max_triangles: int | None = 300_000,
    overlay_points: np.ndarray | None = None,
    overlay_colors: np.ndarray | None = None,
    overlay_point_size: float | None = None,
) -> None:
    """Save a colored triangle mesh PNG, using Open3D when available."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _save_open3d_mesh_png(
        vertices,
        triangles,
        colors,
        out_path,
        overlay_points=overlay_points,
        overlay_colors=overlay_colors,
        overlay_point_size=overlay_point_size,
    ):
        return

    if max_triangles is not None and len(triangles) > max_triangles:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        tri_idx = rng.choice(len(triangles), max_triangles, replace=False)
        triangles = triangles[tri_idx]

    xy, depth = _project_points_for_png_with_depth(vertices)
    tri_depth = depth[triangles].mean(axis=1)
    order = np.argsort(tri_depth)
    polys = xy[triangles][order]
    face_colors = (colors[triangles].mean(axis=1).astype(np.float32) / 255.0)[order]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="black")
    ax.set_facecolor("black")
    coll = PolyCollection(polys, facecolors=face_colors, edgecolors="none", closed=True)
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=dpi, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_ply_scatter_png(
    points: np.ndarray,
    colors_u8: np.ndarray,
    out_path: Path,
    title: str,
    *,
    dpi: int = 150,
    max_points: int | None = 80_000,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Render a colored point cloud to PNG; prefer Open3D viewer-style output."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(points) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        ax.set_title(f"{title} (empty)")
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return

    points, colors_u8, suffix = _subsample_for_png(points, colors_u8, max_points)
    if _save_open3d_png(points, colors_u8, out_path):
        return

    xy = _project_points_for_png(points)
    rgb = colors_u8.astype(np.float32) / 255.0
    n = len(points)
    point_size = max(0.2, min(4.0, 50_000 / max(n, 1)))

    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=rgb,
        s=point_size,
        alpha=1.0,
        marker=".",
        linewidths=0,
    )
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=dpi, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def export_ply_pngs(
    steps: list[int] | None = None,
    out_dir: Path | None = None,
    *,
    dpi: int = 150,
    max_points: int | None = 80_000,
) -> list[Path]:
    """
    Export all pipeline PLYs from ``collect_ply_paths`` to PNG scatter plots.

    Saves under ``outputs/{room}/viz/png/`` by default.
    """
    out_dir = out_dir or viz_png_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    items = collect_ply_paths(steps)
    mesh_items = collect_mesh_items(steps)
    saved: list[Path] = []

    if not items and not mesh_items:
        print("No PLY or mesh files found to export.")
        return saved

    print(f"Exporting {len(items)} PLY + {len(mesh_items)} mesh -> PNG to {out_dir}/")
    manifest: list[dict] = []

    for label, ply_path in items:
        if not ply_path.exists():
            print(f"  skip (missing): {ply_path.name}")
            continue
        points, colors = load_export_points_colors(label, ply_path)
        png_name = png_name_for_ply(ply_path, label)
        png_path = out_dir / png_name
        title = label
        if "segmentation errors" in label.lower():
            title += "\n(green=TP, red=FP, yellow=FN, gray=unknown)"
        save_ply_scatter_png(
            points,
            colors,
            png_path,
            title,
            dpi=dpi,
            max_points=max_points,
        )
        saved.append(png_path)
        manifest.append(
            {
                "label": label,
                "ply": str(ply_path),
                "png": str(png_path),
                "num_points": int(len(points)),
            }
        )
        print(f"  {png_name}  ({len(points):,} points)")

        mesh_png_path = out_dir / f"mesh_{png_name}"
        sparse_mesh = is_segmentation_errors_viz(label, ply_path)
        vertices, triangles, mesh_colors = load_mesh_arrays_from_point_colors(
            points,
            colors,
            sparse_only=sparse_mesh,
        )
        overlay_points = overlay_colors = None
        overlay_point_size = None
        if sparse_mesh:
            marker_mask = _segmentation_error_tp_fp_mask(colors)
            overlay_points = points[marker_mask]
            overlay_colors = colors[marker_mask]
            overlay_point_size = SEGMENTATION_ERROR_TP_FP_POINT_SIZE
        save_mesh_png(
            vertices,
            triangles,
            mesh_colors,
            mesh_png_path,
            dpi=dpi,
            overlay_points=overlay_points,
            overlay_colors=overlay_colors,
            overlay_point_size=overlay_point_size,
        )
        saved.append(mesh_png_path)
        manifest.append(
            {
                "label": f"Mesh {label}",
                "source_ply": str(ply_path),
                "png": str(mesh_png_path),
                "type": "mesh_from_point_colors",
                "num_vertices": int(len(vertices)),
                "num_triangles": int(len(triangles)),
            }
        )
        print(f"  mesh_{png_name}  ({len(vertices):,} vertices, {len(triangles):,} triangles)")

    for label, mesh_path in mesh_items:
        png_path = out_dir / "semantic_mesh.png"
        save_semantic_mesh_png(png_path, dpi=dpi)
        saved.append(png_path)
        manifest.append(
            {
                "label": label,
                "mesh": str(mesh_path),
                "png": str(png_path),
                "type": "triangle_mesh",
            }
        )
        print(f"  semantic_mesh.png  ({mesh_path})")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"exports": manifest, "steps": steps or list(range(1, 10))}, f, indent=2)
    print(f"Manifest: {manifest_path}")
    return saved


def errors_ply_path(method: str | None = None) -> Path:
    """Resolve segmentation error PLY from pipeline results (step 9)."""
    method = method or cfg.AGGREGATION_METHOD
    root = results_dir()
    candidates = (
        root / "errors.ply",
        root / f"errors_{method}.ply",
        root / f"segmentation_errors_{method}.ply",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def _colors_to_uint8(colors: np.ndarray) -> np.ndarray:
    """Open3D stores colors as float [0, 1]; convert to uint8 for bucket matching."""
    if colors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if colors.dtype == np.uint8:
        return colors
    scaled = np.asarray(colors, dtype=np.float64)
    if scaled.max() <= 1.0 + 1e-6:
        scaled *= 255.0
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def summarize_error_ply(colors_u8: np.ndarray) -> dict[str, int]:
    """Count TP / FP / FN / unknown buckets from vertex colors."""
    buckets = {
        "true_positive": COLOR_TP,
        "false_positive": COLOR_FP,
        "false_negative": COLOR_FN,
        "unknown": COLOR_UNKNOWN,
    }
    counts: dict[str, int] = {name: 0 for name in buckets}
    other = 0
    for row in colors_u8:
        matched = False
        for name, ref in buckets.items():
            if tuple(row) == ref:
                counts[name] += 1
                matched = True
                break
        if not matched:
            other += 1
    counts["other"] = other
    counts["total"] = len(colors_u8)
    return counts


def print_error_legend() -> None:
    print("  Color legend:")
    print(f"    green  {COLOR_TP}  true positive  (correct class)")
    print(f"    red    {COLOR_FP}  false positive (wrong class assigned)")
    print(f"    yellow {COLOR_FN}  false negative (GT class missed → predicted unknown)")
    print(f"    gray   {COLOR_UNKNOWN}  unknown / no prediction")


def print_error_summary(counts: dict[str, int], metrics_path: Path | None = None) -> None:
    total = counts["total"]
    if total == 0:
        print("  (empty point cloud)")
        return

    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    unk = counts["unknown"]
    other = counts["other"]
    correct_pct = 100.0 * tp / total

    print(f"  Points: {total:,}")
    print(f"    TP (green):  {tp:6,}  ({100.0 * tp / total:5.1f}%)")
    print(f"    FP (red):    {fp:6,}  ({100.0 * fp / total:5.1f}%)")
    print(f"    FN (yellow): {fn:6,}  ({100.0 * fn / total:5.1f}%)")
    print(f"    unknown:     {unk:6,}  ({100.0 * unk / total:5.1f}%)")
    if other:
        print(f"    other:       {other:6,}  ({100.0 * other / total:5.1f}%)")
    print(f"  Approx accuracy (TP / total): {correct_pct:.2f}%")

    if metrics_path and metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"  Saved metrics: mIoU={metrics.get('miou', 0):.4f}, accuracy={metrics.get('accuracy', 0):.4f}")


def show_ply(path: Path, title: str | None = None) -> bool:
    """
    Open an interactive Open3D window for one PLY file.

    Returns True if the viewer opened, False on headless / missing display.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — pip install open3d")
        return False

    path = Path(path)
    if not path.exists():
        print(f"  skip (missing): {path}")
        return False

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        print(f"  skip (empty): {path}")
        return False

    window_name = title or path.name
    print(f"  Opening: {window_name} ({len(pcd.points)} points) — close window for next")
    try:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
            width=1280,
            height=720,
            lookat=np.asarray(pcd.points).mean(axis=0),
            front=ISO_FRONT,
            up=ISO_UP,
            zoom=ISO_ZOOM,
        )
    except Exception as exc:
        print(f"  Could not open viewer ({exc}). Open manually in MeshLab: {path}")
        return False
    return True


def show_ply_as_mesh(path: Path, title: str | None = None) -> bool:
    """Open the full semantic mesh colored by a PLY point-cloud visualization."""
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — pip install open3d")
        return False

    path = Path(path)
    if not path.exists():
        print(f"  skip mesh (missing source PLY): {path}")
        return False

    label = title or path.stem
    points, colors = load_export_points_colors(label, path)
    sparse_mesh = is_segmentation_errors_viz(label, path)
    if sparse_mesh:
        geometries, point_size = build_segmentation_error_geometries(points, colors)
        vertices = np.asarray(geometries[0].vertices)
        window_name = f"Mesh {label}"
        print(f"  Opening: {window_name} ({len(vertices):,} vertices) — close window for next")
        try:
            _show_iso_geometries(
                geometries,
                window_name=window_name,
                lookat=vertices.mean(axis=0),
                point_size=point_size,
            )
        except Exception as exc:
            print(f"  Could not open mesh viewer ({exc}). Source PLY: {path}")
            return False
        return True

    mesh = load_open3d_mesh_from_point_colors(points, colors, sparse_only=False)
    vertices = np.asarray(mesh.vertices)
    window_name = f"Mesh {label}"
    print(f"  Opening: {window_name} ({len(vertices):,} vertices) — close window for next")
    try:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=window_name,
            width=1280,
            height=720,
            lookat=vertices.mean(axis=0),
            front=ISO_FRONT,
            up=ISO_UP,
            zoom=ISO_ZOOM,
        )
    except Exception as exc:
        print(f"  Could not open mesh viewer ({exc}). Source PLY: {path}")
        return False
    return True


def show_segmentation_errors_ply(
    path: Path | None = None,
    method: str | None = None,
) -> bool:
    """
    Open the step-9 segmentation error PLY with a printed color legend and counts.

    Default path: ``outputs/room2/results/segmentation_errors_{method}.ply``
    (also accepts ``errors.ply`` or ``errors_{method}.ply`` in the same folder).
    """
    method = method or cfg.AGGREGATION_METHOD
    path = Path(path) if path is not None else errors_ply_path(method)
    metrics_path = results_dir() / f"metrics_{method}.json"

    print(f"\nSegmentation errors — method={method}")
    print(f"  PLY: {path}")
    if not path.exists():
        print("  File not found. Run pipeline step 9 first (segmentation_viz).")
        return False

    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — pip install open3d")
        return False

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        print(f"  skip (empty): {path}")
        return False

    colors_u8 = _colors_to_uint8(np.asarray(pcd.colors))
    counts = summarize_error_ply(colors_u8)
    print_error_legend()
    print_error_summary(counts, metrics_path)

    points = np.asarray(pcd.points, dtype=np.float32)
    geometries, point_size = build_segmentation_error_geometries(points, colors_u8)
    vertices = np.asarray(geometries[0].vertices)
    window_name = f"Segmentation errors ({method}) — close window to exit"
    print(f"\n  Opening mesh viewer ({len(vertices):,} vertices, {len(points):,} sampled)...")
    try:
        _show_iso_geometries(
            geometries,
            window_name=window_name,
            lookat=vertices.mean(axis=0),
            point_size=point_size,
        )
    except Exception as exc:
        print(f"  Could not open viewer ({exc}). Open manually in MeshLab: {path}")
        return False
    return True


def show_ply_list(ply_items: list[tuple[str, Path]]) -> None:
    """Show each PLY in sequence; user closes each window to advance."""
    if not ply_items:
        print("No PLY files to visualize.")
        return

    print(f"\nPLY viewer — {len(ply_items)} file(s). Close each window to continue.")
    for label, path in ply_items:
        show_ply(path, title=label)
        show_ply_as_mesh(path, title=label)


def show_ply_mesh_list(ply_items: list[tuple[str, Path]]) -> None:
    """Show each PLY visualization transferred onto the full mesh surface."""
    if not ply_items:
        print("No PLY files to visualize as mesh.")
        return

    print(f"\nMesh viewer — {len(ply_items)} file(s). Close each window to continue.")
    for label, path in ply_items:
        show_ply_as_mesh(path, title=label)


def show_semantic_mesh(title: str = "Semantic mesh (GT labels)") -> bool:
    """Open an interactive Open3D window for the Replica semantic mesh."""
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — pip install open3d")
        return False

    mesh_path = Path(cfg.MESH_PATH)
    if not mesh_path.exists():
        print(f"  skip (missing): {mesh_path}")
        return False

    mesh = load_semantic_open3d_mesh()
    vertices = np.asarray(mesh.vertices)
    print(f"  Opening: {title} ({len(vertices):,} vertices, {len(mesh.triangles):,} triangles)")
    try:
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=title,
            width=1280,
            height=720,
            lookat=vertices.mean(axis=0),
            front=ISO_FRONT,
            up=ISO_UP,
            zoom=ISO_ZOOM,
        )
    except Exception as exc:
        print(f"  Could not open mesh viewer ({exc}). Open manually in MeshLab: {mesh_path}")
        return False
    return True


def show_dev_ply_outputs(steps: list[int] | None = None) -> None:
    """Show PLY outputs plus the source semantic mesh relevant to a dev run."""
    items = collect_ply_paths(steps)
    show_ply_list(items)
    for label, _path in collect_mesh_items(steps):
        show_semantic_mesh(title=label)


def show_dev_mesh_outputs(steps: list[int] | None = None) -> None:
    """Show PLY outputs as full mesh surfaces; no point-cloud windows, no PNG saves."""
    show_ply_mesh_list(collect_ply_paths(steps))
    for label, _path in collect_mesh_items(steps):
        show_semantic_mesh(title=label)


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        show_dev_ply_outputs()
        return

    if argv[0] in ("export", "export-png", "--export", "--export-png"):
        step_args = [int(x) for x in argv[1:] if x.isdigit()]
        export_ply_pngs(steps=step_args if step_args else None)
        return

    if argv[0] in ("errors", "--errors", "-e"):
        method = argv[1] if len(argv) > 1 else None
        show_segmentation_errors_ply(method=method)
        return

    if argv[0] in ("mesh", "--mesh"):
        rest = argv[1:]
        step_args = [int(x) for x in rest if x.isdigit()]
        ply_args = [Path(x) for x in rest if Path(x).suffix == ".ply"]
        if ply_args:
            items = [(p.name, p) for p in ply_args]
            show_ply_mesh_list(items)
        else:
            show_dev_mesh_outputs(steps=step_args if step_args else None)
        return

    if argv[0] == "semantic-mesh":
        show_semantic_mesh()
        return

    if len(argv) == 1 and Path(argv[0]).suffix == ".ply":
        path = Path(argv[0])
        if "error" in path.stem.lower():
            show_segmentation_errors_ply(path=path)
            return

    items = [(Path(p).name, Path(p)) for p in argv]
    show_ply_list(items)


if __name__ == "__main__":
    main()
