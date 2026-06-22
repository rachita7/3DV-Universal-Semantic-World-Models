# 3DV-Universal-Semantic-World-Models

DINOv3 produces strong per-image patch features, but the **same 3D surface point looks different from different camera views**, i.e., its patch feature "flickers" across frames. This project quantifies that cross-view inconsistency on the [Replica](https://github.com/facebookresearch/Replica-Dataset) indoor scenes, then reduces it by aggregating each point's multi-view observations into a single, view-stable descriptor.

Using DINOtxt-aligned (text-aligned) features, it then performs **open-vocabulary 3D semantic segmentation**: each aggregated per-point feature is matched against text-prompt class embeddings, and the labeling is scored against Replica's ground truth (IoU).

End to end: sample points on the scene mesh → resolve per-view visibility → extract DINOv3 patch features per view → measure cross-view cosine dispersion → aggregate features per point (mean / weighted SLERP / SAM-guided SLERP) → segment via text prompts → evaluate.

## Setup

```bash
git submodule update --init --recursive
conda create -n 3dvision python=3.10 && conda activate 3dvision
# or: python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision   # install the build for your CUDA version first
pip install -e .                # project + deps; makes config/src importable
pip install -e dinov3/          # uses the torch you just installed
```

Or install deps only: `pip install -r requirements.txt`

## Models

Download these checkpoints into `pretrainedmodels/` (git-ignored). Paths are set in [`config/settings.py`](config/settings.py).

DINOv3 weights: request access and download from [Meta's DINOv3 downloads page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/):

| File | Setting | Description |
|------|---------|-------------|
| `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | `VITS16_WEIGHTS` | ViT-S/16 backbone (small) — variance analysis |
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | `VITL16_WEIGHTS` | ViT-L/16 backbone (large) |
| `dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth` | `DINOTXT_WEIGHTS` | DINOtxt vision head + text encoder |

SAM weights: download from [HuggingFace](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth):

| File | Setting | Description |
|------|---------|-------------|
| `sam_vit_h_4b8939.pth` | `SAM_CHECKPOINT` | SAM ViT-H — masks for `sam_slerp` aggregation |

Expected layout:

```
pretrainedmodels/
├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
├── dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth
└── sam_vit_h_4b8939.pth
```


## Dataset

All data lives under `dataset/` (git-ignored). The pipeline supports two geometry sources.

### Replica (default)

Posed RGB-D frames plus a semantically-labelled mesh — used for feature extraction **and** ground-truth evaluation.

1. Create the data folder at the repo root: `mkdir dataset`
2. Download the prepared bundle: **[Replica (Polybox)](https://polybox.ethz.ch/index.php/s/TZmTjBJ6dCbwG29)**
3. Unpack it into `dataset/replica/` so it matches the layout below:

```
dataset/replica/
├── mesh_semantic.ply       # semantic mesh           (MESH_PATH)
├── info_semantic.json      # class id ↔ name map      (INFO_SEMANTIC_PATH)
├── cam_params.json         # intrinsics + depth scale (CAM_PARAMS_PATH)
└── room2/                  # scene selected by ROOM
    ├── traj.txt            # per-frame camera poses   (TRAJ_PATH)
    └── results/            # RGB + depth frames        (FRAMES_DIR)
        ├── frame000000.jpg
        ├── depth000000.png
        └── ...
```


### VGGT / COLMAP (geometry only, optional)

Build per-view visibility from a VGGT / COLMAP reconstruction instead of Replica depth. This is **geometry only — no GT labels**, so the GT-dependent steps (1, 4, 5, 9) don't apply; run it only for the feature/aggregation steps (2, 3, 6, 7).

1. Create the data folder at the repo root: `mkdir VGGT Output`
2. Put the COLMAP-style binaries at the paths configured in `config/settings.py`. Defaults (relative to the repo root):

```
VGGT Output/
├── images.bin      # VGGT_IMAGES_BIN
└── points3D.bin    # VGGT_POINTS3D_BIN
```

3. In `config/settings.py` set `VISIBILITY_SOURCE = "vggt"` (and edit `VGGT_IMAGES_BIN` / `VGGT_POINTS3D_BIN` if your files live elsewhere).
4. Run step 2 to build visibility, then the feature steps — set `STEPS = [2, 3, 6, 7]` and run:

```bash
python main.py
```

Step 2 writes `outputs/<ROOM>/visibility.npz` (same schema as Replica) and `outputs/<ROOM>/vggt_points.npy`. 

## Run

Edit [`config/settings.py`](config/settings.py), then run the end-to-end pipeline:

```bash
python main.py
```

## Pipeline steps

`STEPS` selects which of these run (default: 1-9).

| Step | Description |
|------|-------------|
| 1 | Sample 100k mesh points + GT labels |
| 2 | Project to views, occlusion cull, save visibility |
| 3 | Extract ViT-S/16 patch features (variance analysis) |
| 4 | Compute cosine dispersion + heatmap |
| 5 | Variance-weighted 50k subsample |
| 6 | Extract DINOtxt-aligned patch features (ViT-L) |
| 7 | Aggregate multi-view features (`AGGREGATION_METHOD`) |
| 8 | Text prompt ensemble embeddings |
| 9 | Cosine segmentation, IoU metrics, error visualization |
| 10 | PCA comparison video (dense vs. aggregated) |

## Cases

All configured in [`config/settings.py`](config/settings.py), then run `python main.py`.

- **Visibility source (step 2)** — `VISIBILITY_SOURCE ∈ {replica, vggt}` (see [Dataset](#dataset)). `replica` projects sampled mesh points with depth; `vggt` reads COLMAP-style `VGGT_IMAGES_BIN` / `VGGT_POINTS3D_BIN` (geometry only, no GT labels).

- **Aggregation method(s)** — `ABLATION_METHODS` is the single knob (each entry ∈ `{mean, weighted_slerp, frechet, sam_slerp}`); `AGGREGATION_METHOD` is derived from its first entry, so you only ever edit this one list:
  - **One method:** `ABLATION_METHODS = ["sam_slerp"]` → only that method is aggregated, segmented, and scored (steps 7 & 9).
  - **Full ablation (default):** `ABLATION_METHODS = ["mean", "weighted_slerp", "sam_slerp"]` → with step 9 in `STEPS`, every listed method is run and compared. The first entry is the "primary" used for viz defaults.

- **PCA video (step 10)** — add `10` to `STEPS`. `PCA_VIDEO_METHODS` takes 1 or 2 method names (2 → side-by-side 2×2 sharing a PCA basis; `None` → `[AGGREGATION_METHOD]`); `PCA_VIDEO_DENSE_SOURCE ∈ {dinotxt, vits16}` picks the dense panel. Each method needs its step-7 output.

## Outputs

Results saved autmatically under `outputs/room2/` (the PCA video is saved in `outputs/room2/pca_video/`)
