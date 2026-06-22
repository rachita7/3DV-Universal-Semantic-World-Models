# --- paths ---
PROJECT_ROOT = "."  # relative to repo root
REPLICA_ROOT = "dataset/replica"
ROOM = "room2"
DINOV3_REPO = "dinov3"
PRETRAINED_ROOT = "pretrainedmodels"
OUTPUT_ROOT = "outputs"

MESH_PATH = f"{REPLICA_ROOT}/mesh_semantic.ply"
INFO_SEMANTIC_PATH = f"{REPLICA_ROOT}/info_semantic.json"
CAM_PARAMS_PATH = f"{REPLICA_ROOT}/cam_params.json"
TRAJ_PATH = f"{REPLICA_ROOT}/{ROOM}/traj.txt"
FRAMES_DIR = f"{REPLICA_ROOT}/{ROOM}/results"

# Checkpoints
VITS16_WEIGHTS = f"{PRETRAINED_ROOT}/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
VITL16_WEIGHTS = f"{PRETRAINED_ROOT}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINOTXT_WEIGHTS = (
    f"{PRETRAINED_ROOT}/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
)

# --- pipeline control ---
STEPS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
FORCE_RECOMPUTE = False
RANDOM_SEED = 42

# --- sampling ---
NUM_POINTS_INITIAL = 100_000
NUM_POINTS_SUBSAMPLE = 50_000

# --- frames (None = all frames in traj) ---
NUM_FRAMES = None  # set e.g. 10 for dev

# --- image geometry ---
IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 680
PAD_HEIGHT = 688  # padded to multiple of 16
PATCH_SIZE = 16
PATCH_GRID_W = IMAGE_WIDTH // PATCH_SIZE  # 75
PATCH_GRID_H = PAD_HEIGHT // PATCH_SIZE  # 43

# --- camera / visibility ---
DEPTH_TOLERANCE = 0.05  # meters
DEPTH_SCALE = 6553.5  # from cam_params.json (overwritten at load if different)

# --- model inference ---
DEVICE = "cuda"  # "cuda" or "cpu"
BATCH_SIZE = 4
VITS16_FEATURE_DIM = 384
DINOTXT_FEATURE_DIM = 1024  # aligned patch token dim (second half of 2048-d joint embedding)

# --- aggregation ---
AGGREGATION_METHOD = "sam_slerp"  # "mean" | "slerp" | "frechet" | "weighted_slerp"
FRECHET_MAX_ITER = 50
FRECHET_TOL = 1e-6
COMPUTE_PAIRWISE = False  # slow; dispersion-only is enough for subsampling
# Per-point methods (slerp / frechet / weighted_slerp) gather observations in chunks.
# Higher = fewer disk passes but more RAM (~obs × 1024 × 4 bytes per chunk).
# 24 GB RAM: 1_500_000 (~6 GB buffer, ~14 chunks). 32 GB+: try 2_000_000.
AGGREGATION_MAX_OBS_PER_CHUNK = 1_500_000
AGGREGATION_FRAME_MMAP = True  # mmap .npy frame files during gather (lower RAM, better OS cache)

# --- segmentation ---
COSINE_THRESHOLD = 0.0  # 0 = pure argmax; raise to reject low-confidence points

# Indoor prompt templates — {} is filled with each class name below.
TEMPLATES_INDOOR = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a close-up photo of a {}.",
    "a good photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a blurry photo of a {}.",
    "a rendering of a {}.",
    "a 3D render of a {}.",
    "a synthetic image of a {}.",
    "a photorealistic render of a {}.",
    "a {} inside a room.",
    "a {} in an indoor scene.",
    "a {} in a living room.",
    "a {} in a furnished room.",
    "a {} photographed indoors.",
    "a {} under indoor lighting.",
    "a {} against a wall.",
    "a {} near a wall.",
    "a {} on the floor.",
    "a {} in a well-lit room.",
    "an interior design photo of a {}.",
    "a photo of the wooden {}.",
    "a photo of the textured {}.",
    "a frontal view of the {}.",
    "a side view of the {}.",
    "a top-down view of the {}.",
    "the {} seen from above.",
    "a low resolution photo of the {}.",
]

# Class names must match info_semantic.json
SEMANTIC_CLASSES = [
    "chair",
    "table",
    "sofa",
    "bed",
    "toilet",
    "sink",
    "bathtub",
    "tv",
    "lamp",
    "rug",
    "floor",
    "wall",
    "ceiling",
    "door",
    "window",
    "cabinet",
    "shelf",
    "desk",
    "monitor",
    "book",
    "pillow",
    "towel",
    "plant",
    "bin",
]

TEXT_PROMPTS = {
    cls: [template.format(cls) for template in TEMPLATES_INDOOR]
    for cls in SEMANTIC_CLASSES
}

# Classes to evaluate (None = all classes found in GT that have prompts)
EVAL_CLASSES = None

# --- SAM-based aggregation (method "sam_slerp") ---
# Requires: pip install segment-anything  (checkpoint already in pretrainedmodels/).
# Step: SAM segments each frame -> per-mask DINOtxt embedding (mean of patch tokens
# in the mask) -> per-point area-weighted SLERP over the mask embeddings of its views.
SAM_CHECKPOINT = f"{PRETRAINED_ROOT}/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"  # "vit_h" | "vit_l" | "vit_b"
SAM_POINTS_PER_SIDE = 32  # grid density; lower (e.g. 16) = faster + less VRAM, coarser masks
SAM_POINTS_PER_BATCH = 32  # mask-decoder batch; lower (16) if 6 GB VRAM OOMs (lib default 64)
SAM_PRED_IOU_THRESH = 0.86
SAM_STABILITY_SCORE_THRESH = 0.92
SAM_MIN_MASK_REGION_AREA = 100
SAM_MIN_MASK_AREA_FRAC = 0.002  # drop masks smaller than this fraction of the image
SAM_NMS_THRESHOLD = 0.8  # suppress near-duplicate masks by overlap
SAM_USE_AMP = True  # fp16 autocast during SAM inference (needed to fit ViT-H in 6 GB VRAM)
SAM_OVERLAP_SLERP = False  # Pe3R-style intra-frame overlap adjustment between masks

# --- ablation (phase 4) ---
ABLATION_METHODS = ["mean", "weighted_slerp", "frechet", "sam_slerp"]
