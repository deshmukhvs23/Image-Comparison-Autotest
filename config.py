"""
config.py
All tunable parameters for the Image Comparison Autotest pipeline.
Change values here — no need to touch stage scripts.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
STL_VIEWS_DIR = os.path.join(BASE_DIR, "stl_views")
TDP_VIEWS_DIR = os.path.join(BASE_DIR, "tdp_views")
DIFF_VIEWS_DIR= os.path.join(BASE_DIR, "diff_views")
SAMPLE_STL    = os.path.join(BASE_DIR, "sample_data", "simple_block.stl")

# ── Stage 1 & 2: Rendering parameters ────────────────────────────────────────
RENDER_DPI         = 150          # higher = more detail, slower
FIGURE_SIZE        = (8, 6)       # inches — affects canvas pixel count

BACKGROUND_DARK    = "#1E2025"    # Stage 1: dark bg matches NX viewport
BACKGROUND_WHITE   = "#FFFFFF"    # Stage 2: white bg matches Adobe Acrobat
FACE_COLOR         = "#C8C8C8"    # neutral grey — no saturated colours
EDGE_COLOR         = "#505050"    # dark edges for clean silhouette extraction

# Lambertian shading: shade_i = AMBIENT + DIFFUSE * max(n_i · l, 0)
# AMBIENT=0.30 prevents pure-black faces that would punch holes in the mask
# DIFFUSE=0.70 provides depth cues without specular highlight instability
AMBIENT            = 0.30
DIFFUSE            = 0.70

# ── NX standard views — (elevation°, azimuth°) ───────────────────────────────
# These match exactly the system-defined view set in Siemens NX.
# Changing these angles would break view-to-view alignment with TDP exports.
NX_VIEWS = {
    "Top"       : ( 90.00,  -90.00),
    "Front"     : (  0.00,  -90.00),
    "Right"     : (  0.00,    0.00),
    "Back"      : (  0.00,   90.00),
    "Bottom"    : (-90.00,  -90.00),
    "Left"      : (  0.00,  180.00),
    "Isometric" : ( 35.26,  -45.00),
    "Trimetric" : ( 20.00,  -35.00),
}

# ── Stage 3: Mask extraction ──────────────────────────────────────────────────
# Pixel is classified as foreground if any RGB channel deviates from
# background by more than this value (0-255 scale).
# < 20: catches compression speckle (false positives)
# > 60: erodes thin edge-on faces (false negatives)
# 40: empirically tuned sweet spot
BG_DEVIATION_THRESHOLD = 40

# Crop bottom N% of image to remove view-name label bars
# These labels are useful for human inspection but corrupt IoU calculation
LABEL_BAR_EXCLUSION    = 0.10     # 10%

# ── Stage 3: Morphological dilation ──────────────────────────────────────────
# Expands silhouette outward ~4px to absorb cross-renderer edge-width noise.
# < (5,5): fails to absorb full cross-engine edge-width difference
# > (13,13): bridges genuinely separate features, masks real defects
# (9,9): empirically tuned to sit between both failure modes
DILATION_FOOTPRINT     = (9, 9)

# ── Stage 3: Canvas normalisation ────────────────────────────────────────────
# Both masks are cropped to silhouette bounding box, padded, then resized
# to this common canvas so SSIM and IoU compare at identical scale.
CANVAS_SIZE            = (800, 600)
CANVAS_PADDING         = 30       # pixels of padding around silhouette bounding box

# Nearest-neighbor interpolation preserves strictly binary mask values.
# Bilinear/bicubic would introduce grey halo pixels at boundaries,
# corrupting SSIM and IoU computations.
RESIZE_INTERPOLATION   = "nearest"

# ── Stage 3: Pass/Fail thresholds ────────────────────────────────────────────
# Both must be met simultaneously (AND rule).
# High score on one cannot compensate for failure on the other —
# SSIM and IoU detect different defect classes.
SSIM_THRESHOLD         = 0.78
IOU_THRESHOLD          = 0.72

# ── Stage 3: Difference map colours (RGB) ────────────────────────────────────
DIFF_COLOR_MATCH       = (0,   200,  50)   # green  — geometry in both masks
DIFF_COLOR_MISSING     = (220,  30,  30)   # red    — in STL, missing from TDP
DIFF_COLOR_EXTRA       = (255, 200,   0)   # yellow — in TDP, not in STL
