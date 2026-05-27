"""
utils.py
Shared helper functions used across all three pipeline stages.

Functions:
    extract_foreground_mask()  — BG deviation threshold → binary mask
    apply_dilation()           — morphological dilation on binary mask
    normalize_canvas()         — crop → pad → resize to standard canvas
    compute_iou()              — intersection over union on two binary masks
    save_diff_map()            — generate color-coded difference image
    ensure_dirs()              — create output directories if missing
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from config import (
    BG_DEVIATION_THRESHOLD,
    LABEL_BAR_EXCLUSION,
    DILATION_FOOTPRINT,
    CANVAS_SIZE,
    CANVAS_PADDING,
    DIFF_COLOR_MATCH,
    DIFF_COLOR_MISSING,
    DIFF_COLOR_EXTRA,
    STL_VIEWS_DIR,
    TDP_VIEWS_DIR,
    DIFF_VIEWS_DIR,
)


# ── Directory management ──────────────────────────────────────────────────────

def ensure_dirs(*dirs):
    """Create output directories if they don't already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ── Mask extraction ───────────────────────────────────────────────────────────

def extract_foreground_mask(image_path: str, is_dark_background: bool) -> np.ndarray:
    """
    Convert an RGB image to a binary foreground mask using background
    deviation thresholding.

    mask(x,y) = 1  if  max_c |I(x,y,c) - BG(c)| > threshold  else 0

    Args:
        image_path:          Path to the PNG image
        is_dark_background:  True for Stage 1 (dark bg), False for Stage 2 (white bg)

    Returns:
        Binary uint8 array (0 or 255), shape (H, W)
    """
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)

    # Determine background colour based on which stage produced the image
    if is_dark_background:
        bg = np.array([30, 32, 37], dtype=np.float32)    # #1E2025
    else:
        bg = np.array([255, 255, 255], dtype=np.float32) # #FFFFFF

    # Per-pixel maximum channel deviation from background
    deviation = np.max(np.abs(img - bg), axis=2)

    # Crop bottom LABEL_BAR_EXCLUSION% to remove view-name label bars
    h = deviation.shape[0]
    crop_rows = int(h * (1 - LABEL_BAR_EXCLUSION))
    deviation = deviation[:crop_rows, :]

    # Threshold: pixels deviating more than BG_DEVIATION_THRESHOLD are foreground
    mask = (deviation > BG_DEVIATION_THRESHOLD).astype(np.uint8) * 255
    return mask


# ── Morphological dilation ────────────────────────────────────────────────────

def apply_dilation(mask: np.ndarray) -> np.ndarray:
    """
    Apply morphological dilation to a binary mask using a rectangular
    structuring element of size DILATION_FOOTPRINT.

    Purpose: absorbs ~4px cross-renderer edge-width differences without
    masking genuine geometric defects.

    Args:
        mask: Binary uint8 array (0 or 255)

    Returns:
        Dilated binary uint8 array (0 or 255)
    """
    binary = mask > 0
    struct = np.ones(DILATION_FOOTPRINT, dtype=bool)
    dilated = binary_dilation(binary, structure=struct)
    return dilated.astype(np.uint8) * 255


# ── Canvas normalisation ──────────────────────────────────────────────────────

def normalize_canvas(mask: np.ndarray) -> np.ndarray:
    """
    Normalize a binary mask to a standard canvas for metric comparison:
    1. Crop to the silhouette's tight bounding box
    2. Add uniform CANVAS_PADDING pixel border
    3. Resize to CANVAS_SIZE using nearest-neighbor interpolation

    Why nearest-neighbor?
        Bilinear/bicubic introduce grey halo pixels at boundaries,
        corrupting SSIM and IoU which assume strictly binary inputs.

    Why bounding-box crop?
        Without it, an Isometric view occupying a larger diagonal canvas
        area would have a larger raw foreground area than a Top view,
        making raw IoU comparison meaningless.

    Args:
        mask: Binary uint8 array (0 or 255)

    Returns:
        Normalized binary uint8 array of shape CANVAS_SIZE (H, W)
    """
    binary = mask > 0

    # Find bounding box of foreground pixels
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)

    if not rows.any():
        # Empty mask — return blank canvas
        return np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0]), dtype=np.uint8)

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    cropped = mask[r_min:r_max+1, c_min:c_max+1]

    # Add padding
    pad = CANVAS_PADDING
    padded = np.pad(cropped, pad, mode='constant', constant_values=0)

    # Resize to standard canvas using nearest-neighbor (PIL NEAREST)
    img = Image.fromarray(padded)
    resized = img.resize(CANVAS_SIZE, Image.NEAREST)

    return np.array(resized)


# ── IoU computation ───────────────────────────────────────────────────────────

def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.

    IoU = |A ∩ B| / |A ∪ B|

    IoU = 1.0: perfect overlap
    IoU = 0.0: no overlap
    Drops linearly with defect size — directly proportional to geometry loss/gain.

    Args:
        mask_a: Binary array (STL mask)
        mask_b: Binary array (TDP mask)

    Returns:
        Float in [0.0, 1.0]
    """
    a = mask_a > 0
    b = mask_b > 0

    intersection = np.logical_and(a, b).sum()
    union        = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


# ── Difference map generation ─────────────────────────────────────────────────

def save_diff_map(mask_stl: np.ndarray, mask_tdp: np.ndarray,
                  output_path: str) -> None:
    """
    Generate and save a color-coded difference map.

    Color scheme:
        Green  (#00C832) — geometry present in both masks (correct overlap)
        Red    (#DC1E1E) — present in STL, missing from TDP (dropped geometry)
        Yellow (#FFC800) — present in TDP, not in STL (spurious geometry added)
        Black            — background in both

    Args:
        mask_stl:    Normalized binary mask from Stage 1
        mask_tdp:    Normalized binary mask from Stage 2
        output_path: Where to save the PNG
    """
    a = mask_stl > 0
    b = mask_tdp > 0

    h, w = a.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)

    # Green: both present
    both = np.logical_and(a, b)
    diff[both] = DIFF_COLOR_MATCH

    # Red: in STL only (geometry missing from TDP export)
    stl_only = np.logical_and(a, ~b)
    diff[stl_only] = DIFF_COLOR_MISSING

    # Yellow: in TDP only (spurious geometry in TDP export)
    tdp_only = np.logical_and(~a, b)
    diff[tdp_only] = DIFF_COLOR_EXTRA

    Image.fromarray(diff).save(output_path)


# ── Image loading helper ──────────────────────────────────────────────────────

def load_image_as_array(path: str) -> np.ndarray:
    """Load a PNG as a numpy RGB array."""
    return np.array(Image.open(path).convert("RGB"))


# ── View name matching ────────────────────────────────────────────────────────

def match_view_pairs(stl_dir: str, tdp_dir: str) -> list[tuple[str, str, str]]:
    """
    Match Stage 1 and Stage 2 output images by view name.

    Returns:
        List of (view_name, stl_path, tdp_path) tuples
    """
    stl_files = {os.path.splitext(f)[0]: os.path.join(stl_dir, f)
                 for f in os.listdir(stl_dir) if f.endswith(".png")}
    tdp_files = {os.path.splitext(f)[0]: os.path.join(tdp_dir, f)
                 for f in os.listdir(tdp_dir) if f.endswith(".png")}

    common = sorted(set(stl_files.keys()) & set(tdp_files.keys()))

    if not common:
        raise ValueError(
            f"No matching view names found.\n"
            f"STL views: {list(stl_files.keys())}\n"
            f"TDP views: {list(tdp_files.keys())}"
        )

    pairs = [(name, stl_files[name], tdp_files[name]) for name in common]
    print(f"[Utils] Matched {len(pairs)} view pairs: {[p[0] for p in pairs]}")
    return pairs
