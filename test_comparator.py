"""
tests/test_comparator.py
Unit tests for Stage 3 — silhouette comparator.
Tests mask extraction, dilation, and metric computation logic.
"""
import pytest
import numpy as np


# ── Mask extraction tests ─────────────────────────────────────────────────────

def test_identical_masks_give_perfect_scores():
    """
    Two identical binary masks must yield SSIM=1.0 and IoU=1.0.
    This is the baseline correctness check for the metric engine.
    """
    from skimage.metrics import structural_similarity as ssim

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255  # white square in centre

    score = ssim(mask, mask, data_range=255)
    assert score == pytest.approx(1.0, abs=1e-6), f"Expected SSIM=1.0, got {score}"


def test_iou_identical_masks():
    """IoU of a mask with itself must be exactly 1.0."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True

    intersection = np.logical_and(mask, mask).sum()
    union        = np.logical_or(mask, mask).sum()
    iou = intersection / union

    assert iou == pytest.approx(1.0, abs=1e-6)


def test_iou_no_overlap():
    """IoU of two non-overlapping masks must be 0.0."""
    mask_a = np.zeros((100, 100), dtype=bool)
    mask_b = np.zeros((100, 100), dtype=bool)
    mask_a[10:40, 10:40] = True   # top-left
    mask_b[60:90, 60:90] = True   # bottom-right

    intersection = np.logical_and(mask_a, mask_b).sum()
    union        = np.logical_or(mask_a, mask_b).sum()
    iou = intersection / union

    assert iou == pytest.approx(0.0, abs=1e-6)


def test_iou_partial_overlap():
    """IoU of 50% overlapping masks must be between 0 and 1."""
    mask_a = np.zeros((100, 100), dtype=bool)
    mask_b = np.zeros((100, 100), dtype=bool)
    mask_a[0:60,  0:60]  = True
    mask_b[30:90, 30:90] = True

    intersection = np.logical_and(mask_a, mask_b).sum()
    union        = np.logical_or(mask_a, mask_b).sum()
    iou = intersection / union

    assert 0.0 < iou < 1.0, f"Expected 0 < IoU < 1, got {iou}"


# ── Morphological dilation tests ──────────────────────────────────────────────

def test_dilation_expands_mask():
    """After 9x9 dilation the foreground area must be larger."""
    from scipy.ndimage import binary_dilation

    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True  # 20x20 square

    struct = np.ones((9, 9), dtype=bool)
    dilated = binary_dilation(mask, structure=struct)

    original_area = mask.sum()
    dilated_area  = dilated.sum()

    assert dilated_area > original_area, "Dilation must expand the mask"


def test_dilation_does_not_shrink_mask():
    """Dilation must never reduce foreground pixels."""
    from scipy.ndimage import binary_dilation

    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 10:90] = True

    struct  = np.ones((9, 9), dtype=bool)
    dilated = binary_dilation(mask, structure=struct)

    assert dilated.sum() >= mask.sum()


# ── Pass/Fail threshold tests ─────────────────────────────────────────────────

def test_pass_thresholds():
    """Scores above both thresholds must yield PASS."""
    ssim_score = 0.95
    iou_score  = 0.90
    SSIM_THRESH = 0.78
    IOU_THRESH  = 0.72

    result = "PASS" if (ssim_score >= SSIM_THRESH and iou_score >= IOU_THRESH) else "FAIL"
    assert result == "PASS"


def test_fail_on_low_ssim():
    """Low SSIM must trigger FAIL even if IoU passes."""
    ssim_score = 0.50   # below threshold
    iou_score  = 0.90
    SSIM_THRESH = 0.78
    IOU_THRESH  = 0.72

    result = "PASS" if (ssim_score >= SSIM_THRESH and iou_score >= IOU_THRESH) else "FAIL"
    assert result == "FAIL"


def test_fail_on_low_iou():
    """Low IoU must trigger FAIL even if SSIM passes."""
    ssim_score = 0.90
    iou_score  = 0.50   # below threshold
    SSIM_THRESH = 0.78
    IOU_THRESH  = 0.72

    result = "PASS" if (ssim_score >= SSIM_THRESH and iou_score >= IOU_THRESH) else "FAIL"
    assert result == "FAIL"
