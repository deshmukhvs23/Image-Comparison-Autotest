# Results

This document presents results from three test runs demonstrating the pipeline's ability to correctly validate clean exports, detect complete failures, and isolate partial failures.

| Test | Description | Result |
|---|---|---|
| Test 1 | Clean export — correct 3D PDF | 8/8 PASS |
| Test 2 | Mixed — 6 clean views, 2 mismatched views | 6/8 PASS |
| Test 3 | Full failure — completely mismatched PDF | 0/8 PASS |

---

## Test environment

| Parameter | Value |
|---|---|
| Test model | simple_block.stl — rectangular block (2.0 × 1.0 × 1.5 units) |
| Render DPI | 150 |
| Canvas size | 800 × 600 |
| Dilation footprint | 9 × 9 |
| SSIM threshold | ≥ 0.78 |
| IoU threshold | ≥ 0.72 |

---

## Test 1 — Clean export (8/8 PASS)

STL geometry and 3D PDF views are geometrically consistent across all views.

| View | SSIM | IoU | Status |
|---|---|---|---|
| Back | 1.0000 | 100.0% | ✓ PASS |
| Bottom | 1.0000 | 100.0% | ✓ PASS |
| Front | 1.0000 | 100.0% | ✓ PASS |
| Isometric | 0.9149 | 91.5% | ✓ PASS |
| Left | 1.0000 | 100.0% | ✓ PASS |
| Right | 1.0000 | 100.0% | ✓ PASS |
| Top | 1.0000 | 100.0% | ✓ PASS |
| Trimetric | 0.8953 | 89.3% | ✓ PASS |

**Overall: 8/8 PASS**

### Analysis
Six orthographic views achieved perfect SSIM=1.0 and IoU=100%. Isometric and Trimetric show minor reductions due to sub-degree floating-point drift in C2W matrix derivation — both clear both thresholds comfortably. Diff maps are fully green with zero red or yellow pixels.

---

## Test 2 — Mixed result (6/8 PASS)

Six orthographic views match correctly. Two non-orthographic views (Isometric, Trimetric) are mismatched — the embedded C2W matrices correspond to a different model geometry. This is the most realistic test scenario: a TDP export where most views are valid but specific views have rendering artifacts.

| View | SSIM | IoU | Status | Diff map |
|---|---|---|---|---|
| Back | 1.0000 | 100.0% | ✓ PASS | Fully green |
| Bottom | 1.0000 | 100.0% | ✓ PASS | Fully green |
| Front | 1.0000 | 100.0% | ✓ PASS | Fully green |
| Isometric | 0.7024 | 62.1% | ✗ FAIL | Green core, red flanges, yellow notch |
| Left | 1.0000 | 100.0% | ✓ PASS | Fully green |
| Right | 1.0000 | 100.0% | ✓ PASS | Fully green |
| Top | 1.0000 | 100.0% | ✓ PASS | Fully green |
| Trimetric | 0.6338 | 55.0% | ✗ FAIL | Green upper, red left, split silhouette |

**Overall: 6/8 PASS**

### Passing views — Back, Bottom, Front, Left, Right, Top
All six orthographic views show SSIM=1.0 and IoU=100%. Diff maps are entirely green — perfect silhouette overlap with zero missing or extra geometry. The pipeline correctly identifies these views as valid and does not produce false failures.

### Failing view — Isometric (SSIM=0.7024, IoU=62.1%)
STL shows a clean hexagonal isometric silhouette. TDP shows a T-shaped cutout silhouette — the exported model has a T-slot feature absent from the reference STL geometry.

Diff map breakdown:
- **Green** (centre) — shared geometry between STL and TDP
- **Red** (flanges, left and right) — geometry present in STL but missing from TDP export
- **Yellow** (notch) — extra geometry present in TDP not in STL

Both red and yellow regions appear simultaneously — indicating the TDP export contains a different model, not just a shifted camera. SSIM=0.70 falls below the 0.78 threshold; IoU=62.1% falls below the 0.72 threshold. Correctly flagged as FAIL.

### Failing view — Trimetric (SSIM=0.6338, IoU=55.0%)
STL shows a single continuous hexagonal silhouette. TDP shows the silhouette split into two disconnected parts — an upper trapezoid and a lower rounded rectangle — indicating a physical gap in the exported model.

Diff map breakdown:
- **Green** (upper portion and lower block) — partial overlap where shapes coincide
- **Red** (left region and gaps) — geometry present in STL but absent from the split TDP silhouette
- **Yellow** (corners) — extra geometry in TDP extending beyond the STL boundary

SSIM=0.63 and IoU=55.0% both fall well below thresholds. Correctly flagged as FAIL.

### Key insight
The pipeline successfully distinguishes between views where the export is correct (SSIM=1.0, fully green) and views where the export is defective (SSIM<0.78, red+yellow regions) — no false positives, no false negatives.

---

## Test 3 — Full failure (0/8 PASS)

A completely mismatched PDF where all embedded C2W matrices correspond to a different model.

| View | SSIM | IoU | Status | Failure type |
|---|---|---|---|---|
| Back | 0.4726 | 36.1% | ✗ FAIL | Wrong aspect ratio — TDP renders narrow vertical strips |
| Bottom | 0.2344 | 9.4% | ✗ FAIL | Severe mismatch — TDP shows two small disconnected fragments |
| Front | 0.4747 | 36.4% | ✗ FAIL | Wrong aspect ratio — same pattern as Back |
| Isometric | 0.7024 | 62.1% | ✗ FAIL | Shape mismatch — TDP has T-shaped cutout absent from STL |
| Left | 0.5340 | 44.7% | ✗ FAIL | Wrong aspect ratio — TDP strips vs STL rectangle |
| Right | 0.5329 | 44.5% | ✗ FAIL | Wrong aspect ratio — similar to Left |
| Top | 0.2339 | 9.3% | ✗ FAIL | Severe mismatch — TDP shows two small mispositioned fragments |
| Trimetric | 0.6338 | 55.0% | ✗ FAIL | Shape mismatch — TDP silhouette split into two parts |

**Overall: 0/8 PASS**

Scores scale with defect severity — IoU ~9% for catastrophic misalignment (Top, Bottom), IoU ~36% for severe aspect ratio errors (Back, Front), IoU 55–62% for partial structural mismatch (Isometric, Trimetric).

---

## Score comparison across all three tests

| View | Test 1 Clean | Test 2 Mixed | Test 3 Fail |
|---|---|---|---|
| Back | 1.00 / 100% ✓ | 1.00 / 100% ✓ | 0.47 / 36% ✗ |
| Bottom | 1.00 / 100% ✓ | 1.00 / 100% ✓ | 0.23 / 9% ✗ |
| Front | 1.00 / 100% ✓ | 1.00 / 100% ✓ | 0.47 / 36% ✗ |
| Isometric | 0.91 / 92% ✓ | 0.70 / 62% ✗ | 0.70 / 62% ✗ |
| Left | 1.00 / 100% ✓ | 1.00 / 100% ✓ | 0.53 / 45% ✗ |
| Right | 1.00 / 100% ✓ | 1.00 / 100% ✓ | 0.53 / 45% ✗ |
| Top | 1.00 / 100% ✓ | 1.00 / 100% ✓ | 0.23 / 9% ✗ |
| Trimetric | 0.90 / 89% ✓ | 0.63 / 55% ✗ | 0.63 / 55% ✗ |

---

## Validation summary

| Property | Evidence |
|---|---|
| No false positives | Test 2 passing views all score SSIM=1.0, IoU=100% — clean views never flagged |
| No false negatives | Test 2 failing views correctly detected despite 6/8 views passing |
| Severity scaling | Scores drop proportionally — IoU 9% catastrophic, 62% partial mismatch |
| Defect type isolation | Diff maps show red-only (geometry dropped) vs red+yellow (different model) |
| Threshold placement | Clean/failure score distributions have zero overlap |
