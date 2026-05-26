# Methodology

This document explains the technical reasoning behind every design decision in the Image Comparison Autotest pipeline. It is intended for reviewers who want to understand *why* each component was built the way it was, not just *what* it does.

---

## Problem context

When Siemens NX exports a 3D CAD model as a 3D PDF Technical Data Package (TDP), the exported views must exactly match the original CAD views. Manual inspection is unreliable at scale — subtle camera shifts, missing boundary edges, and geometry drops go undetected by human reviewers.

This pipeline provides an automated, objective, metric-driven quality gate.

---

## Stage 1 — Ground-truth reference generation (`stl_renderer.py`)

### Why STL as the reference source?

The `.stl` format stores only vertex coordinates and outward-facing facet normals. It contains no proprietary rendering hints, no NX display preferences, no metadata. This makes it a **renderer-independent ground truth** — any render produced from it is mathematically traceable to the original mesh with no dependency on the publishing pipeline being tested.

### Why orthographic projection?

Orthographic projection eliminates perspective foreshortening, preserving the true physical proportions of the geometry. A perspective projection would scale facets closer to the camera differently from those farther away, introducing distance-dependent silhouette distortion that would contaminate the IoU calculation in Stage 3.

Setting `ax.set_proj_type('ortho')` and `ax.set_box_aspect(None)` enforces this.

### Why the headless matplotlib Agg backend?

The Agg backend operates independently of system display hardware, guaranteeing mathematically identical pixel outputs across different operating systems and server environments. This determinism is essential for a CI-grade validation tool: a render that varies between a developer's Windows machine and a Linux build agent would silently corrupt the comparison metrics.

### Why Lambertian shading?

```
shade_i = 0.30 + 0.70 * max(n_i · l, 0)
```

- **0.30 ambient baseline** — prevents surface facets oriented away from the light from dropping to pure black, which would punch holes in the silhouette mask during Stage 3.
- **0.70 diffuse term** — provides enough topological contrast to distinguish geometry without introducing volatile specular highlights that vary across renderers.
- **Light parallel to camera axis** — ensures consistent shading regardless of view angle.

### Why 8 specific views?

The elevation/azimuth angles in `config.py` correspond exactly to the system-defined view set in Siemens NX: Top, Front, Right, Back, Bottom, Left, Isometric, Trimetric. Matching these exactly makes Stage 1 output directly comparable view-by-view with the named views an engineer authors in NX before publishing.

---

## Stage 2 — TDP view reconstruction (`tdp_reconstructor.py`)

### Why not extract the embedded poster images from the PDF?

The NX publishing pipeline frequently omits the `/P` poster streams entirely, or renders them at compression ratios unsuitable for precise geometric comparison. Relying on raster extraction would make the pipeline brittle.

### Why C2W matrix extraction instead?

Every named view in a 3D PDF's `/VA` (View Array) contains a 12-element `/C2W` Camera-to-World transformation matrix. This matrix is:
- **Always present** — it is required for the PDF reader to render the view
- **Mathematically deterministic** — exact floating-point values, no compression
- **Complete** — fully defines camera position, orientation, and up-vector

This makes C2W extraction far more reliable than raster extraction.

### How the C2W matrix is structured

```
C2W = [r | u | b | p]

Indices 0–2  : right vector (r)  — camera's local X axis
Indices 3–5  : up vector    (u)  — camera's local Y axis
Indices 6–8  : back vector  (b)  — camera's local Z axis (camera looks in -b direction)
Indices 9–11 : position     (p)  — camera's world coordinates
```

### Deriving elevation and azimuth from the back vector

The view direction is `v = -b` (camera looks opposite to its back vector).

matplotlib's 3D camera always points at the origin, so its view direction is `-pos`. Setting these equal:

```
v = -pos
=> pos = -v = b

elevation = arcsin(clip(v_z, -1, 1))
azimuth   = arctan2(-v_y, -v_x)
```

The `clip()` is critical — IEEE-754 floating-point drift can produce values like `1.0000000002`, which makes `arcsin` return `NaN` and silently produce a blank render.

### Why re-render the STL instead of using the PDF's embedded geometry?

Re-rendering from the same STL used in Stage 1 ensures both images come from identical geometry. Any difference in Stage 3 is therefore purely due to camera orientation errors in the PDF, not geometry differences between the STL and the PDF's internal PRC mesh.

---

## Stage 3 — Silhouette comparison (`silhouette_comparator.py`)

### Why silhouette comparison instead of direct RGB comparison?

Two renderers producing the same geometry will disagree on exact RGB values for boundary pixels due to different anti-aliasing kernels, shading models, and sub-pixel coverage resolution. A raw pixel-difference metric would flag these as defects when the underlying geometry is identical.

By abstracting to binary silhouettes, the comparison becomes immune to rendering style differences and focuses purely on structural shape.

### Foreground mask extraction

```python
mask(x,y) = 1  if  max_c |I(x,y,c) - BG(c)| > 40  else 0
```

- **Channel maximum** (not average) — ensures a strong shift in even one RGB channel registers as foreground. Averaging would dilute single-channel shifts and risk dropping legitimate geometry.
- **Threshold = 40** — empirically tuned: below ~20 catches JPEG compression speckle; above ~60 erodes thin shaded faces oriented edge-on to the camera.
- **Dual background handling** — Stage 1 renders on dark background (`#1E2025`), Stage 2 on white (`#FFFFFF`). The mask extractor applies conditionally tailored thresholds to handle both.

### Why morphological dilation?

Different rendering engines draw boundary edges at slightly different widths (e.g. one engine renders a 2px stroke, another renders 3px). Without dilation, this sub-pixel mismatch would trigger false failures on geometrically identical renders.

A **9×9 rectangular structuring element** dilates each silhouette outward by ~4 pixels, safely absorbing 1–2 pixel edge-width differences.

**Why 9×9 specifically?**
- Below 5×5: fails to absorb full cross-engine edge-width difference
- Above 13×13: begins to bridge across genuinely separate features (e.g. fusing a drilled hole with surrounding material), masking real defects
- 9×9 sits comfortably between these failure modes

### SSIM — structural similarity index

```
SSIM(x,y) = [(2μ_x μ_y + C1)(2σ_xy + C2)] / [(μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)]
```

SSIM uses a sliding 7×7 window and evaluates luminance, contrast, and structure locally. Applied to binary masks it detects:
- Boundary deformations
- Directional skewing
- Rotational shifts (which IoU would miss if total area is preserved)

### IoU — intersection over union

```
IoU = |M_STL ∩ M_TDP| / |M_STL ∪ M_TDP|
```

IoU measures pure area overlap:
- Missing geometry: shrinks intersection, IoU drops
- Extra geometry: inflates union, IoU drops
- Score is directly proportional to defect size

### Why both SSIM and IoU together?

| Metric | Catches | Misses |
|--------|---------|--------|
| SSIM   | Boundary distortion, rotation | Large uniform area shifts |
| IoU    | Area-level geometry loss/gain | Rotation that preserves area |

Together they eliminate each other's blind spots. A defect must fool both metrics to go undetected — which is practically impossible for any meaningful geometric error.

### Pass/Fail thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| SSIM   | ≥ 0.78    | Calibrated on confirmed-correct exports; separates rendering noise floor from visible defects |
| IoU    | ≥ 0.72    | Tight enough to flag missing geometry; loose enough to absorb dilation noise |

Both thresholds must be met simultaneously (AND rule). A high score on one cannot mask a failure on the other.

### Difference map color scheme

| Color  | Meaning |
|--------|---------|
| Green  | Geometry present in both — correct overlap |
| Red    | Present in STL, missing in TDP — geometry dropped during export |
| Yellow | Present in TDP, missing in STL — spurious geometry added |

---

## CI/CD deployability

The pipeline was deliberately designed to run without:
- Active NX CAD license
- Adobe Acrobat installation
- GUI display hardware (headless Agg backend)

This makes it directly deployable on standard Linux CI servers using only Python standard libraries and open-source packages.
