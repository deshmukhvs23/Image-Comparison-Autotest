"""
╔══════════════════════════════════════════════════════════════╗
║  STAGE 3 — Geometric Silhouette Comparison                   ║
║  Validates STL views against 3D PDF TDP views                ║
║  using SSIM and IoU on binary silhouette masks.              ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import os
import base64
import logging
import numpy as np
from datetime import datetime
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim_fn
from skimage.morphology import dilation, footprint_rectangle

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Background colours for mask extraction
BG_NX  = (30, 32, 37)      # Stage 1 dark background (matplotlib)
BG_TDP = (255, 255, 255)    # Stage 2 white background (pdf2image / Poppler)

# Thresholds calibrated for cross-renderer geometric comparison
# Strict pixel-perfect thresholds (0.90/0.85) are too tight when two
# different rendering engines produce the same silhouette shape but with
# slightly different edge anti-aliasing and line widths.
SSIM_THRESHOLD = 0.78
IOU_THRESHOLD  = 0.72

# Dilation footprint: 9×9 absorbs inter-renderer edge width differences
DILATION_SIZE  = 9

@dataclass
class ViewResult:
    name:   str
    ssim:   float
    iou:    float
    status: str           # "PASS" or "FAIL"
    reason: str = ""      # human-readable fail reason

class ImageUtils:
    @staticmethod
    def load_raw(path: str) -> np.ndarray:
        try:
            img = Image.open(path).convert("RGB")
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
            return None

    @staticmethod
    def get_binary_mask(arr: np.ndarray, bg: tuple) -> np.ndarray:
        """Pixels that differ from bg by >40 on any channel → foreground."""
        h, _ = arr.shape[:2]
        diff = np.abs(arr.astype(np.int16) - np.array(bg, dtype=np.int16))
        mask = np.max(diff, axis=2) > 40
        mask[int(h * 0.90):, :] = False   # ignore label bar at bottom
        return mask

    @staticmethod
    def sync_and_binary(nx_raw: np.ndarray, tdp_raw: np.ndarray,
                        pad: int = 30) -> tuple:
        """
        Crops each image to its silhouette bounding box, resizes to a common
        canvas (800×600), then applies morphological dilation so that minor
        edge-width differences between renderers don't penalise the score.
        """
        m_nx  = ImageUtils.get_binary_mask(nx_raw,  BG_NX)
        m_tdp = ImageUtils.get_binary_mask(tdp_raw, BG_TDP)

        c_nx, c_tdp = np.argwhere(m_nx), np.argwhere(m_tdp)
        if c_nx.size == 0 or c_tdp.size == 0:
            return None, None

        def _crop_and_resize(mask, coords):
            y0, x0 = coords.min(axis=0) - pad
            y1, x1 = coords.max(axis=0) + pad
            h, w   = mask.shape
            crop   = mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            return Image.fromarray(crop).resize((800, 600), Image.NEAREST)

        nx_bin  = _crop_and_resize(m_nx,  c_nx)
        tdp_bin = _crop_and_resize(m_tdp, c_tdp)

        fprint = footprint_rectangle((DILATION_SIZE, DILATION_SIZE))
        nx_d   = dilation(np.array(nx_bin),  fprint)
        tdp_d  = dilation(np.array(tdp_bin), fprint)
        return nx_d, tdp_d

# ---------------------------------------------------------------------------
# Visual diff image
# ---------------------------------------------------------------------------

def _build_diff_image(nx_raw: np.ndarray, tdp_raw: np.ndarray,
                      nx_m: np.ndarray, tdp_m: np.ndarray,
                      result: ViewResult) -> Image.Image:
    """
    Side-by-side: STL silhouette | TDP silhouette | Overlap diff.
    - Green  = common silhouette (both renderers agree)
    - Red    = only in STL (missing in TDP)
    - Yellow = only in TDP (missing in STL)
    """
    H, W = 300, 400
    canvas = Image.new("RGB", (W * 3 + 4, H + 40), (20, 22, 26))
    draw   = ImageDraw.Draw(canvas)

    def _mask_to_img(mask, colour):
        img = Image.new("RGB", (W, H), (20, 22, 26))
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        m = Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
        m_arr = np.array(m) > 128
        overlay[m_arr] = colour
        return Image.blend(img, Image.fromarray(overlay), alpha=1.0)

    # Overlap diff
    both    = np.logical_and(nx_m, tdp_m)
    only_nx = np.logical_and(nx_m, ~tdp_m)
    only_td = np.logical_and(~nx_m, tdp_m)

    diff_img = np.full((600, 800, 3), (20, 22, 26), dtype=np.uint8)
    diff_img[both]    = (50, 200, 80)     # green  — match
    diff_img[only_nx] = (220, 50, 50)     # red    — STL only
    diff_img[only_td] = (220, 200, 50)    # yellow — TDP only
    diff_pil = Image.fromarray(diff_img).resize((W, H), Image.LANCZOS)

    nx_pil  = _mask_to_img(nx_m,  (100, 180, 255))
    tdp_pil = _mask_to_img(tdp_m, (255, 160,  60))

    canvas.paste(nx_pil,  (0,         0))
    canvas.paste(tdp_pil, (W + 2,     0))
    canvas.paste(diff_pil,(W * 2 + 4, 0))

    # Labels
    try:
        font_sm = ImageFont.truetype("arial.ttf", 12)
        font_hd = ImageFont.truetype("arialbd.ttf", 13)
    except Exception:
        font_sm = font_hd = ImageFont.load_default()

    col   = "#4FC3F7" if result.status == "PASS" else "#EF5350"
    title = f"{result.name}  [{result.status}]  SSIM={result.ssim:.4f}  IoU={result.iou:.1%}"
    draw.text((4, H + 6),  "STL (Stage 1)",      font=font_sm, fill="#90CAF9")
    draw.text((W + 6, H + 6), "TDP (Stage 2)",   font=font_sm, fill="#FFCC80")
    draw.text((W*2+6, H + 6), "Diff  ■Green=Match  ■Red=STL-only  ■Yellow=TDP-only",
              font=font_sm, fill="#CCCCCC")
    draw.text((4, H + 22), title, font=font_hd, fill=col)
    return canvas

# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(nx_dir: str, tdp_dir: str,
                   s_thresh: float = SSIM_THRESHOLD,
                   i_thresh: float = IOU_THRESHOLD,
                   diff_dir: str   = None) -> list:

    nx_path, tdp_path = Path(nx_dir).resolve(), Path(tdp_dir).resolve()

    if diff_dir:
        os.makedirs(diff_dir, exist_ok=True)

    common = sorted(set(os.listdir(nx_path)) & set(os.listdir(tdp_path)))
    common = [f for f in common if f.lower().endswith(".png")]

    print(f"\nSTAGE 3 — Geometric Silhouette Comparison")
    print(f"  STL views : {nx_path}")
    print(f"  TDP views : {tdp_path}")
    print(f"  Thresholds: SSIM ≥ {s_thresh:.2f}  |  IoU ≥ {i_thresh:.2f}")
    print(f"  Views found: {len(common)}\n")
    print(f"  {'View':<14} {'SSIM':>8} {'IoU':>8}  Status   Note")
    print(f"  {'-'*14} {'-'*8} {'-'*8}  {'-'*6}   ----")

    results = []
    for f in common:
        nx_raw  = ImageUtils.load_raw(nx_path  / f)
        tdp_raw = ImageUtils.load_raw(tdp_path / f)

        if nx_raw is None or tdp_raw is None:
            logger.warning(f"Skipping {f} — could not load one or both images.")
            continue

        nx_m, tdp_m = ImageUtils.sync_and_binary(nx_raw, tdp_raw)
        if nx_m is None:
            logger.warning(f"Skipping {f} — empty silhouette mask.")
            continue

        inter = np.logical_and(nx_m, tdp_m).sum()
        union = np.logical_or(nx_m, tdp_m).sum()
        iou   = inter / union if union > 0 else 0.0
        score = ssim_fn(nx_m.astype(np.uint8) * 255,
                        tdp_m.astype(np.uint8) * 255,
                        data_range=255)

        fail_reasons = []
        if score < s_thresh: fail_reasons.append(f"SSIM below {s_thresh:.2f}")
        if iou   < i_thresh: fail_reasons.append(f"IoU below {i_thresh:.0%}")

        status = "PASS" if not fail_reasons else "FAIL"
        reason = "; ".join(fail_reasons)

        r = ViewResult(name=f.replace(".png", ""), ssim=score,
                       iou=iou, status=status, reason=reason)
        results.append(r)

        icon = "✓" if status == "PASS" else "✗"
        note = f"  ← {reason}" if reason else ""
        print(f"  {icon} {r.name:<13} {score:>8.4f} {iou:>7.1%}  {status:<6}{note}")

        if diff_dir:
            diff_img = _build_diff_image(nx_raw, tdp_raw, nx_m, tdp_m, r)
            diff_img.save(os.path.join(diff_dir, f))

    # Summary
    passed = sum(1 for r in results if r.status == "PASS")
    total  = len(results)
    print(f"\n  {'─'*54}")
    print(f"  Result: {passed}/{total} views PASS")
    if passed == total:
        print("  ✓ OVERALL: PASS — STL geometry matches TDP")
    else:
        print("  ✗ OVERALL: FAIL — geometry mismatch detected")

    # HTML report
    if diff_dir:
        rpath = os.path.join(diff_dir, "report.html")
        _write_html_report(results, diff_dir, rpath,
                           s_thresh, i_thresh,
                           str(nx_path), str(tdp_path))
        print(f"\n  Diff images + report.html saved to: {diff_dir}")

    return results


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _img_to_b64(path: str) -> str:
    """Encode a PNG file as a base64 data URI so the HTML is self-contained."""
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()

def _write_html_report(results: list, diff_dir: str, out_path: str,
                        s_thresh: float, i_thresh: float,
                        nx_dir: str, tdp_dir: str):
    passed  = sum(1 for r in results if r.status == "PASS")
    total   = len(results)
    overall = "PASS" if passed == total else "FAIL"
    ts      = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # Build one card per view
    cards_html = ""
    for r in results:
        diff_file = os.path.join(diff_dir, f"{r.name}.png")
        img_tag   = (f'<img src="{_img_to_b64(diff_file)}" alt="{r.name} diff">'
                     if os.path.isfile(diff_file) else
                     '<p style="color:#888">Diff image not available</p>')

        status_cls = "pass" if r.status == "PASS" else "fail"
        note       = f'<span class="note">{r.reason}</span>' if r.reason else ""
        ssim_bar   = min(r.ssim, 1.0) * 100
        iou_bar    = r.iou * 100

        cards_html += f"""
        <div class="card {status_cls}">
          <div class="card-header">
            <span class="view-name">{r.name}</span>
            <span class="badge {status_cls}">{r.status}</span>
          </div>
          <div class="metrics">
            <div class="metric">
              <span class="mlabel">SSIM</span>
              <div class="bar-track"><div class="bar-fill {'bar-ok' if r.ssim >= s_thresh else 'bar-bad'}"
                   style="width:{ssim_bar:.1f}%"></div></div>
              <span class="mval">{r.ssim:.4f}</span>
            </div>
            <div class="metric">
              <span class="mlabel">IoU</span>
              <div class="bar-track"><div class="bar-fill {'bar-ok' if r.iou >= i_thresh else 'bar-bad'}"
                   style="width:{iou_bar:.1f}%"></div></div>
              <span class="mval">{r.iou:.1%}</span>
            </div>
          </div>
          {note}
          <div class="diff-wrap">{img_tag}</div>
          <div class="legend">
            <span class="dot green"></span>Match &nbsp;
            <span class="dot red"></span>STL only &nbsp;
            <span class="dot yellow"></span>TDP only
          </div>
        </div>"""

    overall_cls = "pass" if overall == "PASS" else "fail"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Geometric Validation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #13151a; color: #dde; font-family: 'Segoe UI', Arial, sans-serif;
          padding: 24px; }}
  h1   {{ font-size: 1.5rem; color: #90caf9; margin-bottom: 4px; }}
  .meta {{ font-size: 0.82rem; color: #778; margin-bottom: 20px; }}
  .summary {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
  .stat  {{ background: #1e2130; border-radius: 8px; padding: 14px 24px;
            min-width: 140px; text-align: center; }}
  .stat .val {{ font-size: 2rem; font-weight: bold; line-height: 1.1; }}
  .stat .lbl {{ font-size: 0.78rem; color: #778; margin-top: 4px; }}
  .overall-pass {{ color: #66bb6a; }} .overall-fail {{ color: #ef5350; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
           gap: 20px; }}
  .card {{ background: #1e2130; border-radius: 10px; overflow: hidden;
           border: 1px solid #2a2d3a; }}
  .card.pass {{ border-left: 4px solid #66bb6a; }}
  .card.fail {{ border-left: 4px solid #ef5350; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center;
                  padding: 10px 14px; background: #181b25; }}
  .view-name {{ font-size: 1rem; font-weight: 600; color: #cdd; letter-spacing:.5px; }}
  .badge {{ font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 20px; }}
  .badge.pass {{ background: #1b3a1e; color: #66bb6a; }}
  .badge.fail {{ background: #3a1b1b; color: #ef5350; }}
  .metrics {{ padding: 10px 14px 4px; }}
  .metric  {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  .mlabel  {{ font-size: 0.75rem; color: #778; width: 36px; flex-shrink:0; }}
  .mval    {{ font-size: 0.82rem; color: #aac; width: 60px; text-align: right;
              flex-shrink:0; }}
  .bar-track {{ flex: 1; height: 6px; background: #2c2f3e; border-radius: 4px;
                overflow: hidden; }}
  .bar-fill  {{ height: 100%; border-radius: 4px; }}
  .bar-ok    {{ background: #66bb6a; }}
  .bar-bad   {{ background: #ef5350; }}
  .note  {{ font-size: 0.75rem; color: #ef9a9a; padding: 2px 14px 6px; }}
  .diff-wrap {{ padding: 10px 14px; }}
  .diff-wrap img {{ width: 100%; border-radius: 4px; display: block; }}
  .legend {{ padding: 4px 14px 10px; font-size: 0.72rem; color: #778;
             display: flex; align-items: center; gap: 4px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .dot.green  {{ background: #32c850; }}
  .dot.red    {{ background: #dc3232; }}
  .dot.yellow {{ background: #dcc832; }}
  .thresh {{ font-size: 0.78rem; color: #556; margin-top: 20px; }}
</style>
</head>
<body>
<h1>Geometric Validation Report — MBD / TDP vs STL</h1>
<p class="meta">Generated: {ts} &nbsp;|&nbsp; STL: {nx_dir} &nbsp;|&nbsp; TDP: {tdp_dir}</p>

<div class="summary">
  <div class="stat">
    <div class="val overall-{overall_cls}">{overall}</div>
    <div class="lbl">Overall Result</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#90caf9">{total}</div>
    <div class="lbl">Views Compared</div>
  </div>
  <div class="stat">
    <div class="val overall-pass">{passed}</div>
    <div class="lbl">Pass</div>
  </div>
  <div class="stat">
    <div class="val overall-fail">{total - passed}</div>
    <div class="lbl">Fail</div>
  </div>
  <div class="stat">
    <div class="val" style="color:#aaa;font-size:1.2rem">{s_thresh:.2f} / {i_thresh:.0%}</div>
    <div class="lbl">SSIM / IoU Thresholds</div>
  </div>
</div>

<div class="grid">
{cards_html}
</div>

<p class="thresh">
  Diff legend: <b style="color:#32c850">■ Green</b> = both renderers agree &nbsp;
  <b style="color:#dc3232">■ Red</b> = present in STL, missing in TDP &nbsp;
  <b style="color:#dcc832">■ Yellow</b> = present in TDP, missing in STL
</p>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info(f"HTML report written: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Compare STL views vs TDP views geometrically")
    parser.add_argument("--nx_dir",
        default=r"D:\MTech\ImageComparisonAutotest\stl_views",
        help="Directory with Stage 1 STL view PNGs")
    parser.add_argument("--tdp_dir",
        default=r"D:\MTech\ImageComparisonAutotest\tdp_views",
        help="Directory with Stage 2 TDP view PNGs")
    parser.add_argument("--diff_dir",
        default=r"D:\MTech\ImageComparisonAutotest\diff_views",
        help="Output directory for diff images and JSON report")
    parser.add_argument("--ssim_thresh", type=float, default=SSIM_THRESHOLD,
        help=f"SSIM pass threshold (default: {SSIM_THRESHOLD})")
    parser.add_argument("--iou_thresh",  type=float, default=IOU_THRESHOLD,
        help=f"IoU pass threshold (default: {IOU_THRESHOLD})")
    args = parser.parse_args()

    run_comparison(
        nx_dir   = args.nx_dir,
        tdp_dir  = args.tdp_dir,
        s_thresh = args.ssim_thresh,
        i_thresh = args.iou_thresh,
        diff_dir = args.diff_dir,
    )
