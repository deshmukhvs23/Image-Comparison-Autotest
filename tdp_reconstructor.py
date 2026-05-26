"""
╔══════════════════════════════════════════════════════════════╗
║  STAGE 2 — 3D PDF View Capture                               ║
║  Reads the camera matrices (C2W) from each named view in     ║
║  the 3D PDF, converts them to elev/azim angles, and renders  ║
║  the matching STL from those exact viewpoints.               ║
║                                                              ║
║  This produces geometrically accurate TDP-aligned views      ║
║  without requiring Adobe Acrobat.                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import os
import sys
import logging
import numpy as np
import pikepdf
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

VIEW_NAMES = ["Top", "Front", "Right", "Back", "Bottom", "Left", "Isometric", "Trimetric"]

BG_HEX     = "#FFFFFF"     # White background — matches TDP PDF background
FACE_COLOR = "#C8C8C8"
EDGE_COLOR = "#505050"

# ---------------------------------------------------------------------------
# C2W → elev / azim conversion
# ---------------------------------------------------------------------------

def c2w_to_elev_azim(c2w_values: list) -> tuple:
    """
    PDF 3D stores C2W column-major: [right | up | back | position]
    indices 6,7,8 = back vector = camera -Z axis in world space.
    back = -viewDir  →  viewDir = -back
    matplotlib camera pos:  (cos(e)*cos(a), cos(e)*sin(a), sin(e))
    viewDir_mpl = -pos, so:
      sin(e) = viewDir[2]  →  elev = arcsin( viewDir[2] )
      azim   = arctan2(-viewDir[1], -viewDir[0])
    """
    back     = np.array([float(c2w_values[6]),
                         float(c2w_values[7]),
                         float(c2w_values[8])])
    view_dir = -back
    elev = float(np.degrees(np.arcsin(np.clip(view_dir[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(-view_dir[1], -view_dir[0])))
    return round(elev, 2), round(azim, 2)

# ---------------------------------------------------------------------------
# PDF view discovery
# ---------------------------------------------------------------------------

def _clean_name(raw) -> str:
    return str(raw).split(".")[0].strip().replace("/", "")

def _map_pdf_name(pdf_name: str, tri_seen: bool) -> tuple:
    n = pdf_name.strip().lower()
    if "top"       in n: return "Top",       tri_seen
    if "front"     in n: return "Front",     tri_seen
    if "right"     in n: return "Right",     tri_seen
    if "back"      in n or "rear" in n: return "Back", tri_seen
    if "bottom"    in n: return "Bottom",    tri_seen
    if "left"      in n: return "Left",      tri_seen
    if "trimetric" in n or "tri" in n: return "Trimetric", True
    if "isometric" in n or "iso" in n:
        return ("Isometric" if not tri_seen else "Trimetric"), tri_seen
    return None, tri_seen

def extract_views_from_pdf(pdf_path: str) -> list:
    """
    Returns list of dicts:
      { name, pdf_name, elev, azim }
    """
    views     = []
    tri_seen  = False
    used      = set()

    with pikepdf.open(pdf_path) as pdf:
        for page in pdf.pages:
            if "/Annots" not in page:
                continue
            for ann in page.Annots:
                if ann.get("/Subtype") != "/3D":
                    continue
                dd = ann.get("/3DD")
                if dd is None:
                    continue
                va = dd.get("/VA") or dd.get("/Views")
                if va is None or len(va) == 0:
                    logger.warning("No /VA found — falling back to 8 standard NX angles.")
                    return []

                logger.info(f"Found {len(va)} views in /VA.")
                for view_obj in va:
                    name_obj = (view_obj.get("/XN") or
                                view_obj.get("/IN") or
                                view_obj.get("/Name"))
                    pdf_name  = _clean_name(name_obj) if name_obj else "Unknown"
                    s1_name, tri_seen = _map_pdf_name(pdf_name, tri_seen)

                    if s1_name is None or s1_name in used:
                        for vn in VIEW_NAMES:
                            if vn not in used:
                                s1_name = vn
                                break
                        else:
                            s1_name = pdf_name
                    used.add(s1_name)

                    c2w = view_obj.get("/C2W")
                    if c2w is not None:
                        elev, azim = c2w_to_elev_azim(c2w)
                    else:
                        # Fallback: use standard NX hardcoded angles
                        fallback = {
                            "Top": (90, -90), "Front": (0, -90),
                            "Right": (0, 0), "Back": (0, 90),
                            "Bottom": (-90, -90), "Left": (0, 180),
                            "Isometric": (35.26, -45), "Trimetric": (20, -35),
                        }
                        elev, azim = fallback.get(s1_name, (35.26, -45))

                    views.append({
                        "name":     s1_name,
                        "pdf_name": pdf_name,
                        "elev":     elev,
                        "azim":     azim,
                    })
                return views
    return views

# ---------------------------------------------------------------------------
# STL rendering  (white background to match TDP)
# ---------------------------------------------------------------------------

def load_mesh(stl_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(stl_path, force="mesh")
    mesh.apply_translation(-mesh.centroid)
    ext = mesh.extents.max()
    if ext > 0:
        mesh.apply_scale(1.0 / ext)
    return mesh

def render_view(mesh: trimesh.Trimesh, view: dict, dpi: int = 150) -> plt.Figure:
    elev, azim = view["elev"], view["azim"]
    verts, faces = mesh.vertices, mesh.faces
    tris = verts[faces]

    e, a  = np.radians(elev), np.radians(azim)
    light = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    dot   = np.clip(mesh.face_normals @ light, 0, 1)
    shade = 0.30 + 0.70 * dot
    base  = np.array(mcolors.to_rgb(FACE_COLOR))
    colors_rgba = np.hstack([
        np.clip(shade[:, None] * base[None, :], 0, 1),
        np.full((len(faces), 1), 0.95)
    ])

    fig = plt.figure(figsize=(8, 6), facecolor=BG_HEX)
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG_HEX)
    ax.set_proj_type('ortho')
    ax.set_box_aspect(None)

    ax.add_collection3d(Poly3DCollection(
        tris, facecolors=colors_rgba,
        edgecolors=EDGE_COLOR, linewidths=0.3, zsort="average"))

    pad = 0.05
    lo, hi = verts.min(0) - pad, verts.max(0) + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # Label bar — black on white, same layout as TDP
    fig.text(0.5, 0.03, view["name"], ha="center", va="bottom",
             fontdict={"family": "monospace", "size": 13,
                       "color": "#3A7EB5", "weight": "bold"})
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def capture_all_views(pdf_path: str, stl_path: str,
                      output_dir: str, dpi: int = 150):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSTAGE 2 — 3D PDF View Capture")
    print(f"  PDF   : {pdf_path}")
    print(f"  STL   : {stl_path}")
    print(f"  Output: {output_dir}\n")

    views = extract_views_from_pdf(pdf_path)

    if not views:
        # No /VA in PDF — use standard NX angles
        logger.warning("Using hardcoded NX standard angles.")
        views = [
            {"name": "Top",       "pdf_name": "Top",       "elev":  90,    "azim":  -90},
            {"name": "Front",     "pdf_name": "Front",     "elev":   0,    "azim":  -90},
            {"name": "Right",     "pdf_name": "Right",     "elev":   0,    "azim":    0},
            {"name": "Back",      "pdf_name": "Back",      "elev":   0,    "azim":   90},
            {"name": "Bottom",    "pdf_name": "Bottom",    "elev": -90,    "azim":  -90},
            {"name": "Left",      "pdf_name": "Left",      "elev":   0,    "azim":  180},
            {"name": "Isometric", "pdf_name": "Isometric", "elev":  35.26, "azim":  -45},
            {"name": "Trimetric", "pdf_name": "Trimetric", "elev":  20,    "azim":  -35},
        ]

    mesh = load_mesh(stl_path)
    print(f"  {'View':<12} {'PDF Name':<14} {'elev':>7} {'azim':>7}")
    print(f"  {'-'*12} {'-'*14} {'-'*7} {'-'*7}")

    for view in views:
        print(f"  {view['name']:<12} {view['pdf_name']:<14} "
              f"{view['elev']:>7.2f} {view['azim']:>7.2f}", end="  ")
        fig   = render_view(mesh, view, dpi=dpi)
        fpath = os.path.join(output_dir, f"{view['name']}.png")
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor=BG_HEX)
        plt.close(fig)
        print("✓")

    print(f"\nDone. {len(views)} views saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Capture TDP views from 3D PDF camera matrices + STL renderer")
    parser.add_argument("pdf",
        help="Path to input 3D PDF file")
    parser.add_argument("stl",
        help="Path to matching STL file")
    parser.add_argument("-o", "--output",
        default=r"D:\MTech\ImageComparisonAutotest\tdp_views",
        help="Output directory for PNG views")
    parser.add_argument("--dpi", type=int, default=150,
        help="Render DPI (default: 150)")
    args = parser.parse_args()
    capture_all_views(args.pdf, args.stl, args.output, args.dpi)

if __name__ == "__main__":
    main()
