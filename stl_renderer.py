"""
╔══════════════════════════════════════════════════════════════╗
║  STAGE 1 — STL View Capture (Physical Fidelity)              ║
║  Renders 8 standard views with true aspect ratios            ║
║  using orthographic projection for MBD validation.           ║
╚══════════════════════════════════════════════════════════════╝
"""

import argparse
import os
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

# Standard views aligned with Siemens NX
VIEWS = [
    {"name": "Top",        "elev":  90,    "azim":  -90},
    {"name": "Front",      "elev":   0,    "azim":  -90},
    {"name": "Right",      "elev":   0,    "azim":    0},
    {"name": "Back",       "elev":   0,    "azim":   90},
    {"name": "Bottom",     "elev": -90,    "azim":  -90},
    {"name": "Left",       "elev":   0,    "azim":  180},
    {"name": "Isometric",  "elev":  35.26, "azim":  -45},
    {"name": "Trimetric",  "elev":  20,    "azim":  -35},
]

BG_HEX    = "#1E2025"
FACE_COLOR = "#C8C8C8"   # Neutral gray — professional CAD appearance
EDGE_COLOR = "#505050"

def load_mesh(stl_path: str) -> trimesh.Trimesh:
    """Loads mesh and scales to unit size for consistent framing."""
    mesh = trimesh.load(stl_path, force="mesh")
    mesh.apply_translation(-mesh.centroid)
    ext = mesh.extents.max()
    if ext > 0:
        mesh.apply_scale(1.0 / ext)
    return mesh

def render_view(mesh: trimesh.Trimesh, view: dict, dpi: int = 150) -> plt.Figure:
    """Renders mesh using orthographic projection and true proportions."""
    elev, azim = view["elev"], view["azim"]
    verts, faces = mesh.vertices, mesh.faces
    tris = verts[faces]

    # Lambert shading for depth cues
    e, a = np.radians(elev), np.radians(azim)
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

    # Orthographic projection — matches CAD standard views exactly
    ax.set_proj_type('ortho')
    # None allows true model aspect ratios (not forced cube)
    ax.set_box_aspect(None)

    ax.add_collection3d(Poly3DCollection(
        tris, facecolors=colors_rgba,
        edgecolors=EDGE_COLOR, linewidths=0.3, zsort="average"))

    # Tight limits from actual model extents
    pad = 0.05
    lo, hi = verts.min(0) - pad, verts.max(0) + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    fig.text(0.5, 0.03, view["name"], ha="center", va="bottom",
             fontdict={"family": "monospace", "size": 13,
                       "color": "#A0C8E6", "weight": "bold"})
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig

def capture_all_views(stl_path: str, output_dir: str, dpi: int = 150):
    os.makedirs(output_dir, exist_ok=True)
    mesh = load_mesh(stl_path)
    print(f"\nSTAGE 1 — Rendering {len(VIEWS)} standard NX views...")
    print(f"  Model : {stl_path}")
    print(f"  Output: {output_dir}\n")
    for view in VIEWS:
        fig   = render_view(mesh, view, dpi=dpi)
        fpath = os.path.join(output_dir, f"{view['name']}.png")
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor=BG_HEX)
        plt.close(fig)
        print(f"  ✓ {view['name']:<12} → {view['name']}.png")
    print(f"\nDone. {len(VIEWS)} views saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: Render 8 NX standard views from STL")
    parser.add_argument("stl",
        help="Path to input STL file")
    parser.add_argument("--output", "-o",
        default=r"D:\MTech\ImageComparisonAutotest\stl_views",
        help="Output directory for PNG views")
    parser.add_argument("--dpi",
        type=int, default=150,
        help="Render DPI (default: 150)")
    args = parser.parse_args()
    capture_all_views(args.stl, args.output, args.dpi)
