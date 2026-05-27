"""
tests/test_stl_renderer.py
Unit tests for Stage 1 — STL renderer.
"""
import os
import pytest
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

SAMPLE_STL = os.path.join(os.path.dirname(__file__), "../sample_data/simple_block.stl")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "../stl_views")

EXPECTED_VIEWS = ["Top", "Front", "Right", "Back", "Bottom", "Left", "Isometric", "Trimetric"]


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_stl_file_exists():
    """Sample STL must exist before renderer can run."""
    assert os.path.exists(SAMPLE_STL), f"Sample STL not found at {SAMPLE_STL}"


def test_stl_file_not_empty():
    """STL file must have non-zero size."""
    size = os.path.getsize(SAMPLE_STL)
    assert size > 0, "STL file is empty"


def test_eight_views_generated(tmp_path):
    """Stage 1 must produce exactly 8 PNG files."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    try:
        from stl_renderer import render_stl_views
        output_dir = str(tmp_path / "stl_views")
        render_stl_views(SAMPLE_STL, output_dir=output_dir)

        generated = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        assert len(generated) == 8, f"Expected 8 PNGs, got {len(generated)}"
    except ImportError:
        pytest.skip("stl_renderer module not available in this environment")


def test_expected_view_names_present(tmp_path):
    """Each of the 8 NX standard views must have a corresponding PNG."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    try:
        from stl_renderer import render_stl_views
        output_dir = str(tmp_path / "stl_views")
        render_stl_views(SAMPLE_STL, output_dir=output_dir)

        generated_names = [f.replace(".png", "") for f in os.listdir(output_dir)]
        for view in EXPECTED_VIEWS:
            assert view in generated_names, f"Missing view: {view}"
    except ImportError:
        pytest.skip("stl_renderer module not available in this environment")


def test_png_files_not_empty(tmp_path):
    """All generated PNGs must have non-zero file size."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    try:
        from stl_renderer import render_stl_views
        output_dir = str(tmp_path / "stl_views")
        render_stl_views(SAMPLE_STL, output_dir=output_dir)

        for fname in os.listdir(output_dir):
            if fname.endswith(".png"):
                size = os.path.getsize(os.path.join(output_dir, fname))
                assert size > 1000, f"{fname} is suspiciously small ({size} bytes)"
    except ImportError:
        pytest.skip("stl_renderer module not available in this environment")
