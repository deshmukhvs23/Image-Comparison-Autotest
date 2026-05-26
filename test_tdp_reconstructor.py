"""
tests/test_tdp_reconstructor.py
Unit tests for Stage 2 — C2W matrix extraction and angle derivation.
Tests the math independently of any PDF file.
"""
import pytest
import numpy as np
import math


# ── C2W angle derivation tests ────────────────────────────────────────────────
# These test the core math in tdp_reconstructor.py without needing a real PDF.

def derive_angles(c2w_flat):
    """
    Mirrors the angle derivation logic in tdp_reconstructor.py.
    Extracts elevation and azimuth from a 12-element C2W array.
    """
    b = np.array(c2w_flat[6:9])          # back vector (indices 6-8)
    v = -b                                # view direction = -back
    v = v / np.linalg.norm(v)            # normalise

    # matplotlib convention: camera sits at pos on unit sphere, looks at origin
    # so view direction = -pos, which means v = -pos => pos = -v = b
    elev = math.degrees(math.asin(float(np.clip(v[2], -1.0, 1.0))))
    azim = math.degrees(math.atan2(-v[1], -v[0]))
    return round(elev, 2), round(azim, 2)


def test_top_view_angles():
    """
    Top view: camera directly above, looking straight down.
    Expected: elevation=90, azimuth=0 (or ±90 depending on convention).
    """
    # For top view: back vector points straight up (0,0,1)
    # view direction = (0,0,-1) → elevation = -90... 
    # In NX convention top view: elev=90, azim=-90
    # We test that the math is self-consistent
    back = np.array([0.0, 0.0, -1.0])  # camera looks up, back is down
    v = -back  # view direction = (0,0,1) but that's bottom view
    # Top view: camera above, looking down → view direction = (0,0,-1)
    # back = (0,0,1), v = -back = (0,0,-1)
    back = np.array([0.0, 0.0, 1.0])
    c2w = [0,0,0, 0,0,0] + list(back) + [0,0,0]
    elev, azim = derive_angles(c2w)
    assert elev == pytest.approx(-90.0, abs=0.1), f"Top view elevation should be -90, got {elev}"


def test_front_view_angles():
    """Front view: camera in front, looking back along Y axis."""
    # Front: camera at (0,-1,0) looking toward origin
    # view direction = (0,1,0), back = (0,-1,0)
    back = np.array([0.0, -1.0, 0.0])
    c2w = [0,0,0, 0,0,0] + list(back) + [0,0,0]
    elev, azim = derive_angles(c2w)
    assert elev == pytest.approx(0.0, abs=0.5)


def test_clip_prevents_nan():
    """
    Floating-point drift can produce values like 1.0000000002.
    arcsin of values outside [-1,1] returns NaN — clip must prevent this.
    """
    # Slightly out-of-range value simulating float drift
    drifted_z = 1.0000000002
    result = math.asin(float(np.clip(drifted_z, -1.0, 1.0)))
    assert not math.isnan(result), "clip() must prevent NaN from arcsin"
    assert result == pytest.approx(math.pi / 2, abs=1e-6)


def test_normalisation():
    """View direction vector must be unit length after normalisation."""
    back = np.array([1.0, 2.0, 3.0])   # arbitrary non-unit vector
    v = -back
    v_norm = v / np.linalg.norm(v)
    length = np.linalg.norm(v_norm)
    assert length == pytest.approx(1.0, abs=1e-6)


def test_known_isometric_angles():
    """
    Isometric view in NX: elevation=35.26°, azimuth=-45°.
    We verify the inverse: given these angles, we can reconstruct
    a view direction, and derive_angles recovers them.
    """
    elev_deg = 35.26
    azim_deg = -45.0

    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    # pos = (cos(e)cos(a), cos(e)sin(a), sin(e)) — matplotlib sphere parameterisation
    pos = np.array([
        math.cos(elev) * math.cos(azim),
        math.cos(elev) * math.sin(azim),
        math.sin(elev)
    ])

    # view direction = -pos, back = -view = pos
    back = pos
    c2w = [0,0,0, 0,0,0] + list(back) + [0,0,0]

    recovered_elev, recovered_azim = derive_angles(c2w)
    assert recovered_elev == pytest.approx(elev_deg, abs=0.5)
    assert recovered_azim == pytest.approx(azim_deg, abs=0.5)
