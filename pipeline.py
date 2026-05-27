"""
pipeline.py
End-to-end Image Comparison Autotest pipeline.
Runs Stage 1 → Stage 2 → Stage 3 in sequence.

Usage:
    # Full pipeline
    python pipeline.py --stl sample_data/simple_block.stl --pdf path/to/tdp.pdf

    # Stage 1 only (generate STL reference views)
    python pipeline.py --stl sample_data/simple_block.stl --stage 1

    # Stage 3 only (compare existing stl_views/ and tdp_views/)
    python pipeline.py --stage 3

    # Skip Stage 2 if you don't have a 3D PDF — compare Stage 1 output with itself
    python pipeline.py --stl sample_data/simple_block.stl --self-test
"""

import argparse
import json
import os
import sys
import time

from config import STL_VIEWS_DIR, TDP_VIEWS_DIR, DIFF_VIEWS_DIR
from utils import ensure_dirs


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_stage1(stl_path: str) -> bool:
    """Stage 1: Render 8 orthographic views from STL geometry."""
    print("\n" + "─" * 60)
    print("STAGE 1 — Ground-Truth Reference Image Generation")
    print("─" * 60)

    try:
        from stl_renderer import render_stl_views
        ensure_dirs(STL_VIEWS_DIR)
        render_stl_views(stl_path, output_dir=STL_VIEWS_DIR)
        print(f"[Stage 1] ✓ Complete — views saved to {STL_VIEWS_DIR}/")
        return True
    except ImportError as e:
        print(f"[Stage 1] ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"[Stage 1] ✗ Failed: {e}")
        return False


def run_stage2(stl_path: str, pdf_path: str) -> bool:
    """Stage 2: Extract C2W matrices from PDF and reconstruct TDP views."""
    print("\n" + "─" * 60)
    print("STAGE 2 — TDP View Reconstruction from C2W Matrices")
    print("─" * 60)

    if not os.path.exists(pdf_path):
        print(f"[Stage 2] ✗ PDF not found: {pdf_path}")
        return False

    try:
        from tdp_reconstructor import reconstruct_tdp_views
        ensure_dirs(TDP_VIEWS_DIR)
        reconstruct_tdp_views(stl_path, pdf_path, output_dir=TDP_VIEWS_DIR)
        print(f"[Stage 2] ✓ Complete — views saved to {TDP_VIEWS_DIR}/")
        return True
    except ImportError as e:
        print(f"[Stage 2] ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"[Stage 2] ✗ Failed: {e}")
        return False


def run_stage3() -> dict:
    """Stage 3: Silhouette comparison and metric computation."""
    print("\n" + "─" * 60)
    print("STAGE 3 — Silhouette Comparison and Metric Computation")
    print("─" * 60)

    try:
        from silhouette_comparator import compare_views
        ensure_dirs(DIFF_VIEWS_DIR)
        results = compare_views(
            stl_dir=STL_VIEWS_DIR,
            tdp_dir=TDP_VIEWS_DIR,
            output_dir=DIFF_VIEWS_DIR,
        )
        print(f"[Stage 3] ✓ Complete — diff maps and report saved to {DIFF_VIEWS_DIR}/")
        return results
    except ImportError as e:
        print(f"[Stage 3] ✗ Import error: {e}")
        return {}
    except Exception as e:
        print(f"[Stage 3] ✗ Failed: {e}")
        return {}


def run_self_test(stl_path: str) -> dict:
    """
    Self-test mode: compare Stage 1 output against itself.
    All views should score SSIM=1.0 and IoU=100%.
    Useful for verifying the pipeline without a real 3D PDF.
    """
    import shutil
    print("\n[Self-test] Copying stl_views/ → tdp_views/ for self-comparison")
    ensure_dirs(TDP_VIEWS_DIR)

    for f in os.listdir(STL_VIEWS_DIR):
        if f.endswith(".png"):
            shutil.copy(
                os.path.join(STL_VIEWS_DIR, f),
                os.path.join(TDP_VIEWS_DIR, f)
            )
    print("[Self-test] ✓ Copied. Running Stage 3...")
    return run_stage3()


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(results: dict, elapsed: float) -> None:
    """Print a clean pass/fail summary table."""
    if not results:
        print("\n[Pipeline] No results to display.")
        return

    views     = results.get("views", {})
    passed    = sum(1 for v in views.values() if v.get("status") == "PASS")
    failed    = sum(1 for v in views.values() if v.get("status") == "FAIL")
    total     = len(views)
    overall   = results.get("overall", "UNKNOWN")

    print("\n" + "=" * 60)
    print(f"  RESULT: {overall}   ({passed}/{total} views passed)")
    print("=" * 60)
    print(f"  {'View':<14} {'SSIM':>8} {'IoU':>8} {'Status':>8}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8}")

    for view_name, data in sorted(views.items()):
        ssim   = data.get("ssim", 0)
        iou    = data.get("iou", 0)
        status = data.get("status", "?")
        icon   = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {view_name:<13} {ssim:>8.4f} {iou*100:>7.1f}% {status:>8}")

    print(f"\n  Thresholds: SSIM ≥ 0.78  |  IoU ≥ 0.72")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Report: {DIFF_VIEWS_DIR}/report.html")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Image Comparison Autotest — end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline:
    python pipeline.py --stl sample_data/simple_block.stl --pdf path/to/tdp.pdf

  Stage 1 only:
    python pipeline.py --stl sample_data/simple_block.stl --stage 1

  Stage 3 only (views already generated):
    python pipeline.py --stage 3

  Self-test (no PDF needed):
    python pipeline.py --stl sample_data/simple_block.stl --self-test
        """
    )
    parser.add_argument("--stl",       default=None,  help="Path to .stl file")
    parser.add_argument("--pdf",       default=None,  help="Path to 3D PDF TDP")
    parser.add_argument("--stage",     type=int, choices=[1, 2, 3], default=None,
                        help="Run a single stage only")
    parser.add_argument("--self-test", action="store_true",
                        help="Compare Stage 1 output against itself (no PDF needed)")
    parser.add_argument("--output",    default=None,
                        help="Save results JSON to this path")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    print("=" * 60)
    print("  IMAGE COMPARISON AUTOTEST PIPELINE")
    print("=" * 60)

    results = {}

    # ── Single stage mode ─────────────────────────────────────────────────────
    if args.stage == 1:
        if not args.stl:
            print("Error: --stl required for Stage 1"); sys.exit(1)
        run_stage1(args.stl)

    elif args.stage == 2:
        if not args.stl or not args.pdf:
            print("Error: --stl and --pdf required for Stage 2"); sys.exit(1)
        run_stage2(args.stl, args.pdf)

    elif args.stage == 3:
        results = run_stage3()

    # ── Self-test mode ────────────────────────────────────────────────────────
    elif args.self_test:
        if not args.stl:
            print("Error: --stl required for self-test"); sys.exit(1)
        ok1 = run_stage1(args.stl)
        if ok1:
            results = run_self_test(args.stl)

    # ── Full pipeline mode ────────────────────────────────────────────────────
    else:
        if not args.stl:
            print("Error: --stl required"); sys.exit(1)

        ok1 = run_stage1(args.stl)
        if not ok1:
            print("[Pipeline] Stage 1 failed — aborting"); sys.exit(1)

        if args.pdf:
            ok2 = run_stage2(args.stl, args.pdf)
            if not ok2:
                print("[Pipeline] Stage 2 failed — aborting"); sys.exit(1)
        else:
            print("\n[Pipeline] No --pdf provided — skipping Stage 2.")
            print("[Pipeline] Tip: use --self-test to run Stage 3 without a PDF.\n")
            sys.exit(0)

        results = run_stage3()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    if results:
        print_summary(results, elapsed)

    # ── Optional JSON output ──────────────────────────────────────────────────
    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Pipeline] Results saved → {args.output}")

    # Exit with non-zero code if any view failed (useful for CI)
    if results.get("overall") == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
