# Image-Comparison-Autotest
Automated 3D PDF export validation pipeline · STL rendering · C2W matrix reconstruction · SSIM · IoU · Python

## Problem
Manual visual inspection of 3D PDF TDP exports is
unreliable at scale — subtle camera shifts and geometry
drops go undetected.

## Solution (3-stage pipeline)
Stage 1 → Stage 2 → Stage 3 diagram

## Architecture
Table: each stage, what it does, key decision

## Key design decisions
- Why C2W matrix extraction over poster stream extraction
- Why SSIM + IoU together, not just one
- Why morphological dilation (9x9)
- Why orthographic projection not perspective
- Why headless matplotlib (CI deployable)

## Results
Table: per-view SSIM, IoU, Pass/Fail

## How to run
pip install + commands

## Future scope
CI/CD integration, ML-based defect classification
