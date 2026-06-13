# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-06-12

First public, production-ready release of **MPASdiag**, a Python package for
MPAS model output analysis and visualization on unstructured meshes.

### Added
- **Data processing**: readers and processors for 2D and 3D MPAS output,
  with parallel (MPI / multiprocessing) batch processing support.
- **Remapping**: pluggable regridding via KDTree (nearest/linear) and
  ESMPy/xESMF (conservative) engines.
- **2D diagnostics**: precipitation analysis, surface fields, and wind
  diagnostics (speed, barbs, arrows, streamlines).
- **3D diagnostics**: vertical soundings with Skew-T Log-P plots and
  thermodynamic indices (CAPE, CIN, SRH, bulk shear), vertical cross-sections
  (pressure and height coordinates), and integrated vapor transport (IVT/IWV).
- **Visualization**: publication-quality Cartopy-based maps with configurable
  overlays and multi-layer composites.
- **Command-line interface** (`mpasdiag`) for single-time and batch workflows.
- Comprehensive test suite with coverage reporting and CI across
  Python 3.10–3.13.

[Unreleased]: https://github.com/mrixlam/MPASdiag/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/mrixlam/MPASdiag/releases/tag/v1.0.0
