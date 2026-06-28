# Security Policy

## Supported versions

Security updates are provided for the most recent release line of MPASdiag.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Security considerations for users

MPASdiag is a **local command-line tool and Python library**. It runs with the
privileges of the user who invokes it and is not a network service. Please keep
the following trust boundaries in mind:

### Only process data you trust

MPASdiag reads NetCDF/HDF5 model output, grid files, and pre-computed remapping
weight caches. Parsing of these binary files is delegated to the underlying
`netCDF4`, `h5netcdf`/`HDF5`, and `xarray` libraries. As with any scientific data
tool, **opening a maliciously crafted or corrupt file can crash the process or,
through a vulnerability in those underlying libraries, be unsafe.** Only process
files obtained from sources you trust, and keep your scientific stack updated.

As a defense-in-depth measure, MPASdiag rejects inputs whose declared dimensions
exceed generous safety limits before allocating large arrays, to avoid
out-of-memory crashes on malformed files. If you legitimately work with very
large grids, you can raise these limits via environment variables:
`MPASDIAG_MAX_SOURCE_CELLS`, `MPASDIAG_MAX_TARGET_POINTS`,
`MPASDIAG_MAX_WEIGHTS_NNZ`, and `MPASDIAG_MAX_NUM_POINTS`.

The remapping **weights cache directory** (`weights_dir`) is treated as trusted
input: only point it at a location you control. A tampered cache file is
validated for internal consistency before use, but should not be shared across
trust boundaries.

### Output, log, and config paths

Output directories, log files (`--log-file`), configuration files (`--config`),
and weights paths are taken from the operator and are honored as given — this is
intended behavior for a local tool that writes to your own filesystem.
Configuration paths are confined to the working directory (or an explicit
`base_dir`) and must be `.yaml`/`.yml` files; this guards against accidental
path traversal, not against a user who deliberately targets their own files.

### Reproducible installation

Lower bounds in `pyproject.toml`/`requirements.txt` are set above known-vulnerable
releases, but `pip` will otherwise resolve to the latest compatible versions. For
a fully reproducible environment, install into the provided conda `environment.yml`,
or generate a pinned constraints file from a known-good environment
(`pip freeze > constraints.txt`) and install with `pip install mpasdiag -c constraints.txt`.

## Reporting a vulnerability

We take the security of MPASdiag seriously. If you discover a security
vulnerability, please report it **privately** so we can address it before it is
publicly disclosed.

Please do **not** open a public GitHub issue for security problems.

Instead, use one of the following channels:

- **GitHub private vulnerability reporting** (preferred): open a report from the
  repository's [Security advisories](https://github.com/mrixlam/MPASdiag/security/advisories/new)
  page.
- **Email**: contact the maintainer at **mrislam@ucar.edu** with the details.

When reporting, please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a proof-of-concept, if available.
- The affected version(s) and your environment details.

## What to expect

- We will acknowledge your report within **5 business days**.
- We will investigate and keep you informed of our progress.
- Once a fix is available, we will coordinate a release and credit you for the
  discovery (unless you prefer to remain anonymous).

Thank you for helping keep MPASdiag and its users safe.
