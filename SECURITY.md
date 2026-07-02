# Security Policy

## Supported versions

Security updates are provided for the most recent release line of MPASdiag.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Security considerations for users

MPASdiag is a **local command-line tool and Python library**. It runs with the
privileges of the user who invokes it and is not a network service. It is,
however, hardened so that it can be run safely against **inputs authored by
someone other than the operator** — a shared or downloaded `config.yaml`, a
NetCDF/HDF5 file from an untrusted source, or invocation by an automated/LLM
agent. Please keep the following trust boundaries in mind:

### Only process data you trust

MPASdiag reads NetCDF/HDF5 model output, grid files, and pre-computed remapping
weight caches. Parsing of these binary files is delegated to the underlying
`netCDF4`, `h5netcdf`/`HDF5`, and `xarray` libraries. As with any scientific data
tool, **opening a maliciously crafted or corrupt file can crash the process or,
through a vulnerability in those underlying libraries, be unsafe.** Only process
files obtained from sources you trust, and keep your scientific stack updated.

As a defense-in-depth measure, MPASdiag rejects inputs whose declared dimensions
exceed generous safety limits **before allocating large arrays or building a
regridder**, to avoid out-of-memory crashes and CPU-exhaustion on malformed
files. These limits are enforced uniformly on the raw data/grid load, the
in-memory cache, the live regridder build (source cells and per-cell vertices),
and the cached-weights load. If you legitimately work with very large grids, you
can raise them via environment variables: `MPASDIAG_MAX_SOURCE_CELLS`,
`MPASDIAG_MAX_TARGET_POINTS`, `MPASDIAG_MAX_WEIGHTS_NNZ`, `MPASDIAG_MAX_NUM_POINTS`,
`MPASDIAG_MAX_CELL_VERTICES`, `MPASDIAG_MAX_WORKERS`, and `MPASDIAG_MAX_INPUT_FILES`.

Untrusted text read from files (variable names, `long_name`/`units` attributes)
is sanitized before it is embedded in log/error messages or rendered as plot
text, so it cannot forge log lines, inject content into the output an automated
agent reads, or abort rendering via a malformed matplotlib mathtext expression.

The remapping **weights cache directory** (`weights_dir`) is validated for
internal consistency and confined like other paths; a tampered cache file is
rejected before use, but you should still point it only at a location you control.

### Output, log, config, grid, and data paths

All operator- or config-supplied filesystem paths — `--output-dir`, `--output`,
`--log-file`, `--config`, `--grid-file`, `--data-dir`, and the weights cache — are
**confined to the working directory** (or an explicit base directory). A path
that resolves outside that directory, whether via `..` traversal or an absolute
path, is refused. Untrusted filename components (e.g. a variable name taken from a
data file) are sanitized so they cannot inject a path separator or escape the
output directory. Configuration files must be `.yaml`/`.yml`.

Because scientific grid and data files legitimately live outside the working
directory (e.g. `/scratch`, `/glade`, project mounts), pass **`--base-dir <dir>`**
to move the containment boundary to a directory you trust; all input and output
paths must then resolve within it. This lets you work with out-of-tree data while
still refusing an untrusted config's attempt to read or write arbitrary locations.

Configuration loaded from a file is fully re-validated after any command-line
overrides are merged, and numeric parameters (DPI, figure size, worker count,
time/level indices, etc.) are range-checked, so a hostile or careless config
cannot slip an invalid or abusive value past the validators.

### Parallel (MPI) execution

When run under MPI, MPASdiag assumes that **all ranks in `MPI_COMM_WORLD` belong
to the same trusted job and user**. Inter-rank messages use Python `pickle`
(via `mpi4py`), so a compromised or malicious co-rank could deliver arbitrary
objects to its peers — do not launch MPASdiag across a communicator that spans a
trust boundary. As defense-in-depth, sizes carried in broadcast metadata are
bounded before peer ranks allocate buffers. The in-memory data cache is pickled
only within a single machine's process pool to seed multiprocessing workers; it
is never persisted, and must never be rehydrated from an on-disk or untrusted
pickle.

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
