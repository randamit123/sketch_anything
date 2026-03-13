#!/usr/bin/env python3
"""Inspect LIBERO HDF5 demo files and report BDDL resolution status.

Usage:
    # Inspect specific files
    conda run -n libero python -m sketch_anything.tools.inspect_hdf5 demo1.hdf5 [demo2.hdf5 ...]

    # Scan all HDF5 files in the LIBERO datasets directory
    conda run -n libero python -m sketch_anything.tools.inspect_hdf5 --all

For each file the tool reports:
  - Task instruction stored in the HDF5
  - BDDL filename stored in the HDF5
  - Whether a name correction from BDDL_NAME_MAP applies
  - The resolved absolute path (or NOT FOUND with diagnostic info)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Import the shared name map and resolver from run_pipeline.
# This guarantees inspect_hdf5 always reflects the same corrections.
def _import_pipeline_helpers():
    """Import BDDL_NAME_MAP and _resolve_bddl_path from run_pipeline."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sketch_anything.tools.run_pipeline import BDDL_NAME_MAP, _resolve_bddl_path
    return BDDL_NAME_MAP, _resolve_bddl_path


def inspect_file(hdf5_path: Path, bddl_name_map: dict, resolve_fn) -> bool:
    """Inspect one HDF5 file. Returns True if BDDL was found."""
    import h5py

    print(f"\n{hdf5_path.name}")

    try:
        with h5py.File(hdf5_path, "r") as f:
            env_args_raw = json.loads(f["data"].attrs.get("env_args", "{}"))
            problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
            task = problem_info.get("language_instruction", "(not stored)")
            bddl_attr = f["data"].attrs.get("bddl_file_name", None)
            if isinstance(bddl_attr, bytes):
                bddl_attr = bddl_attr.decode("utf-8")
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return False

    env_kwargs = env_args_raw.get("env_kwargs", env_args_raw)
    stored_bddl_path = env_kwargs.get("bddl_file_name") or env_args_raw.get("bddl_file") or bddl_attr or "(not found)"
    stored_basename = os.path.basename(stored_bddl_path)

    print(f"  task            : {task}")
    print(f"  stored bddl     : {stored_basename}", end="")

    corrected = bddl_name_map.get(stored_basename)
    if corrected:
        print(f"  [WRONG NAME — known mismatch]")
        print(f"  corrected to    : {corrected}")
        search_name = corrected
    else:
        print()
        search_name = stored_basename

    # Check if stored path exists as-is
    if os.path.exists(stored_bddl_path):
        print(f"  resolved path   : {stored_bddl_path} ✓ FOUND (exact path)")
        return True

    # Try resolution
    resolved = resolve_fn(search_name)
    if resolved:
        print(f"  resolved path   : {resolved} ✓ FOUND")
        return True

    # Not found
    print(f"  resolved path   : NOT FOUND")

    # Diagnostic hints
    try:
        from libero.libero import get_libero_path
        bddl_root = get_libero_path("bddl_files")
        if not Path(bddl_root).exists():
            print(f"  LIBERO config path '{bddl_root}' does not exist")
            print(f"    → run: conda run -n libero python -m sketch_anything.tools.setup_libero_config")
        else:
            print(f"  LIBERO config path: {bddl_root} (exists but file not found inside)")
    except Exception:
        print(f"  LIBERO config unavailable")

    if corrected is None:
        print(f"  No entry in BDDL_NAME_MAP for '{stored_basename}'")
        print(f"    → Add a mapping to BDDL_NAME_MAP in sketch_anything/tools/run_pipeline.py")
        print(f"    → Or use --bddl-file /path/to/correct.bddl when running run_pipeline")

    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect LIBERO HDF5 files for BDDL resolution status.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="HDF5",
        help="HDF5 file(s) to inspect.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan all HDF5 files in LIBERO/libero/datasets/ recursively.",
    )
    args = parser.parse_args()

    if not args.files and not args.all:
        parser.print_help()
        sys.exit(1)

    bddl_name_map, resolve_fn = _import_pipeline_helpers()

    hdf5_files: list[Path] = []
    if args.all:
        datasets_root = Path(__file__).resolve().parents[2] / "LIBERO" / "libero" / "datasets"
        if not datasets_root.exists():
            print(f"Datasets directory not found: {datasets_root}")
            sys.exit(1)
        hdf5_files = sorted(datasets_root.rglob("*.hdf5"))
        print(f"Scanning {len(hdf5_files)} HDF5 files in {datasets_root}")
    else:
        hdf5_files = [Path(p) for p in args.files]

    found = 0
    not_found = 0
    for path in hdf5_files:
        ok = inspect_file(path, bddl_name_map, resolve_fn)
        if ok:
            found += 1
        else:
            not_found += 1

    total = found + not_found
    print(f"\n{'─'*60}")
    print(f"Summary: {found}/{total} BDDL files resolved successfully")
    if not_found:
        print(f"         {not_found} file(s) could not be resolved — see details above")
        sys.exit(1)


if __name__ == "__main__":
    main()
