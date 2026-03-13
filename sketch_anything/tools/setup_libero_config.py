#!/usr/bin/env python3
"""Write a correct ~/.libero/config.yaml pointing to this project's LIBERO submodule.

Run once after cloning or moving the repository to fix LIBERO path resolution:

    conda run -n libero python -m sketch_anything.tools.setup_libero_config

The script auto-detects the project root, validates that each expected directory
exists, then writes (or overwrites) ~/.libero/config.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    libero_root = project_root / "LIBERO" / "libero" / "libero"
    datasets_dir = project_root / "LIBERO" / "libero" / "datasets"

    paths = {
        "assets":         libero_root / "assets",
        "bddl_files":     libero_root / "bddl_files",
        "benchmark_root": libero_root,
        "datasets":       datasets_dir,
        "init_states":    libero_root / "init_files",
    }

    print("Validating LIBERO paths...")
    all_ok = True
    for key, path in paths.items():
        if path.exists():
            print(f"  ✓  {key}: {path}")
        else:
            print(f"  ✗  {key}: {path}  [NOT FOUND]")
            all_ok = False

    if not all_ok:
        print(
            "\nSome paths are missing. Is the LIBERO submodule initialized?\n"
            "  git submodule update --init"
        )
        sys.exit(1)

    config_dir = Path.home() / ".libero"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.yaml"

    lines = [f"{key}: {path}\n" for key, path in paths.items()]
    config_path.write_text("".join(lines))

    print(f"\nWrote {config_path}")
    print("LIBERO path resolution is now configured correctly.")


if __name__ == "__main__":
    main()
