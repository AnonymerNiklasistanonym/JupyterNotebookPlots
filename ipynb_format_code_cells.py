import argparse
import black
import json
import re

from pathlib import Path


def ipynb_format_code_cells(path: Path, verbose: bool) -> tuple[Path, int]:
    json_data = {}
    formatted_cells_count = 0
    with open(path, "r") as f:
        data = json.load(f)
        if "cells" not in data:
            raise RuntimeError("The attribute 'cells' was not found!")
        for cell in data["cells"]:
            if "cell_type" not in cell:
                raise RuntimeError("The attribute 'cell_type' was not found!")
            if cell["cell_type"] != "code":
                continue
            if "source" not in cell:
                raise RuntimeError("The attribute 'source' was not found!")
            cell_str = "".join(cell["source"])
            cell_str_formatted = black.format_str(cell_str, mode=black.Mode()).rstrip()
            if cell_str != cell_str_formatted:
                if verbose:
                    print(f"Black formatted {cell_str!r} to {cell_str_formatted!r}")
                formatted_cells_count += 1
                cell["source"] = re.split(r"(?<=[\n])", cell_str_formatted)
                if verbose:
                    print(f"Updated {cell_str!r} with {''.join(cell['source'])!r}")
    if formatted_cells_count > 0:
        with open(path, "w") as f:
            f.write(json.dumps(data, indent=" ", ensure_ascii=False) + "\n")
    return path, formatted_cells_count


def main(paths: list[Path], verbose: bool):
    results: list[tuple[Path, int]] = []
    for path in paths:
        results.append(ipynb_format_code_cells(path, verbose))
    formatted_files = 0
    for path, formatted_cells_count in results:
        if formatted_cells_count > 0:
            formatted_files += 1
            print(
                f"reformatted {path} ({formatted_cells_count} cell{'s' if formatted_cells_count > 0 else ''})"
            )
    unchanged_files = len(paths) - formatted_files
    if formatted_files > 0:
        print(
            f"{formatted_files} file{'s' if formatted_files > 1 else ''} reformatted{', ' if unchanged_files > 0 else '.'}",
            end="" if unchanged_files > 0 else None,
        )
    if unchanged_files > 0:
        print(
            f"{unchanged_files} file{'s' if unchanged_files > 1 else ''} left unchanged."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format the python code in a jupyter notebook (*.ipynb)"
    )
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1")
    parser.add_argument(
        "notebook",
        type=Path,
        nargs="+",
        help="file path to the jupyter notebook (*.ipynb)",
    )
    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
    )
    args = parser.parse_args()
    main(args.notebook, args.verbose)
