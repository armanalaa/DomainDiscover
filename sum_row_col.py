"""
csv_row_col_sums.py
--------------------
Run this from the DomainMiner root folder. Pass the name of a dataset
subfolder (e.g. "DellStore2") and it will look for CSV files in:

    <dataset_folder>/csv/

(NOT inside ccm_output/, which holds pipeline run artifacts, not the
actual source tables.)

For every CSV found, it prints:
  - the sum of each numeric column
  - the sum of each numeric row

At the end, it prints the TOTAL number of rows and TOTAL number of
columns across all tables found.

Usage (from DomainMiner root):
    python csv_row_col_sums.py DellStore2

Options:
    --subdir NAME       Name of the subfolder to look in (default: "csv")
    --recursive          Search subfolders too (off by default — use with
                          caution, will also pick up ccm_output/ contents
                          if pointed at the dataset root instead of csv/)
    --all-rows           Print every row sum instead of just the first 20
"""

import os
import sys
import argparse
from pathlib import Path

import pandas as pd


def print_file_sums(csv_path: Path, all_rows: bool) -> tuple:
    """Returns (n_rows, n_columns) for this file."""
    print("=" * 90)
    print(f"FILE: {csv_path}")
    print("=" * 90)

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  ERROR reading file: {exc}\n")
        return 0, 0

    n_rows = len(df)
    n_columns = len(df.columns)

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print(f"  Rows: {n_rows}   Columns: {n_columns}   (no numeric columns found)\n")
        return n_rows, n_columns

    print(f"  Rows: {n_rows}   Columns: {n_columns}   Numeric columns: {len(numeric_df.columns)}")

    print("\n  -- Column sums --")
    col_sums = numeric_df.sum(numeric_only=True)
    for col, total in col_sums.items():
        print(f"    {col:<30} {total:,.2f}")

    print("\n  -- Row sums --" if all_rows else "\n  -- Row sums (first 20 shown) --")
    row_sums = numeric_df.sum(axis=1, numeric_only=True)
    rows_to_show = row_sums if all_rows else row_sums.head(20)
    for idx, total in rows_to_show.items():
        print(f"    row {idx:<6} {total:,.2f}")
    if not all_rows and len(row_sums) > 20:
        print(f"    ... ({len(row_sums) - 20} more rows not shown; use --all-rows to see all)")

    print()
    return n_rows, n_columns


def find_csv_files(root_dir: str, recursive: bool) -> list:
    found = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(".csv"):
                    found.append(Path(dirpath) / fname)
    else:
        try:
            for fname in os.listdir(root_dir):
                full = Path(root_dir) / fname
                if full.is_file() and fname.lower().endswith(".csv"):
                    found.append(full)
        except Exception as exc:
            print(f"  Could not list directory: {exc}")
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(
        description="Print row/column sums for CSV tables in <dataset_folder>/csv/."
    )
    parser.add_argument(
        "dataset_folder",
        help="Name of the dataset folder (e.g. DellStore2), or a full path to it",
    )
    parser.add_argument(
        "--subdir",
        default="csv",
        help='Subfolder to look in, relative to dataset_folder (default: "csv")',
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders too (off by default)",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Print every row sum instead of just the first 20",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    candidate = Path(args.dataset_folder)

    if candidate.is_absolute() and candidate.exists():
        dataset_dir = candidate
    else:
        dataset_dir = cwd / args.dataset_folder

    target_dir = dataset_dir / args.subdir if args.subdir else dataset_dir

    if not target_dir.exists():
        print(f"ERROR: folder not found: {target_dir}")
        print("Make sure you're either:")
        print("  (a) running this script FROM the DomainMiner root folder, and")
        print("      passing just the dataset name (e.g. DellStore2), OR")
        print("  (b) passing the FULL path to the dataset folder directly.")
        print(f"(Looking for a '{args.subdir}' subfolder inside it — use --subdir to change this.)")
        sys.exit(1)

    csv_files = find_csv_files(str(target_dir), recursive=args.recursive)

    if not csv_files:
        print(f"No CSV files found under {target_dir}")
        print("\nFolders/files actually present there:")
        try:
            for item in sorted(target_dir.iterdir()):
                print(f"   {'[DIR] ' if item.is_dir() else '       '}{item.name}")
        except Exception as exc:
            print(f"  Could not list directory: {exc}")
        sys.exit(0)

    print(f"Found {len(csv_files)} CSV file(s) under {target_dir}\n")

    total_rows = 0
    total_columns = 0
    for csv_path in csv_files:
        n_rows, n_columns = print_file_sums(csv_path, args.all_rows)
        total_rows += n_rows
        total_columns += n_columns

    print("=" * 90)
    print("OVERALL TOTALS (across all tables)")
    print("=" * 90)
    print(f"  Total tables  : {len(csv_files)}")
    print(f"  Total rows    : {total_rows:,}")
    print(f"  Total columns : {total_columns:,}")


if __name__ == "__main__":
    main()