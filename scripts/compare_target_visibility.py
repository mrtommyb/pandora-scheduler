#!/usr/bin/env python3
"""Compare one target visibility parquet between two target roots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare visibility parquet for a target between two roots, e.g. "
            "data/targets vs data_earth_keepout_96/targets."
        )
    )
    parser.add_argument("--target", required=True, help="Target/planet name, e.g. WASP-107b")
    parser.add_argument(
        "--root-a",
        type=Path,
        default=Path("output_directory/data/targets"),
        help="First targets root (default: output_directory/data/targets)",
    )
    parser.add_argument(
        "--root-b",
        type=Path,
        default=Path("output_directory/data_earth_keepout_96/targets"),
        help="Second targets root (default: output_directory/data_earth_keepout_96/targets)",
    )
    return parser.parse_args()


def find_target_file(root: Path, target: str) -> Path:
    pattern = f"Visibility for {target}.parquet"
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' found under {root}")
    if len(matches) > 1:
        # Prefer exact planet directory match .../<target>/Visibility for <target>.parquet
        exact = [m for m in matches if m.parent.name == target]
        if len(exact) == 1:
            return exact[0]
        raise ValueError(f"Multiple matches for target '{target}' under {root}: {matches}")
    return matches[0]


def pick_join_keys(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    if "Time(MJD_UTC)" in a.columns and "Time(MJD_UTC)" in b.columns:
        return ["Time(MJD_UTC)"]
    if {"Transit_Start", "Transit_Stop"}.issubset(a.columns) and {
        "Transit_Start",
        "Transit_Stop",
    }.issubset(b.columns):
        return ["Transit_Start", "Transit_Stop"]
    return []


def compare_numeric_columns(merged: pd.DataFrame, cols: list[str]) -> None:
    if not cols:
        print("No comparable numeric columns.")
        return
    print("Numeric differences (on matched rows):")
    any_reported = False
    for col in cols:
        a_col = f"{col}__a"
        b_col = f"{col}__b"
        if a_col not in merged.columns or b_col not in merged.columns:
            continue
        a_num = pd.to_numeric(merged[a_col], errors="coerce")
        b_num = pd.to_numeric(merged[b_col], errors="coerce")
        mask = a_num.notna() & b_num.notna()
        if not mask.any():
            continue
        diff = (b_num[mask] - a_num[mask]).abs()
        any_reported = True
        print(
            f"  {col}: matched={int(mask.sum())}, "
            f"max_abs_diff={diff.max():.6g}, mean_abs_diff={diff.mean():.6g}"
        )
    if not any_reported:
        print("  (No numeric overlap to compare)")


def main() -> int:
    args = parse_args()
    root_a = args.root_a.resolve()
    root_b = args.root_b.resolve()

    file_a = find_target_file(root_a, args.target)
    file_b = find_target_file(root_b, args.target)

    df_a = pd.read_parquet(file_a)
    df_b = pd.read_parquet(file_b)

    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    common_cols = sorted(cols_a & cols_b)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)

    print(f"Target: {args.target}")
    print(f"A: {file_a}")
    print(f"B: {file_b}")
    print()
    print(f"Rows: A={len(df_a)}, B={len(df_b)}")
    print(f"Columns in common ({len(common_cols)}): {common_cols}")
    if only_a:
        print(f"Columns only in A ({len(only_a)}): {only_a}")
    if only_b:
        print(f"Columns only in B ({len(only_b)}): {only_b}")

    join_keys = pick_join_keys(df_a, df_b)
    if not join_keys:
        print("\nNo natural join key found. Falling back to index-based comparison.")
        n = min(len(df_a), len(df_b))
        merged = pd.DataFrame()
        for col in common_cols:
            merged[f"{col}__a"] = df_a[col].iloc[:n].to_numpy()
            merged[f"{col}__b"] = df_b[col].iloc[:n].to_numpy()
        print(f"Matched rows by index: {n}")
    else:
        merged = df_a.merge(
            df_b,
            on=join_keys,
            how="inner",
            suffixes=("__a", "__b"),
        )
        print(f"\nJoin keys: {join_keys}")
        print(f"Matched rows on keys: {len(merged)}")

    comparable_numeric = []
    for col in common_cols:
        if col in join_keys:
            continue
        if pd.api.types.is_numeric_dtype(df_a[col]) or pd.api.types.is_numeric_dtype(df_b[col]):
            comparable_numeric.append(col)

    print()
    compare_numeric_columns(merged, comparable_numeric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

