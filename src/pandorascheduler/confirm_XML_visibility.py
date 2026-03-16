#!/usr/bin/env python3
"""Create per-visit visibility confirmation plots for a Pandora XML calendar."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-visit XML visibility confirmation plots."
    )
    parser.add_argument(
        "xml",
        type=Path,
        help="Path to Pandora_science_calendar.xml",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Run data directory containing targets/ and aux_targets/ visibility parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated figures (default: <data-dir>/confirm_visibility)",
    )
    return parser.parse_args()


def _resolve_visibility_df(data_dir: Path, target: str) -> pd.DataFrame:
    targets_root = data_dir / "targets"
    aux_root = data_dir / "aux_targets"

    star_name = target[:-1] if target.endswith(("b", "c", "d", "e", "f")) else target
    candidate_paths = [
        targets_root / star_name / f"Visibility for {star_name}.parquet",
        aux_root / target / f"Visibility for {target}.parquet",
    ]

    for path in candidate_paths:
        if path.exists():
            return pd.read_parquet(path, columns=["Time(MJD_UTC)", "Visible"])

    raise FileNotFoundError(f"No visibility parquet found for target {target}")


def check_visibility(
    data_dir: Path,
    target: str,
    start_time: str,
    stop_time: str,
) -> tuple[list[bool], list[datetime]]:
    start_mjd = Time(start_time, format="isot", scale="utc").mjd
    stop_mjd = Time(stop_time, format="isot", scale="utc").mjd

    v_data = _resolve_visibility_df(data_dir, target)
    mask = (v_data["Time(MJD_UTC)"] >= start_mjd) & (v_data["Time(MJD_UTC)"] <= stop_mjd)
    period_data = v_data.loc[mask]

    times = Time(
        period_data["Time(MJD_UTC)"].to_numpy(dtype=float),
        format="mjd",
        scale="utc",
    ).to_datetime()
    times = [
        (t.replace(tzinfo=None) if getattr(t, "tzinfo", None) is not None else t)
        for t in times
    ]
    visibility = period_data["Visible"].to_numpy(dtype=float) >= 0.5
    return visibility.tolist(), times


def create_visit_figure(
    visit_data: list[dict[str, object]],
    visit_id: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    unique_targets = list(dict.fromkeys(str(d["target"]) for d in visit_data))
    target_to_y = {target: i for i, target in enumerate(unique_targets)}

    for data in visit_data:
        target = str(data["target"])
        times = np.asarray(data["times"], dtype=object)
        visible = np.asarray(data["visibility"], dtype=bool)
        y_value = target_to_y[target]
        priority = str(data["priority"])

        if priority == "2":
            color = "orange"
        elif priority == "1":
            color = "red"
        else:
            color = "gray"

        if len(times) > 0:
            ax.plot(times, np.full(len(times), y_value), color=color, linewidth=1)

        if len(times) > 0 and not visible.all():
            for t in times[~visible]:
                ax.axvline(
                    t,
                    ymin=y_value / max(len(target_to_y), 1),
                    ymax=(y_value + 1) / max(len(target_to_y), 1),
                    color="red",
                    linewidth=1,
                    alpha=1,
                )

    ax.set_yticks(range(len(unique_targets)))
    ax.set_yticklabels(unique_targets)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()
    plt.title(f"Target Visibility During Visit {visit_id}")
    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.tight_layout()

    output_file = output_dir / f"visit_{visit_id}_visibility.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    xml_path = args.xml
    data_dir = args.data_dir
    output_dir = args.output_dir or (data_dir / "confirm_visibility")
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    namespace = {"ns": "/pandora/calendar/"}

    for visit in tqdm(root.findall("ns:Visit", namespace), desc="Visits"):
        visit_id_elem = visit.find("ns:ID", namespace)
        if visit_id_elem is None or visit_id_elem.text is None:
            continue
        visit_id = visit_id_elem.text
        visit_data: list[dict[str, object]] = []

        for obs_seq in visit.findall("ns:Observation_Sequence", namespace):
            target_elem = obs_seq.find("./ns:Observational_Parameters/ns:Target", namespace)
            target = target_elem.text if target_elem is not None and target_elem.text else "No target"

            priority_elem = obs_seq.find("./ns:Observational_Parameters/ns:Priority", namespace)
            priority = priority_elem.text if priority_elem is not None and priority_elem.text else "0"

            start_elem = obs_seq.find("./ns:Observational_Parameters/ns:Timing/ns:Start", namespace)
            stop_elem = obs_seq.find("./ns:Observational_Parameters/ns:Timing/ns:Stop", namespace)
            if start_elem is None or stop_elem is None:
                print(f"Warning: Missing timing data for sequence in Visit {visit_id}")
                continue

            start_time = start_elem.text
            stop_time = stop_elem.text
            if not start_time or not stop_time:
                continue

            if target != "No target":
                try:
                    visibility, times = check_visibility(data_dir, target, start_time, stop_time)
                except Exception as exc:
                    print(f"Error checking visibility for target {target} in Visit {visit_id}: {exc}")
                    visibility, times = [], []
            else:
                visibility, times = [], []

            visit_data.append(
                {
                    "target": target,
                    "priority": priority,
                    "start": datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ"),
                    "stop": datetime.strptime(stop_time, "%Y-%m-%dT%H:%M:%SZ"),
                    "visibility": visibility,
                    "times": times,
                }
            )

        create_visit_figure(visit_data, visit_id, output_dir)

    print(f"Saved figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
