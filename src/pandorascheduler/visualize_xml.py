#!/usr/bin/env python3
"""Create windowed priority Gantt plots from a Pandora science calendar XML."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from lxml import etree
from matplotlib.patches import Patch, Rectangle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create windowed priority Gantt plots from a Pandora XML calendar."
    )
    parser.add_argument("xml", type=Path, help="Path to Pandora_science_calendar.xml")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output_directory/confirm_visibility"),
        help="Directory for saved PNGs",
    )
    parser.add_argument(
        "--window-days",
        type=float,
        default=4.0,
        help="Width of each time window in days",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Schedule by Priority",
        help="Base plot title",
    )
    parser.add_argument(
        "--show-sequence-labels",
        action="store_true",
        help="Draw sequence ID / priority labels inside sufficiently long bars",
    )
    return parser.parse_args()


def _namespace(root: etree._Element) -> dict[str, str]:
    uri = root.nsmap.get(None)
    return {"ns": uri} if uri else {}


def parse_calendar(xml_path: Path) -> pd.DataFrame:
    parser = etree.XMLParser(load_dtd=True, no_network=False)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()
    ns = _namespace(root)

    visit_tag = "ns:Visit" if ns else "Visit"
    seq_tag = "ns:Observation_Sequence" if ns else "Observation_Sequence"
    id_tag = "ns:ID" if ns else "ID"
    params_tag = "ns:Observational_Parameters" if ns else "Observational_Parameters"
    target_tag = "ns:Target" if ns else "Target"
    priority_tag = "ns:Priority" if ns else "Priority"
    timing_tag = "ns:Timing" if ns else "Timing"
    start_tag = "ns:Start" if ns else "Start"
    stop_tag = "ns:Stop" if ns else "Stop"

    rows: list[dict[str, object]] = []
    for visit in root.findall(visit_tag, ns):
        visit_id = visit.findtext(id_tag, default="", namespaces=ns)
        for seq in visit.findall(seq_tag, ns):
            seq_id = seq.findtext(id_tag, default="", namespaces=ns)
            params = seq.find(params_tag, ns)
            if params is None:
                continue
            target = params.findtext(target_tag, default="Unknown", namespaces=ns)
            priority_text = params.findtext(priority_tag, default="0", namespaces=ns)
            timing = params.find(timing_tag, ns)
            if timing is None:
                continue
            start_text = timing.findtext(start_tag, default="", namespaces=ns)
            stop_text = timing.findtext(stop_tag, default="", namespaces=ns)
            if not start_text or not stop_text:
                continue

            start_dt = pd.to_datetime(start_text, utc=True).tz_convert(None)
            stop_dt = pd.to_datetime(stop_text, utc=True).tz_convert(None)
            rows.append(
                {
                    "visitid": str(visit_id),
                    "seqid": str(seq_id),
                    "target": str(target),
                    "priority": int(float(priority_text)),
                    "start_dt": start_dt,
                    "stop_dt": stop_dt,
                    "duration_hours": (stop_dt - start_dt).total_seconds() / 3600.0,
                    "start_jd": Time(start_text).jd,
                    "stop_jd": Time(stop_text).jd,
                }
            )

    if not rows:
        raise ValueError(f"No observation sequences found in XML: {xml_path}")
    return pd.DataFrame(rows).sort_values(["start_dt", "visitid", "seqid"]).reset_index(drop=True)


def _priority_color(priority: int) -> str | tuple[float, float, float]:
    color_map = {
        0: "lightgray",
        1: "crimson",
        2: "darkorange",
        3: "gold",
        4: "yellow",
        5: "yellowgreen",
        6: "forestgreen",
        7: "deepskyblue",
        8: "mediumpurple",
    }
    if priority in color_map:
        return color_map[priority]
    gray_value = max(0.3, min(0.8, 0.8 - (priority - 8) * 0.05))
    return (gray_value, gray_value, gray_value)


def _format_time_axis(ax: plt.Axes, df: pd.DataFrame) -> None:
    min_time = df["start_dt"].min()
    max_time = df["stop_dt"].max()
    time_span = max_time - min_time
    padding = time_span * 0.001 if time_span.total_seconds() > 0 else pd.Timedelta(minutes=5)
    ax.set_xlim(min_time - padding, max_time + padding)

    total_hours = time_span.total_seconds() / 3600.0
    if total_hours < 6:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    elif total_hours < 48:
        interval = max(2, int(total_hours / 10))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    else:
        interval = max(1, int(total_hours / 24 / 10))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def build_priority_figure(
    df: pd.DataFrame,
    title: str,
    show_sequence_labels: bool = False,
    figsize: tuple[float, float] = (20, 10),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_dpi(100)

    y_pos = 0.0
    y_positions: list[float] = []
    y_labels: list[str] = []
    visit_boundaries: list[tuple[float, float]] = []
    visit_info: list[tuple[str, float, str]] = []

    for visit_id, visit_df in df.groupby("visitid", sort=False):
        visit_start_y = y_pos
        target_groups = []
        for target, target_df in visit_df.groupby("target", sort=False):
            target_df = target_df.sort_values("start_dt")
            target_groups.append((target, target_df))
        target_groups.sort(key=lambda item: item[1]["start_dt"].iloc[0])

        for target, target_df in target_groups:
            for _, row in target_df.iterrows():
                start_num = float(mdates.date2num(row["start_dt"].to_pydatetime()))
                width = row["duration_hours"] / 24.0
                rect = Rectangle(
                    (start_num, y_pos - 0.35),
                    width,
                    0.7,
                    facecolor=_priority_color(int(row["priority"])),
                    edgecolor="none",
                    linewidth=0,
                    alpha=1.0,
                )
                ax.add_patch(rect)

                if show_sequence_labels and row["duration_hours"] > 0.5:
                    mid_time = start_num + width / 2.0
                    ax.text(
                        mid_time,
                        y_pos,
                        f"{row['seqid']}\nP{int(row['priority'])}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        fontweight="bold",
                        bbox={
                            "boxstyle": "round,pad=0.1",
                            "facecolor": "white",
                            "alpha": 0.9,
                        },
                    )

            priorities = sorted(int(p) for p in target_df["priority"].unique())
            y_positions.append(y_pos)
            y_labels.append(
                f"  {target} ({len(target_df)} seq, P{','.join(str(p) for p in priorities)})"
            )
            y_pos += 1.0

        visit_end_y = y_pos - 1.0
        visit_boundaries.append((visit_start_y, visit_end_y))
        priority_counts = (
            visit_df["priority"].astype(int).value_counts().sort_index().to_dict()
        )
        summary = ", ".join(f"P{p}:{c}" for p, c in priority_counts.items())
        visit_info.append((str(visit_id), (visit_start_y + visit_end_y) / 2.0, summary))

        if visit_id != df["visitid"].iloc[-1]:
            ax.axhline(y=y_pos - 0.5, color="black", linewidth=2, alpha=0.7)
        y_pos += 0.2

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylim(-0.5, max(y_pos - 0.7, 0.5))
    ax.grid(True, axis="x", alpha=0.3)
    ax.grid(False, axis="y")
    ax.set_ylabel("Targets (grouped by visit)", fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)

    norm = max(y_pos - 0.7, 1.0)
    for idx, (visit_id, mid_y, summary) in enumerate(visit_info):
        x_offset = 1.05 + (idx % 2) * 0.12
        ax.text(
            x_offset,
            mid_y / norm,
            f"Visit {visit_id}\n({int((df['visitid'] == visit_id).sum())} seq)\n{summary}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7},
        )

    _format_time_axis(ax, df)
    legend_priorities = sorted(int(p) for p in df["priority"].unique())
    handles = [
        Patch(facecolor=_priority_color(priority), alpha=1.0, label=f"Priority {priority}")
        for priority in legend_priorities
    ]
    if handles:
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    fig.set_constrained_layout(True)
    return fig


def _windowed_frames(df: pd.DataFrame, window_days: float) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    start_date = df["start_dt"].min()
    end_date = df["stop_dt"].max()
    current = start_date
    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []

    while current < end_date:
        window_end = min(current + pd.Timedelta(days=window_days), end_date)
        mask = (df["start_dt"] < window_end) & (df["stop_dt"] > current)
        window_df = df.loc[mask].copy()
        if not window_df.empty:
            windows.append((current, window_end, window_df))
        current = window_end
    return windows


def main() -> int:
    args = parse_args()
    df = parse_calendar(args.xml)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    windows = _windowed_frames(df, args.window_days)
    for index, (start, end, window_df) in enumerate(windows):
        title = (
            f"{args.title} - Window {index + 1} "
            f"({start.strftime('%m/%d')} to {end.strftime('%m/%d')})"
        )
        fig = build_priority_figure(
            window_df,
            title=title,
            show_sequence_labels=args.show_sequence_labels,
            figsize=(20, 10),
        )
        out_path = args.out_dir / f"gantt_plot_window_{index}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
