#!/usr/bin/env python3
"""Render a simple Gantt-style figure from a Pandora science calendar XML."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from astropy.time import Time
from lxml import etree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Gantt-style timeline figure from a Pandora XML calendar."
    )
    parser.add_argument(
        "xml",
        type=Path,
        help="Path to Pandora_science_calendar.xml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output PNG path. If omitted, the figure is shown interactively.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Pandora Calendar",
        help="Plot title",
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
            priority = params.findtext(priority_tag, default="0", namespaces=ns)
            timing = params.find(timing_tag, ns)
            if timing is None:
                continue
            start_text = timing.findtext(start_tag, default="", namespaces=ns)
            stop_text = timing.findtext(stop_tag, default="", namespaces=ns)
            if not start_text or not stop_text:
                continue

            rows.append(
                {
                    "visitid": visit_id,
                    "seqid": seq_id,
                    "target": target,
                    "priority": priority,
                    "start_jd": Time(start_text).jd,
                    "stop_jd": Time(stop_text).jd,
                    "start_dt": pd.to_datetime(start_text),
                    "stop_dt": pd.to_datetime(stop_text),
                    "start_text": start_text,
                    "stop_text": stop_text,
                }
            )

    if not rows:
        raise ValueError(f"No observation sequences found in XML: {xml_path}")

    return pd.DataFrame(rows)


def _target_sort_key(target: str) -> tuple[int, str]:
    name = str(target)
    if "Free Time" in name:
        return (4, name)
    if name.startswith("STD") or name.startswith("BD") or name.startswith("GD"):
        return (3, name)
    if name.endswith(("b", "c", "d", "e", "f")):
        return (1, name)
    return (2, name)


def _bar_color(target: str, priority: str) -> str:
    if "Free Time" in target:
        return "#cbd5e1"
    if target.startswith("STD") or target.startswith("BD") or target.startswith("GD"):
        return "#94a3b8"
    if priority == "2":
        return "#d97706"
    if priority == "1":
        return "#0f766e"
    return "#1f2937"


def build_figure(df: pd.DataFrame, title: str) -> plt.Figure:
    fig_height = max(6, 0.38 * df["target"].nunique() + 1.5)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    targets = sorted(df["target"].unique(), key=_target_sort_key)
    target_to_y = {target: idx for idx, target in enumerate(targets)}

    for _, row in df.iterrows():
        y = target_to_y[str(row["target"])]
        start = mdates.date2num(row["start_dt"].to_pydatetime())
        stop = mdates.date2num(row["stop_dt"].to_pydatetime())
        width = stop - start
        color = _bar_color(str(row["target"]), str(row["priority"]))
        ax.broken_barh(
            [(start, width)],
            (y - 0.32, 0.64),
            facecolors=color,
            edgecolors="none",
            alpha=0.95,
        )

    ax.set(
        xlabel="UTC Time",
        yticks=np.arange(len(targets)),
        yticklabels=targets,
        title=title,
    )
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.grid(axis="x", color="#cbd5e1", alpha=0.6, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return fig


def main() -> int:
    args = parse_args()
    df = parse_calendar(args.xml)
    fig = build_figure(df, args.title)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {args.out}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
