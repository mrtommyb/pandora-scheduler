#!/usr/bin/env python3
"""Analyze gaps between observation sequences in a Pandora science calendar XML."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze idle gaps between consecutive XML observation sequences."
    )
    parser.add_argument(
        "xml",
        nargs="?",
        type=Path,
        help="Path to Pandora_science_calendar.xml. If omitted, use the newest output_*/Pandora_science_calendar.xml.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional output path for the gap CSV (default: <xml_dir>/schedule_gaps.csv).",
    )
    parser.add_argument(
        "--hist-out",
        type=Path,
        help="Optional output path for a histogram PNG of gap sizes in hours.",
    )
    parser.add_argument(
        "--significant-hours",
        type=float,
        default=1.0,
        help="Threshold in hours for the detailed significant-gap section (default: 1.0).",
    )
    return parser.parse_args()


def parse_time(time_str: str) -> datetime:
    """Parse a UTC time string from the XML."""
    return datetime.fromisoformat(time_str.replace("Z", "+00:00"))


def find_default_xml(repo_root: Path) -> Path | None:
    """Return the newest run-root Pandora_science_calendar.xml if present."""
    candidates = sorted(
        repo_root.glob("output_*/Pandora_science_calendar.xml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def extract_sequences(xml_path: Path) -> list[dict[str, object]]:
    """Extract observation sequences from the science calendar XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sequences: list[dict[str, object]] = []

    for visit in root.findall(".//{/pandora/calendar/}Visit"):
        visit_id = visit.find("./{/pandora/calendar/}ID")
        visit_id_str = visit_id.text if visit_id is not None and visit_id.text else "Unknown"

        for obs_seq in visit.findall("./{/pandora/calendar/}Observation_Sequence"):
            seq_id = obs_seq.find("./{/pandora/calendar/}ID")
            seq_id_str = seq_id.text if seq_id is not None and seq_id.text else "Unknown"

            obs_params = obs_seq.find("./{/pandora/calendar/}Observational_Parameters")
            if obs_params is None:
                continue

            target = obs_params.find("./{/pandora/calendar/}Target")
            target_name = target.text if target is not None and target.text else "Unknown"

            timing = obs_params.find("./{/pandora/calendar/}Timing")
            if timing is None:
                continue

            start_elem = timing.find("./{/pandora/calendar/}Start")
            stop_elem = timing.find("./{/pandora/calendar/}Stop")
            if start_elem is None or stop_elem is None:
                continue
            if not start_elem.text or not stop_elem.text:
                continue

            start_time = parse_time(start_elem.text)
            end_time = parse_time(stop_elem.text)
            sequences.append(
                {
                    "type": "observation",
                    "target": target_name,
                    "visit_id": visit_id_str,
                    "seq_id": seq_id_str,
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time,
                }
            )

    sequences.sort(key=lambda item: item["start"])
    return sequences


def compute_gaps(sequences: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compute positive gaps between consecutive sequences."""
    gaps: list[dict[str, object]] = []
    for i in range(len(sequences) - 1):
        current_end = sequences[i]["end"]
        next_start = sequences[i + 1]["start"]
        gap_duration = next_start - current_end
        if gap_duration.total_seconds() <= 0:
            continue

        gaps.append(
            {
                "gap_number": len(gaps) + 1,
                "after_seq": i + 1,
                "before_seq": i + 2,
                "after_target": sequences[i]["target"],
                "after_type": sequences[i]["type"],
                "after_end": current_end,
                "before_target": sequences[i + 1]["target"],
                "before_type": sequences[i + 1]["type"],
                "before_start": next_start,
                "gap_duration": gap_duration,
                "gap_hours": gap_duration.total_seconds() / 3600.0,
            }
        )
    return gaps


def write_gap_csv(gaps: list[dict[str, object]], csv_path: Path) -> None:
    """Write the gap table as CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gap_number",
        "after_seq",
        "before_seq",
        "after_type",
        "after_target",
        "after_end",
        "before_type",
        "before_target",
        "before_start",
        "gap_hours",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for gap in gaps:
            writer.writerow(
                {
                    "gap_number": gap["gap_number"],
                    "after_seq": gap["after_seq"],
                    "before_seq": gap["before_seq"],
                    "after_type": gap["after_type"],
                    "after_target": gap["after_target"],
                    "after_end": gap["after_end"].isoformat(),
                    "before_type": gap["before_type"],
                    "before_target": gap["before_target"],
                    "before_start": gap["before_start"].isoformat(),
                    "gap_hours": f"{float(gap['gap_hours']):.4f}",
                }
            )


def write_histogram(gaps: list[dict[str, object]], hist_path: Path) -> None:
    """Write a histogram of gap sizes in hours."""
    import matplotlib.pyplot as plt

    hist_path.parent.mkdir(parents=True, exist_ok=True)
    values = [float(gap["gap_hours"]) for gap in gaps]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(30, max(5, len(values))), color="#2563eb", alpha=0.8, edgecolor="white")
    ax.set_title("Schedule Gap Histogram")
    ax.set_xlabel("Gap size (hours)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(hist_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_gaps(
    xml_path: Path,
    csv_out: Path | None = None,
    hist_out: Path | None = None,
    significant_hours: float = 1.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Analyze gaps between sequences in the XML calendar."""
    sequences = extract_sequences(xml_path)
    gaps = compute_gaps(sequences)

    print("=" * 80)
    print(f"SCHEDULE GAP ANALYSIS: {xml_path.name}")
    print("=" * 80)
    print(f"\nTotal sequences: {len(sequences)}")
    print(f"Total gaps: {len(gaps)}")

    if sequences:
        print(f"Schedule start: {sequences[0]['start']}")
        print(f"Schedule end: {sequences[-1]['end']}")
        total_scheduled = sum(
            float(seq["duration"].total_seconds()) for seq in sequences
        ) / 3600.0
        total_gap_time = sum(
            float(gap["gap_duration"].total_seconds()) for gap in gaps
        ) / 3600.0
        schedule_span = (
            sequences[-1]["end"] - sequences[0]["start"]
        ).total_seconds() / 3600.0
        print(f"Total scheduled time: {total_scheduled:.2f} hours")
        print(f"Total gap time: {total_gap_time:.2f} hours")
        print(f"Schedule span: {schedule_span:.2f} hours")
        if schedule_span > 0:
            print(f"Efficiency: {100.0 * total_scheduled / schedule_span:.2f}%")

    if gaps:
        gap_hours = [float(gap["gap_hours"]) for gap in gaps]
        print("\nGap statistics:")
        print(f"  Min gap: {min(gap_hours):.2f} hours")
        print(f"  Max gap: {max(gap_hours):.2f} hours")
        print(f"  Mean gap: {statistics.mean(gap_hours):.2f} hours")
        print(f"  Median gap: {statistics.median(gap_hours):.2f} hours")

        small_gaps = [gap for gap in gaps if float(gap["gap_hours"]) < 1.0]
        medium_gaps = [
            gap for gap in gaps if 1.0 <= float(gap["gap_hours"]) < 24.0
        ]
        large_gaps = [gap for gap in gaps if float(gap["gap_hours"]) >= 24.0]

        print("\nGap distribution:")
        print(f"  < 1 hour: {len(small_gaps)} ({100.0 * len(small_gaps) / len(gaps):.1f}%)")
        print(
            f"  1-24 hours: {len(medium_gaps)} ({100.0 * len(medium_gaps) / len(gaps):.1f}%)"
        )
        print(
            f"  >= 24 hours: {len(large_gaps)} ({100.0 * len(large_gaps) / len(gaps):.1f}%)"
        )

    print("\n" + "=" * 80)
    print("DETAILED GAP REPORT")
    print("=" * 80)

    if not gaps:
        print("\nNo gaps found - sequences are perfectly contiguous!")
    else:
        significant_gaps = [
            gap for gap in gaps if float(gap["gap_hours"]) >= significant_hours
        ]
        if significant_gaps:
            print(
                f"\nGaps >= {significant_hours:.1f} hour"
                f"{'' if significant_hours == 1.0 else 's'} ({len(significant_gaps)} total):\n"
            )
            for gap in significant_gaps:
                print(
                    f"Gap #{gap['after_seq']} -> #{gap['before_seq']}: "
                    f"{float(gap['gap_hours']):.2f} hours"
                )
                print(
                    f"  After:  [{gap['after_type']}] {gap['after_target']} ends {gap['after_end']}"
                )
                print(
                    f"  Before: [{gap['before_type']}] {gap['before_target']} starts {gap['before_start']}"
                )
                print()

        print("\n" + "-" * 80)
        print("TOP 20 LARGEST GAPS")
        print("-" * 80 + "\n")
        for index, gap in enumerate(
            sorted(gaps, key=lambda item: float(item["gap_hours"]), reverse=True)[:20],
            start=1,
        ):
            print(f"{index}. Gap of {float(gap['gap_hours']):.2f} hours")
            print(
                f"   After seq #{gap['after_seq']}: [{gap['after_type']}] {gap['after_target']}"
            )
            print(f"   End: {gap['after_end']}")
            print(
                f"   Before seq #{gap['before_seq']}: [{gap['before_type']}] {gap['before_target']}"
            )
            print(f"   Start: {gap['before_start']}")
            print()

    csv_path = csv_out or (xml_path.parent / "schedule_gaps.csv")
    write_gap_csv(gaps, csv_path)
    print(f"\nDetailed gap data exported to: {csv_path}")

    if hist_out is not None:
        if gaps:
            write_histogram(gaps, hist_out)
            print(f"Gap histogram exported to: {hist_out}")
        else:
            print("Histogram skipped: no positive gaps to plot.")

    return sequences, gaps


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    xml_path = args.xml or find_default_xml(repo_root)

    if xml_path is None:
        print(
            "Error: no XML path provided and no output_*/Pandora_science_calendar.xml found."
        )
        return 1
    if not xml_path.exists():
        print(f"Error: XML file not found at {xml_path}")
        return 1

    analyze_gaps(
        xml_path=xml_path,
        csv_out=args.csv_out,
        hist_out=args.hist_out,
        significant_hours=args.significant_hours,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
