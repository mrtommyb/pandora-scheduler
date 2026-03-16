#!/usr/bin/env python3
"""
Analyze gaps between sequences in Pandora science calendar XML.
"""
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
import sys


def parse_time(time_str: str) -> datetime:
    """Parse a time string in ISO format."""
    return datetime.fromisoformat(time_str.replace('Z', '+00:00'))


def analyze_gaps(xml_path: Path):
    """Analyze gaps between sequences in the XML calendar."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract all sequences with their times
    sequences = []
    
    # Parse the Pandora-specific XML format
    for visit in root.findall('.//{/pandora/calendar/}Visit'):
        visit_id = visit.find('.//{/pandora/calendar/}ID')
        visit_id_str = visit_id.text if visit_id is not None else "Unknown"
        
        for obs_seq in visit.findall('.//{/pandora/calendar/}Observation_Sequence'):
            seq_id = obs_seq.find('.//{/pandora/calendar/}ID')
            seq_id_str = seq_id.text if seq_id is not None else "Unknown"
            
            obs_params = obs_seq.find('.//{/pandora/calendar/}Observational_Parameters')
            if obs_params is None:
                continue
                
            target = obs_params.find('.//{/pandora/calendar/}Target')
            target_name = target.text if target is not None else "Unknown"
            
            timing = obs_params.find('.//{/pandora/calendar/}Timing')
            if timing is None:
                continue
                
            start_elem = timing.find('.//{/pandora/calendar/}Start')
            stop_elem = timing.find('.//{/pandora/calendar/}Stop')
            
            if start_elem is None or stop_elem is None:
                continue
                
            start_str = start_elem.text
            end_str = stop_elem.text
            
            start_time = parse_time(start_str)
            end_time = parse_time(end_str)
            duration = end_time - start_time
            
            sequences.append({
                'type': 'observation',
                'target': target_name,
                'visit_id': visit_id_str,
                'seq_id': seq_id_str,
                'start': start_time,
                'end': end_time,
                'duration': duration
            })
    
    # Sort by start time
    sequences.sort(key=lambda x: x['start'])
    
    # Calculate gaps
    gaps = []
    for i in range(len(sequences) - 1):
        current_end = sequences[i]['end']
        next_start = sequences[i + 1]['start']
        gap_duration = next_start - current_end
        
        if gap_duration.total_seconds() > 0:
            gaps.append({
                'after_seq': i,
                'before_seq': i + 1,
                'after_target': sequences[i]['target'],
                'after_type': sequences[i]['type'],
                'after_end': current_end,
                'before_target': sequences[i + 1]['target'],
                'before_type': sequences[i + 1]['type'],
                'before_start': next_start,
                'gap_duration': gap_duration,
                'gap_hours': gap_duration.total_seconds() / 3600
            })
    
    # Print summary statistics
    print("=" * 80)
    print(f"SCHEDULE GAP ANALYSIS: {xml_path.name}")
    print("=" * 80)
    print(f"\nTotal sequences: {len(sequences)}")
    print(f"Total gaps: {len(gaps)}")
    
    if sequences:
        print(f"Schedule start: {sequences[0]['start']}")
        print(f"Schedule end: {sequences[-1]['end']}")
        total_scheduled = sum(seq['duration'].total_seconds() for seq in sequences) / 3600
        total_gap_time = sum(gap['gap_duration'].total_seconds() for gap in gaps) / 3600
        schedule_span = (sequences[-1]['end'] - sequences[0]['start']).total_seconds() / 3600
        print(f"Total scheduled time: {total_scheduled:.2f} hours")
        print(f"Total gap time: {total_gap_time:.2f} hours")
        print(f"Schedule span: {schedule_span:.2f} hours")
        print(f"Efficiency: {100 * total_scheduled / schedule_span:.2f}%")
    
    if gaps:
        gap_hours = [g['gap_hours'] for g in gaps]
        print(f"\nGap statistics:")
        print(f"  Min gap: {min(gap_hours):.2f} hours")
        print(f"  Max gap: {max(gap_hours):.2f} hours")
        print(f"  Mean gap: {sum(gap_hours)/len(gap_hours):.2f} hours")
        print(f"  Median gap: {sorted(gap_hours)[len(gap_hours)//2]:.2f} hours")
        
        # Categorize gaps
        small_gaps = [g for g in gaps if g['gap_hours'] < 1]
        medium_gaps = [g for g in gaps if 1 <= g['gap_hours'] < 24]
        large_gaps = [g for g in gaps if g['gap_hours'] >= 24]
        
        print(f"\nGap distribution:")
        print(f"  < 1 hour: {len(small_gaps)} ({100*len(small_gaps)/len(gaps):.1f}%)")
        print(f"  1-24 hours: {len(medium_gaps)} ({100*len(medium_gaps)/len(gaps):.1f}%)")
        print(f"  >= 24 hours: {len(large_gaps)} ({100*len(large_gaps)/len(gaps):.1f}%)")
    
    # Print detailed gap report
    print("\n" + "=" * 80)
    print("DETAILED GAP REPORT")
    print("=" * 80)
    
    if not gaps:
        print("\nNo gaps found - sequences are perfectly contiguous!")
    else:
        # Show all gaps >= 1 hour
        significant_gaps = [g for g in gaps if g['gap_hours'] >= 1.0]
        
        if significant_gaps:
            print(f"\nGaps >= 1 hour ({len(significant_gaps)} total):\n")
            for gap in significant_gaps:
                print(f"Gap #{gap['after_seq']+1} -> #{gap['before_seq']+1}: {gap['gap_hours']:.2f} hours")
                print(f"  After:  [{gap['after_type']}] {gap['after_target']} ends {gap['after_end']}")
                print(f"  Before: [{gap['before_type']}] {gap['before_target']} starts {gap['before_start']}")
                print()
        
        # Show top 20 largest gaps
        print("\n" + "-" * 80)
        print("TOP 20 LARGEST GAPS")
        print("-" * 80 + "\n")
        
        top_gaps = sorted(gaps, key=lambda x: x['gap_hours'], reverse=True)[:20]
        for i, gap in enumerate(top_gaps, 1):
            print(f"{i}. Gap of {gap['gap_hours']:.2f} hours")
            print(f"   After seq #{gap['after_seq']+1}: [{gap['after_type']}] {gap['after_target']}")
            print(f"   End: {gap['after_end']}")
            print(f"   Before seq #{gap['before_seq']+1}: [{gap['before_type']}] {gap['before_target']}")
            print(f"   Start: {gap['before_start']}")
            print()
    
    # Export gap CSV for further analysis
    csv_path = xml_path.parent / "schedule_gaps.csv"
    with open(csv_path, 'w') as f:
        f.write("gap_number,after_seq,before_seq,after_type,after_target,after_end,before_type,before_target,before_start,gap_hours\n")
        for i, gap in enumerate(gaps, 1):
            f.write(f"{i},{gap['after_seq']+1},{gap['before_seq']+1},"
                   f"{gap['after_type']},{gap['after_target']},{gap['after_end']},"
                   f"{gap['before_type']},{gap['before_target']},{gap['before_start']},"
                   f"{gap['gap_hours']:.4f}\n")
    
    print(f"\nDetailed gap data exported to: {csv_path}")
    
    return sequences, gaps


if __name__ == "__main__":
    if len(sys.argv) > 1:
        xml_path = Path(sys.argv[1])
    else:
        # Default to most recent output
        xml_path = Path("output_standalone_test2/data_91_20_110/Pandora_science_calendar.xml")
    
    if not xml_path.exists():
        print(f"Error: XML file not found at {xml_path}")
        sys.exit(1)
    
    analyze_gaps(xml_path)
