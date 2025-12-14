"""Unified configuration system for Pandora Scheduler.

This module consolidates the scattered configuration classes into a single,
hierarchical system that's easier to understand and maintain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PandoraSchedulerConfig:
    """Master configuration for the Pandora Scheduler pipeline.

    This consolidates SchedulerConfig, ScienceCalendarConfig, and VisibilityConfig
    into a single, coherent configuration object.
    """

    # ============================================================================
    # TIMING & WINDOWS
    # ============================================================================

    window_start: datetime
    """Start of the scheduling window."""

    window_end: datetime
    """End of the scheduling window."""

    schedule_step: timedelta = timedelta(hours=24)
    """Rolling scheduling step size (default: 24 hours).

    This controls how far the scheduler advances its rolling window each
    iteration. It is not a per-target visit duration.
    """

    commissioning_days: int = 0
    """Number of commissioning days at start of mission."""

    # ============================================================================
    # PATHS & DATA SOURCES
    # ============================================================================

    targets_manifest: Optional[Path] = None
    """Path to target definition manifest/directory."""

    gmat_ephemeris: Optional[Path] = None
    """Path to GMAT ephemeris file (for visibility generation)."""

    output_dir: Optional[Path] = None
    """Output directory for generated files."""

    # ============================================================================
    # SCHEDULING THRESHOLDS
    # ============================================================================

    transit_coverage_min: float = 0.2
    """Minimum transit coverage to schedule (0-1). Lower = more transits scheduled."""

    min_visibility: float = 0.0
    """Minimum visibility fraction to consider observable."""

    # ============================================================================
    # TRANSIT EDGE BUFFER PARAMETERS
    # ============================================================================

    short_visit_threshold_hours: float = 12.0
    """Visits shorter than this use short_visit_edge_buffer_hours."""

    short_visit_edge_buffer_hours: float = 1.5
    """Edge buffer (pre/post transit) for visits < short_visit_threshold_hours."""

    long_visit_edge_buffer_hours: float = 4.0
    """Edge buffer (pre/post transit) for visits >= short_visit_threshold_hours."""

    # ============================================================================
    # WEIGHTING FACTORS (must sum to 1.0)
    # ============================================================================

    transit_scheduling_weights: Tuple[float, float, float] = (0.8, 0.0, 0.2)
    """Unified transit scheduling weights: (coverage, saa, schedule).

    This single triple is used both by the scheduling algorithm and is recorded
    into the science calendar metadata. It replaces the previous separate
    `sched_weights` and `calendar_weights` fields.
    """

    # ============================================================================
    # KEEPOUT ANGLES (degrees)
    # ============================================================================

    sun_avoidance_deg: float = 91.0
    """Minimum angle from Sun (degrees)."""

    moon_avoidance_deg: float = 25.0
    """Minimum angle from Moon (degrees)."""

    earth_avoidance_deg: float = 86.0
    """Minimum angle from Earth limb (degrees)."""

    # ============================================================================
    # XML GENERATION PARAMETERS
    # ============================================================================

    obs_sequence_duration_min: int = 90
    """Observation sequence duration in minutes."""

    occ_sequence_limit_min: int = 50
    """Maximum occultation sequence duration in minutes."""

    min_sequence_minutes: int = 5
    """Minimum sequence length to include in XML (shorter sequences dropped)."""

    break_occultation_sequences: bool = True
    """Break long occultation sequences into chunks."""

    # ============================================================================
    # STANDARD OBSERVATIONS
    # ============================================================================

    std_obs_duration_hours: float = 0.5
    """Duration of standard star observations in hours."""

    std_obs_frequency_days: float = 3.0
    """Frequency of standard star observations in days."""

    # ============================================================================
    # BEHAVIOR FLAGS
    # ============================================================================

    show_progress: bool = False
    """Show progress bars during processing."""

    force_regenerate: bool = False
    """Force regeneration of files even if they exist."""

    use_target_list_for_occultations: bool = False
    """Use target list for occultation scheduling (vs. separate list)."""

    prioritise_occultations_by_slew: bool = False
    """Prioritize occultation targets by slew angle."""

    # ============================================================================
    # LEGACY COMPATIBILITY
    # ============================================================================

    use_legacy_mode: bool = False
    """Enable legacy scheduling behavior for validation against old outputs.
    
    When True, uses legacy algorithms that match the original scheduler exactly.
    When False (default), uses improved algorithms that may produce slightly
    different but equally valid (or better) results.
    
    Legacy behaviors controlled by this flag:
    - Visibility filtering: Uses MJD-based filtering (legacy) vs datetime-based
      filtering (modern). MJD filtering can exclude boundary points due to
      floating-point precision, while datetime filtering is more precise.
    
    Set to True when validating against historical baseline outputs.
    Set to False for production use with improved algorithms.
    """

    # ============================================================================
    # AUXILIARY SORTING
    # ============================================================================

    aux_sort_key: str = "sort_by_tdf_priority"
    """Key for sorting auxiliary targets."""

    # ============================================================================
    # METADATA
    # ============================================================================

    author: Optional[str] = None
    """Author name for XML metadata."""

    created_timestamp: Optional[datetime | str] = None
    """Creation timestamp for XML metadata."""

    visit_limit: Optional[int] = None
    """Limit number of visits (for testing). None = no limit."""

    target_filters: Sequence[str] = field(default_factory=tuple)
    """Target name filters for visibility generation."""

    extra_inputs: Dict[str, Path] = field(default_factory=dict)
    """Additional input files (auxiliary lists, etc.)."""

    # ============================================================================
    # VALIDATION
    # ============================================================================

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        # Validate transit_scheduling_weights sum to 1.0
        if not np.isclose(sum(self.transit_scheduling_weights), 1.0):
            raise ValueError(
                "transit_scheduling_weights must sum to 1.0, got %s"
                % (sum(self.transit_scheduling_weights),)
            )

        # Validate transit_coverage_min in range
        if not 0.0 <= self.transit_coverage_min <= 1.0:
            raise ValueError(
                "transit_coverage_min must be in [0, 1], got %s"
                % (self.transit_coverage_min,)
            )
