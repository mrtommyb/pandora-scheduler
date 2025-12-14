from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pandorascheduler_rework.config import PandoraSchedulerConfig
from pandorascheduler_rework.pipeline import (
    SchedulerPaths,
    _maybe_generate_visibility,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_maybe_generate_visibility_invokes_builder(monkeypatch, tmp_path):
    # Create a minimal temporary package/data layout so this test is self-contained
    package_root = tmp_path / "package"
    data_dir = package_root / "data"
    data_dir.mkdir(parents=True)

    # Minimal CSV manifests (headers only) so file paths exist for the test
    (data_dir / "exoplanet_targets.csv").write_text(
        "Planet Name,Star Name,Primary Target,RA,DEC,Number of Transits to Capture\n"
    )
    (data_dir / "auxiliary-standard_targets.csv").write_text(
        "Star Name,RA,DEC,Priority\n"
    )
    (data_dir / "monitoring-standard_targets.csv").write_text(
        "Star Name,RA,DEC,Priority\n"
    )
    (data_dir / "occultation-standard_targets.csv").write_text(
        "Star Name,RA,DEC,Priority\n"
    )

    # Create a small dummy GMAT file so visibility generation configuration can be built
    gmat = data_dir / "Pandora-600km-withoutdrag-20251018.txt"
    gmat.write_text("# dummy GMAT content\n")

    # (Previously created a baseline/fingerprints.json here; not required for CSV manifests.)

    paths = SchedulerPaths.from_package_root(package_root)

    primary_target_csv = (paths.data_dir / "exoplanet_targets.csv").resolve()
    auxiliary_target_csv = (paths.data_dir / "auxiliary-standard_targets.csv").resolve()
    monitoring_target_csv = (paths.data_dir / "monitoring-standard_targets.csv").resolve()
    occultation_target_csv = (paths.data_dir / "occultation-standard_targets.csv").resolve()

    # Re-instantiate with explicit flag for this test since we want to test the flag logic
    # Also provide the temporary GMAT via extra_inputs so the visibility builder can be configured
    config = PandoraSchedulerConfig(
        targets_manifest=primary_target_csv,
        window_start=datetime(2026, 2, 5),
        window_end=datetime(2027, 2, 5),
        output_dir=tmp_path,
        extra_inputs={"generate_visibility": Path("true"), "visibility_gmat": gmat},
    )

    captured_configs: list[PandoraSchedulerConfig] = []

    def fake_builder(cfg, target_list, partner_list=None, output_subpath="targets"):  # type: ignore[no-untyped-def]
        captured_configs.append((cfg, target_list, partner_list, output_subpath))

    monkeypatch.setattr(
        "pandorascheduler_rework.pipeline.build_visibility_catalog",
        fake_builder,
    )

    _maybe_generate_visibility(
        config,
        paths,
        config.window_start,
        config.window_end,
        primary_target_csv,
        auxiliary_target_csv,
        monitoring_target_csv,
        occultation_target_csv,
    )

    # We expect the pipeline orchestration to call the visibility builder once
    # for each of: primary targets, auxiliary targets, monitoring targets,
    # and occultation targets. Each call receives a `VisibilityConfig`.
    assert len(captured_configs) == 4

    # Unpack configs in call order and assert key fields so the test is
    # reasonably comprehensive while remaining fast (the builder itself is
    # still monkeypatched out).
    primary_call, aux_call, mon_call, occ_call = captured_configs
    
    primary_cfg, primary_target, primary_partner, primary_subpath = primary_call
    aux_cfg, aux_target, aux_partner, aux_subpath = aux_call
    mon_cfg, mon_target, mon_partner, mon_subpath = mon_call
    occ_cfg, occ_target, occ_partner, occ_subpath = occ_call

    # Common expectations
    for cfg in (primary_cfg, aux_cfg, mon_cfg, occ_cfg):
        assert isinstance(cfg, PandoraSchedulerConfig)
        assert cfg.window_start == config.window_start
        assert cfg.window_end == config.window_end
        # GMAT path provided via extra_inputs should be propagated
        # Note: In the new implementation, GMAT path is in config.gmat_ephemeris or extra_inputs
        # The builder resolves it. Here we just check the config object is passed through.
        assert cfg == config

    # Primary targets: partner_list should be the auxiliary CSV and output_root
    # should be under the run's `output_dir/data/targets` subpath.
    assert primary_target == primary_target_csv
    assert primary_partner == auxiliary_target_csv
    assert primary_subpath == "targets"

    # Auxiliary targets: no partner list passed, output_root -> aux_targets
    assert aux_target == auxiliary_target_csv
    assert aux_partner is None
    assert aux_subpath == "aux_targets"

    # Monitoring targets: similar to auxiliary
    assert mon_target == monitoring_target_csv
    assert mon_partner is None
    assert mon_subpath == "aux_targets"

    # Occultation targets: also mapped to aux_targets
    assert occ_target == occultation_target_csv
    assert occ_partner is None
    assert occ_subpath == "aux_targets"


def test_maybe_generate_visibility_skips_without_flag(monkeypatch, tmp_path):
    repo_root = _repo_root()
    package_root = (repo_root / "src" / "pandorascheduler").resolve()
    paths = SchedulerPaths.from_package_root(package_root)

    primary_target_csv = (paths.data_dir / "exoplanet_targets.csv").resolve()
    auxiliary_target_csv = (paths.data_dir / "auxiliary-standard_targets.csv").resolve()
    monitoring_target_csv = (paths.data_dir / "monitoring-standard_targets.csv").resolve()
    occultation_target_csv = (paths.data_dir / "occultation-standard_targets.csv").resolve()

    config = PandoraSchedulerConfig(
        targets_manifest=package_root / "data" / "baseline" / "fingerprints.json",
        window_start=datetime(2026, 2, 5),
        window_end=datetime(2027, 2, 5),
        output_dir=tmp_path,
    )

    called = False

    def fake_builder(_cfg, target_list, partner_list=None, output_subpath="targets"):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    monkeypatch.setattr(
        "pandorascheduler_rework.pipeline.build_visibility_catalog",
        fake_builder,
    )

    _maybe_generate_visibility(
        config,
        paths,
        config.window_start,
        config.window_end,
        primary_target_csv,
        auxiliary_target_csv,
        monitoring_target_csv,
        occultation_target_csv,
    )

    assert not called


# test_build_visibility_config_supports_overrides removed as _build_visibility_config was removed