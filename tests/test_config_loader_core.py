import pytest

from tnfr_core.config.loader import get_params, load_detection_config


def test_get_params_merges_compound_overrides() -> None:
    config = {
        "defaults": {"pressure": 1.6, "camber": {"front": -3.0}},
        "tracks": {
            "__default__": {"pressure": 1.7},
            "BL1": {"pressure": 1.8, "compounds": {"soft": {"pressure": 1.9}}},
        },
        "cars": {
            "XFG": {
                "defaults": {"pressure": 2.0, "damping": {"front": 5}},
                "compounds": {"r2": {"pressure": 2.1}},
                "tracks": {
                    "BL1": {
                        "toe": {"front": -0.1},
                        "compounds": {"R2": {"pressure": 2.2}},
                    }
                },
            }
        },
    }

    params = get_params(
        config, car_model="XFG", track_name="BL1", tyre_compound="r2"
    )

    assert params["pressure"] == 2.2
    assert params["damping"]["front"] == 5
    assert params["toe"]["front"] == -0.1
    assert params["camber"]["front"] == -3.0


def test_get_params_uses_default_compound_when_specific_missing() -> None:
    config = {
        "cars": {
            "XFG": {
                "compounds": {
                    "__default__": {"pressure": 2.0},
                    "soft": {"pressure": 2.3},
                }
            }
        }
    }

    baseline = get_params(config, car_model="XFG")
    soft = get_params(config, car_model="XFG", tyre_compound="SOFT")

    assert baseline["pressure"] == 2.0
    assert soft["pressure"] == 2.3


def test_get_params_merges_class_hierarchy_and_global_compounds() -> None:
    config = {
        "defaults": {"pressure": 1.0, "damping": {"front": 4}},
        "compounds": {
            "__default__": {"pressure": 1.1, "brake_bias": 52},
            "R2": {"pressure": 1.15, "brake_bias": 54},
        },
        "tracks": {
            "__default__": {"pressure": 1.2},
            "BL1": {"pressure": 1.25, "compounds": {"r2": {"pressure": 1.3}}},
        },
        "classes": {
            "__default__": {"camber": {"front": -2.0}},
            "TCR": {
                "defaults": {"pressure": 1.35, "damping": {"front": 5}},
                "tracks": {"BL1": {"pressure": 1.38}},
                "compounds": {
                    "__default__": {"pressure": 1.36, "brake_bias": 56},
                    "r2": {"pressure": 1.37, "brake_bias": 58},
                },
            },
        },
        "cars": {
            "XFG": {
                "defaults": {"pressure": 1.4, "camber": {"front": -2.5}},
                "tracks": {
                    "__default__": {"toe": {"front": -0.05}},
                    "BL1": {
                        "pressure": 1.5,
                        "toe": {"front": -0.1},
                        "compounds": {"r2": {"pressure": 1.6, "brake_bias": 60}},
                    },
                },
                "compounds": {"r2": {"pressure": 1.55, "brake_bias": 59}},
            }
        },
    }

    params = get_params(
        config,
        car_class="TCR",
        car_model="XFG",
        track_name="BL1",
        tyre_compound="R2",
    )

    assert params["pressure"] == 1.6
    assert params["damping"]["front"] == 5
    assert params["camber"]["front"] == -2.5
    assert params["toe"]["front"] == -0.1
    assert params["brake_bias"] == 60


def test_load_detection_config_explicit_path(tmp_path) -> None:
    override = tmp_path / "detection.yaml"
    override.write_text("defaults:\n  pressure: 2.5\n", encoding="utf-8")

    payload = load_detection_config(path=override)

    assert payload["defaults"]["pressure"] == 2.5


def test_load_detection_config_missing_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_detection_config(path=tmp_path / "missing.yaml")


def test_load_detection_config_search_paths_directory(tmp_path) -> None:
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "detection.yaml").write_text(
        "defaults:\n  pressure: 2.6\n", encoding="utf-8"
    )

    payload = load_detection_config(search_paths=[site_dir])

    assert payload["defaults"]["pressure"] == 2.6


def test_load_detection_config_pack_root(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "detection.yaml").write_text(
        "defaults:\n  pressure: 2.7\n", encoding="utf-8"
    )

    payload = load_detection_config(pack_root=tmp_path)

    assert payload["defaults"]["pressure"] == 2.7


def test_load_detection_config_packaged_default() -> None:
    payload = load_detection_config()

    assert payload["defaults"]["mutation_window"] == 12
    cars_section = payload["cars"]
    assert cars_section["XFG"]["compounds"]["r2"]["coherence_threshold"] == 0.88
