from tnfr_core.config.loader import get_params


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
