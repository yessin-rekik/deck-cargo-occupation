from pathlib import Path

import pytest

from deckwatch.config import ConfigError, load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "deck.example.yaml"

FULL = """
image_size: [3840, 2160]
deck: {length_m: 61.5, width_m: 19.2, wall_height_m: 2.9, wall_offset_m: 0.4}
detector: {model_path: ../models/m.onnx}
geometry:
  standard_sizes:
    - {name: 10ft, length_m: 2.99, width_m: 2.44}
"""


def write(tmp_path, text):
    p = tmp_path / "deck.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_full_config_loads_with_defaults(tmp_path):
    cfg = load_config(write(tmp_path, FULL))
    assert cfg.image_size == (3840, 2160)
    assert cfg.filename_tz == "UTC"
    assert (cfg.deck.length_m, cfg.deck.width_m, cfg.deck.wall_height_m, cfg.deck.wall_offset_m) == (61.5, 19.2, 2.9, 0.4)
    assert cfg.detector.conf_min == 0.4
    assert cfg.geometry.default_height_m == {"container": 2.6, "other": 1.0}
    assert [s.name for s in cfg.geometry.standard_sizes] == ["10ft"]
    assert cfg.operations.change_min_m2 == 2.0
    assert cfg.calibration is None


def test_relative_paths_resolve_against_config_dir(tmp_path):
    cfg = load_config(write(tmp_path, FULL))
    assert cfg.detector.model_path == tmp_path / "../models/m.onnx"
    assert cfg.db_path == tmp_path / "../deckwatch.db"
    assert cfg.drift.refs_dir == tmp_path / "drift_refs"


def test_shipped_example_has_no_deck_dimensions():
    with pytest.raises(ConfigError, match="deck.length_m must be provided"):
        load_config(EXAMPLE)


@pytest.mark.parametrize("entry", ["length_m: 61.5", "width_m: 19.2", "wall_height_m: 2.9", "wall_offset_m: 0.4"])
def test_every_deck_dimension_is_required(tmp_path, entry):
    key = entry.split(":")[0]
    text = FULL.replace(entry, f"unused_{entry}")
    with pytest.raises(ConfigError, match=f"deck.{key}"):
        load_config(write(tmp_path, text))


def test_null_deck_dimension_raises(tmp_path):
    with pytest.raises(ConfigError, match="deck.width_m must be provided"):
        load_config(write(tmp_path, FULL.replace("width_m: 19.2", "width_m: null")))


def test_non_positive_deck_dimension_raises(tmp_path):
    with pytest.raises(ConfigError, match="deck.length_m must be > 0"):
        load_config(write(tmp_path, FULL.replace("length_m: 61.5", "length_m: 0")))
    with pytest.raises(ConfigError, match="deck.wall_offset_m must be >= 0"):
        load_config(write(tmp_path, FULL.replace("wall_offset_m: 0.4", "wall_offset_m: -1")))


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_filename_tz_raises(tmp_path):
    with pytest.raises(ConfigError, match="filename_tz 'Mars/Olympus' is not a valid IANA time zone"):
        load_config(write(tmp_path, FULL + "filename_tz: Mars/Olympus\n"))
