"""Tests for trajectory loading and validation."""

import json

import pytest

from cross_family_monitor.trajectory import (
    TrajectoryError,
    load_trajectory,
    parse_trajectory,
)

VALID = {
    "id": "run-1",
    "steps": [
        {"kind": "model_output", "text": "Let me check the file."},
        {"kind": "tool_call", "text": "read_file('a.md')"},
        {"kind": "tool_result", "text": "contents"},
        {"kind": "action", "text": "Sent the summary."},
    ],
    "final_action": "Sent the summary.",
}


def test_valid_trajectory_parses():
    traj = parse_trajectory(VALID)
    assert traj.id == "run-1"
    assert len(traj.steps) == 4
    assert traj.steps[0].kind == "model_output"
    assert traj.final_action == "Sent the summary."


def test_final_action_defaults_to_empty():
    data = {"id": "run-1", "steps": [{"kind": "action", "text": "did it"}]}
    assert parse_trajectory(data).final_action == ""


def test_extra_top_level_keys_are_ignored():
    data = dict(VALID, metadata={"source": "harness"}, notes="extra")
    assert parse_trajectory(data).id == "run-1"


def test_non_object_top_level_rejected():
    with pytest.raises(TrajectoryError, match="must be a JSON object"):
        parse_trajectory(["not", "an", "object"])


def test_missing_id_rejected():
    data = {"steps": [{"kind": "action", "text": "x"}]}
    with pytest.raises(TrajectoryError, match="'id'"):
        parse_trajectory(data)


def test_empty_id_rejected():
    data = dict(VALID, id="   ")
    with pytest.raises(TrajectoryError, match="'id'"):
        parse_trajectory(data)


def test_steps_must_be_nonempty_list():
    for bad in ([], "not-a-list", None):
        with pytest.raises(TrajectoryError, match="'steps'"):
            parse_trajectory({"id": "run-1", "steps": bad})


def test_step_must_be_object():
    data = {"id": "run-1", "steps": ["not-an-object"]}
    with pytest.raises(TrajectoryError, match="step 0 must be an object"):
        parse_trajectory(data)


def test_step_unknown_kind_rejected():
    data = {
        "id": "run-1",
        "steps": [{"kind": "telepathy", "text": "x"}],
    }
    with pytest.raises(TrajectoryError, match="unknown kind"):
        parse_trajectory(data)


def test_step_text_must_be_string():
    data = {"id": "run-1", "steps": [{"kind": "action", "text": 42}]}
    with pytest.raises(TrajectoryError, match="step 0 'text'"):
        parse_trajectory(data)


def test_final_action_must_be_string():
    data = dict(VALID, final_action=7)
    with pytest.raises(TrajectoryError, match="'final_action'"):
        parse_trajectory(data)


def test_load_from_file(tmp_path):
    path = tmp_path / "traj.json"
    path.write_text(json.dumps(VALID), encoding="utf-8")
    assert load_trajectory(path).id == "run-1"


def test_load_missing_file(tmp_path):
    with pytest.raises(TrajectoryError, match="not found"):
        load_trajectory(tmp_path / "nope.json")


def test_load_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(TrajectoryError, match="not valid JSON"):
        load_trajectory(path)


def test_render_numbers_steps():
    traj = parse_trajectory(VALID)
    rendered = traj.render()
    assert "Step 0 [model_output]" in rendered
    assert "Step 3 [action]" in rendered
    assert "Final action: Sent the summary." in rendered
