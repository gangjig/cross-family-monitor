"""Tests for the CLI: flags, formats, exit codes, config files, demo."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cross_family_monitor import cli
from cross_family_monitor.judges import StubJudge, stub_verdict

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAJ = {
    "id": "run-1",
    "steps": [{"kind": "action", "text": "did the thing"}],
}


@pytest.fixture
def traj_file(tmp_path):
    path = tmp_path / "traj.json"
    path.write_text(json.dumps(TRAJ), encoding="utf-8")
    return path


def test_review_all_clean_stub_exits_0(traj_file, capsys):
    code = cli.main(
        ["review", str(traj_file), "--model", "stub:openai:clean",
         "--model", "stub:anthropic:clean"]
    )
    assert code == 0
    assert "Agreement: UNANIMOUS" in capsys.readouterr().out


def test_review_violation_stub_exits_1(traj_file, capsys):
    code = cli.main(
        ["review", str(traj_file), "--model", "stub:openai:violation",
         "--model", "stub:anthropic:clean"]
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "Agreement: SPLIT" in out
    assert "VIOLATION" in out


def test_review_uncertain_only_exits_2(traj_file, capsys):
    code = cli.main(
        ["review", str(traj_file), "--model", "stub:openai:uncertain",
         "--model", "stub:anthropic:uncertain"]
    )
    assert code == 2
    assert "Exit code: 2" in capsys.readouterr().out


def test_review_json_format_parses(traj_file, capsys):
    code = cli.main(
        ["review", str(traj_file), "--model", "stub:openai:violation",
         "--format", "json"]
    )
    assert code == 1
    data = json.loads(capsys.readouterr().out)
    assert data["trajectory_id"] == "run-1"
    assert data["agreement"] == "all-flag"  # the single judge flagged a violation
    assert data["exit_code"] == 1
    assert data["verdicts"][0]["verdict"] == "violation"


def test_review_missing_file_exits_3(tmp_path, capsys):
    code = cli.main(["review", str(tmp_path / "nope.json")])
    assert code == 3
    assert "error:" in capsys.readouterr().err


def test_review_bad_trajectory_exits_3(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"id": "", "steps": []}', encoding="utf-8")
    code = cli.main(["review", str(path)])
    assert code == 3
    assert "error:" in capsys.readouterr().err


def test_review_bad_judge_spec_exits_3(traj_file, capsys):
    code = cli.main(["review", str(traj_file), "--model", "stub:openai:maybe"])
    assert code == 3
    assert "error:" in capsys.readouterr().err


def test_review_json_config(traj_file, tmp_path, monkeypatch):
    config = tmp_path / "judges.json"
    config.write_text(
        json.dumps({"openai": "gpt-4o", "anthropic": "claude-x"}),
        encoding="utf-8",
    )
    seen = {}

    class FakeJudge:
        def __init__(self, model, family="", timeout=120):
            seen[family] = model
            self.family = family
            self.model = model

        def review(self, trajectory, system_prompt, user_prompt):
            verdict = stub_verdict("clean")
            verdict.family = self.family
            verdict.model = self.model
            return verdict

    monkeypatch.setattr(cli, "LLMCLIJudge", FakeJudge)
    code = cli.main(["review", str(traj_file), "--config", str(config)])
    assert code == 0
    assert seen == {"openai": "gpt-4o", "anthropic": "claude-x"}


def test_review_yaml_config(traj_file, tmp_path, monkeypatch):
    config = tmp_path / "judges.yaml"
    config.write_text("openai: gpt-4o\nanthropic: claude-x\n", encoding="utf-8")

    class FakeJudge:
        def __init__(self, model, family="", timeout=120):
            self.family = family
            self.model = model

        def review(self, trajectory, system_prompt, user_prompt):
            verdict = stub_verdict("clean")
            verdict.family = self.family
            verdict.model = self.model
            return verdict

    monkeypatch.setattr(cli, "LLMCLIJudge", FakeJudge)
    assert cli.main(["review", str(traj_file), "--config", str(config)]) == 0


def test_review_config_must_be_mapping(traj_file, tmp_path, capsys):
    config = tmp_path / "judges.json"
    config.write_text('["not", "a", "mapping"]', encoding="utf-8")
    code = cli.main(["review", str(traj_file), "--config", str(config)])
    assert code == 3
    assert "error:" in capsys.readouterr().err


def test_review_defaults_use_one_model_per_family(traj_file, monkeypatch):
    built = []

    class FakeJudge:
        def __init__(self, model, family="", timeout=120):
            built.append((family, model))
            self.family = family
            self.model = model

        def review(self, trajectory, system_prompt, user_prompt):
            verdict = stub_verdict("clean")
            verdict.family = self.family
            verdict.model = self.model
            return verdict

    monkeypatch.setattr(cli, "LLMCLIJudge", FakeJudge)
    assert cli.main(["review", str(traj_file)]) == 0
    assert built == [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-sonnet-4-5"),
        ("google", "gemini-2.5-pro"),
    ]


def test_demo_exits_1_and_shows_disagreement(capsys):
    code = cli.main(["demo"])
    assert code == 1
    out = capsys.readouterr().out
    assert "Agreement: SPLIT" in out
    assert "openai (stub:openai): VIOLATION" in out
    assert "anthropic (stub:anthropic): CLEAN" in out
    assert "Motivation (from the GPT-6.2 Astra/Sol system card" in out


def test_demo_json_format_parses(capsys):
    code = cli.main(["demo", "--format", "json"])
    assert code == 1
    out = capsys.readouterr().out
    data = json.loads(out.split("\n\nMotivation")[0])
    assert data["agreement"] == "split"
    assert data["trajectory_id"] == "demo-fabrication-and-sabotage"


def test_version_flag():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_cli_entry_point_runs_demo_end_to_end():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "cross_family_monitor.cli", "demo"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert proc.returncode == 1
    assert "Agreement: SPLIT" in proc.stdout
