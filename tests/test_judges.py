"""Tests for judge implementations and judge specs."""

import subprocess

import pytest

from cross_family_monitor.judges import (
    LLMCLIJudge,
    StubJudge,
    infer_family,
    judge_from_spec,
    stub_verdict,
)
from cross_family_monitor.trajectory import parse_trajectory

TRAJ = parse_trajectory(
    {"id": "run-1", "steps": [{"kind": "action", "text": "x"}]}
)


@pytest.mark.parametrize(
    "model,family",
    [
        ("gpt-4o", "openai"),
        ("o1-mini", "openai"),
        ("claude-sonnet-4-5", "anthropic"),
        ("gemini-2.5-pro", "google"),
        ("gemma-3-27b", "google"),
        ("llama-3-70b", "meta"),
        ("mistral-large", "mistral"),
        ("grok-4", "xai"),
        ("deepseek-chat", "deepseek"),
        ("qwen2.5-72b", "qwen"),
        ("kimi-k2", "moonshot"),
    ],
)
def test_infer_family_known_prefixes(model, family):
    assert infer_family(model) == family


def test_infer_family_unknown():
    assert infer_family("some-future-model-9") == "unknown"


def test_stub_judge_returns_canned_verdict():
    judge = StubJudge(
        family="openai", model="stub:openai", canned=stub_verdict("violation")
    )
    verdict = judge.review(TRAJ, "system", "user")
    assert verdict.verdict == "violation"
    assert verdict.family == "openai"
    assert verdict.model == "stub:openai"


def test_stub_judge_defaults_to_clean():
    verdict = StubJudge(family="anthropic").review(TRAJ, "system", "user")
    assert verdict.verdict == "clean"


def test_stub_verdict_unknown_word():
    with pytest.raises(ValueError, match="unknown stub verdict"):
        stub_verdict("maybe")


def test_judge_from_spec_bare_model_infers_family():
    judge = judge_from_spec("claude-sonnet-4-5")
    assert isinstance(judge, LLMCLIJudge)
    assert judge.family == "anthropic"
    assert judge.model == "claude-sonnet-4-5"


def test_judge_from_spec_explicit_family_wins():
    judge = judge_from_spec("acme:gpt-4o")
    assert isinstance(judge, LLMCLIJudge)
    assert judge.family == "acme"
    assert judge.model == "gpt-4o"


def test_judge_from_spec_stub_with_verdict():
    judge = judge_from_spec("stub:openai:violation")
    assert isinstance(judge, StubJudge)
    assert judge.family == "openai"
    assert judge.review(TRAJ, "s", "u").verdict == "violation"


def test_judge_from_spec_stub_defaults_clean():
    judge = judge_from_spec("stub:google")
    assert judge.review(TRAJ, "s", "u").verdict == "clean"


def test_judge_from_spec_rejects_empty_family_or_model():
    with pytest.raises(ValueError, match="invalid judge spec"):
        judge_from_spec(":gpt-4o")
    with pytest.raises(ValueError, match="invalid judge spec"):
        judge_from_spec("openai:")
    with pytest.raises(ValueError, match="must not be empty"):
        judge_from_spec("")


def test_llm_cli_judge_missing_binary_is_uncertain(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    verdict = LLMCLIJudge(model="gpt-4o").review(TRAJ, "system", "user")
    assert verdict.verdict == "uncertain"
    assert "llm" in verdict.reasons[0]


def test_llm_cli_judge_parses_model_output(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout='{"verdict": "violation", "severity": "medium", '
            '"flagged_steps": [0], "reasons": ["bad step"]}',
            stderr="",
        )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llm")
    monkeypatch.setattr("subprocess.run", fake_run)
    verdict = LLMCLIJudge(model="gpt-4o").review(TRAJ, "system", "user")
    assert verdict.verdict == "violation"
    assert verdict.severity == "medium"
    assert verdict.flagged_steps == [0]


def test_llm_cli_judge_invokes_llm_with_model_and_prompts(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='{"verdict": "clean"}', stderr=""
        )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llm")
    monkeypatch.setattr("subprocess.run", fake_run)
    LLMCLIJudge(model="gemini-2.5-pro").review(TRAJ, "SYS", "USER")
    assert seen["cmd"][0] == "/usr/bin/llm"
    assert seen["cmd"][1:3] == ["-m", "gemini-2.5-pro"]
    assert seen["cmd"][3:5] == ["--system", "SYS"]
    assert seen["cmd"][5] == "USER"


def test_llm_cli_judge_nonzero_exit_is_uncertain(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=1, stdout="", stderr="boom"
        )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llm")
    monkeypatch.setattr("subprocess.run", fake_run)
    verdict = LLMCLIJudge(model="gpt-4o").review(TRAJ, "s", "u")
    assert verdict.verdict == "uncertain"
    assert "boom" in verdict.reasons[0]


def test_llm_cli_judge_timeout_is_uncertain(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="llm", timeout=120)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/llm")
    monkeypatch.setattr("subprocess.run", fake_run)
    verdict = LLMCLIJudge(model="gpt-4o", timeout=120).review(TRAJ, "s", "u")
    assert verdict.verdict == "uncertain"
    assert "timed out" in verdict.reasons[0]
