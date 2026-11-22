"""Tests for verdict parsing and disagreement analysis."""

import pytest

from cross_family_monitor.review import (
    EXIT_CLEAN,
    EXIT_UNCERTAIN,
    EXIT_VIOLATION,
    Verdict,
    analyze,
    build_user_prompt,
    parse_verdict,
    review_trajectory,
)
from cross_family_monitor.trajectory import parse_trajectory

VALID_JSON = (
    '{"verdict": "violation", "severity": "high", '
    '"flagged_steps": [3, 6], "reasons": ["Step 3 fabricates data."]}'
)


def _verdict(word, family="openai", flagged=None):
    return Verdict(
        verdict=word,
        severity="high" if word == "violation" else "low",
        flagged_steps=list(flagged or []),
        reasons=[f"{family} says {word}"],
        family=family,
        model=f"stub:{family}",
    )


def test_parse_valid_verdict():
    verdict = parse_verdict(VALID_JSON, family="openai", model="gpt-4o")
    assert verdict.verdict == "violation"
    assert verdict.severity == "high"
    assert verdict.flagged_steps == [3, 6]
    assert verdict.reasons == ["Step 3 fabricates data."]
    assert verdict.family == "openai"
    assert verdict.model == "gpt-4o"


def test_parse_json_wrapped_in_prose():
    raw = (
        "Here is my audit:\n```json\n"
        + VALID_JSON
        + "\n```\nThat concludes the review."
    )
    assert parse_verdict(raw).verdict == "violation"


def test_parse_malformed_json_is_uncertain():
    verdict = parse_verdict("this is not json at all", family="openai")
    assert verdict.verdict == "uncertain"
    assert "could not be parsed" in verdict.reasons[0]
    assert verdict.raw == "this is not json at all"


def test_parse_missing_verdict_is_uncertain():
    verdict = parse_verdict('{"severity": "high", "reasons": []}')
    assert verdict.verdict == "uncertain"
    assert "valid verdict" in verdict.reasons[0]


def test_parse_unknown_verdict_word_is_uncertain():
    verdict = parse_verdict('{"verdict": "maybe", "reasons": []}')
    assert verdict.verdict == "uncertain"


def test_parse_missing_severity_defaults_low():
    verdict = parse_verdict('{"verdict": "clean", "reasons": []}')
    assert verdict.severity == "low"


def test_parse_invalid_severity_defaults_low():
    verdict = parse_verdict('{"verdict": "clean", "severity": "huge"}')
    assert verdict.severity == "low"


def test_parse_flagged_steps_filters_non_ints():
    verdict = parse_verdict(
        '{"verdict": "violation", "flagged_steps": [1, "two", 3.5, null, 4]}'
    )
    assert verdict.flagged_steps == [1, 4]


def test_parse_reasons_coerced_to_strings():
    verdict = parse_verdict('{"verdict": "clean", "reasons": [1, true]}')
    assert verdict.reasons == ["1", "True"]


def test_build_user_prompt_contains_numbered_steps():
    traj = parse_trajectory(
        {
            "id": "run-1",
            "steps": [
                {"kind": "model_output", "text": "hello"},
                {"kind": "action", "text": "done"},
            ],
        }
    )
    prompt = build_user_prompt(traj)
    assert "Step 0 [model_output]: hello" in prompt
    assert "Step 1 [action]: done" in prompt


def test_agreement_unanimous_clean():
    report = analyze("run-1", [_verdict("clean", "openai"), _verdict("clean", "anthropic")])
    assert report.agreement == "unanimous"
    assert report.exit_code == EXIT_CLEAN
    assert report.disagreements == ["All judges agree: clean."]


def test_agreement_unanimous_uncertain():
    report = analyze("run-1", [_verdict("uncertain", "openai"), _verdict("uncertain", "google")])
    assert report.agreement == "unanimous"
    assert report.exit_code == EXIT_UNCERTAIN


def test_agreement_all_flag():
    report = analyze(
        "run-1",
        [_verdict("violation", "openai", [1]), _verdict("violation", "anthropic", [1])],
    )
    assert report.agreement == "all-flag"
    assert report.exit_code == EXIT_VIOLATION
    assert report.disagreements == ["All judges agree: violation."]


def test_agreement_split():
    report = analyze(
        "run-1", [_verdict("violation", "openai", [3]), _verdict("clean", "anthropic")]
    )
    assert report.agreement == "split"
    assert report.exit_code == EXIT_VIOLATION


def test_exit_code_violation_wins_over_uncertain():
    report = analyze(
        "run-1", [_verdict("uncertain", "google"), _verdict("violation", "openai")]
    )
    assert report.exit_code == EXIT_VIOLATION


def test_disagreement_lines_name_families_and_steps():
    report = analyze(
        "run-1",
        [
            _verdict("violation", "openai", [3, 6]),
            _verdict("clean", "anthropic"),
            _verdict("uncertain", "google", [3]),
        ],
    )
    text = "\n".join(report.disagreements)
    assert "openai: violation (flagged steps [3, 6])" in text
    assert "anthropic: clean (flagged nothing)" in text
    assert "Step 3 was flagged by openai, google but not by anthropic." in text
    assert "Step 6 was flagged by openai but not by anthropic, google." in text


def test_per_step_flags_grouped_by_step():
    report = analyze(
        "run-1",
        [_verdict("violation", "openai", [3]), _verdict("violation", "google", [3, 5])],
    )
    assert report.per_step_flags == {3: ["openai", "google"], 5: ["google"]}


def test_analyze_requires_verdicts():
    with pytest.raises(ValueError, match="at least one judge"):
        analyze("run-1", [])


def test_review_trajectory_runs_every_judge():
    from cross_family_monitor.judges import StubJudge, stub_verdict

    traj = parse_trajectory(
        {"id": "run-1", "steps": [{"kind": "action", "text": "x"}]}
    )
    judges = [
        StubJudge(family="openai", canned=stub_verdict("violation")),
        StubJudge(family="anthropic", canned=stub_verdict("clean")),
    ]
    report = review_trajectory(traj, judges)
    assert report.trajectory_id == "run-1"
    assert report.agreement == "split"
    assert len(report.verdicts) == 2


def test_review_trajectory_requires_judges():
    traj = parse_trajectory(
        {"id": "run-1", "steps": [{"kind": "action", "text": "x"}]}
    )
    with pytest.raises(ValueError, match="at least one judge"):
        review_trajectory(traj, [])
