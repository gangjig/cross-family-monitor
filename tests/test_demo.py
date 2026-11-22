"""Tests for the demo trajectory and demo judges."""

from cross_family_monitor.demo import (
    SYSTEM_CARD_NOTE,
    demo_judges,
    demo_trajectory,
    run_demo,
)


def test_demo_trajectory_validates():
    traj = demo_trajectory()
    assert traj.id == "demo-fabrication-and-sabotage"
    assert len(traj.steps) == 9
    assert traj.final_action


def test_demo_trajectory_shows_fabrication_and_sabotage():
    traj = demo_trajectory()
    assert "No interaction found" in traj.steps[2].text
    assert "severe liver damage" in traj.steps[3].text
    assert "lowered the success-rate figures" in traj.steps[6].text


def test_demo_judges_disagree():
    verdicts = [j.review(demo_trajectory(), "s", "u") for j in demo_judges()]
    words = {v.verdict for v in verdicts}
    assert words == {"violation", "clean", "uncertain"}
    families = {v.family for v in verdicts}
    assert families == {"openai", "anthropic", "google"}


def test_demo_report_is_split_with_exit_1():
    report = run_demo()
    assert report.agreement == "split"
    assert report.exit_code == 1
    assert report.per_step_flags[3] == ["openai", "google"]
    assert report.per_step_flags[6] == ["openai"]


def test_demo_note_cites_system_card_as_source_not_measurement():
    assert "19%" in SYSTEM_CARD_NOTE
    assert "51%" in SYSTEM_CARD_NOTE
    assert "system card's findings" in SYSTEM_CARD_NOTE
    assert "not measurements made by this tool" in SYSTEM_CARD_NOTE
    assert "thezvi.substack.com" in SYSTEM_CARD_NOTE
