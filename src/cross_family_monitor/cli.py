"""Command-line interface: ``cfm review`` and ``cfm demo``.

Exit codes (see also ``cross_family_monitor.review``):

- 0: every judge found the trajectory clean.
- 1: at least one judge flagged a violation (usable as a CI gate).
- 2: no violation, but at least one judge was uncertain.
- 3: input or configuration error (bad trajectory file, bad config, bad
  judge spec).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .demo import SYSTEM_CARD_NOTE, run_demo
from .judges import Judge, LLMCLIJudge, judge_from_spec
from .review import (
    EXIT_CLEAN,
    EXIT_ERROR,
    EXIT_UNCERTAIN,
    EXIT_VIOLATION,
    ReviewReport,
    review_trajectory,
)
from .trajectory import TrajectoryError, load_trajectory

# Used when the user gives neither --model nor --config: one model per
# family, so a bare `cfm review` is already a cross-family review.
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "google": "gemini-2.5-pro",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfm",
        description=(
            "Review one agent trajectory with judges from two or more "
            "different model families and surface where they disagree."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser(
        "review",
        help="Review a trajectory JSON file with cross-family judges.",
    )
    review.add_argument(
        "trajectory",
        help="Path to a trajectory JSON file (see README for the schema).",
    )
    review.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "A judge, repeatable. Forms: 'gpt-4o' (family inferred), "
            "'family:model' (explicit family), or 'stub:family:verdict' "
            "(offline stub judge; verdict is clean, violation, or "
            "uncertain). Defaults to one model per family when neither "
            "--model nor --config is given."
        ),
    )
    review.add_argument(
        "--config",
        metavar="FILE",
        help="JSON or YAML file mapping family -> model, e.g. "
        '{"openai": "gpt-4o", "anthropic": "claude-sonnet-4-5"}.',
    )
    review.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    review.add_argument(
        "--timeout",
        type=int,
        default=120,
        metavar="SECONDS",
        help="Per-judge timeout in seconds for the `llm` CLI (default: 120).",
    )

    demo = sub.add_parser(
        "demo",
        help="Run the bundled example trajectory through stub judges "
        "that disagree. Needs no API keys.",
    )
    demo.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    return parser


def load_judges_from_config(path: str | Path) -> list[Judge]:
    """Load judges from a JSON or YAML config file mapping family -> model."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise TrajectoryError(f"config file not found: {path}") from None
    except OSError as exc:
        raise TrajectoryError(f"cannot read config file {path}: {exc}") from None
    data: Any
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise TrajectoryError(
                "YAML config requires PyYAML; install it or use a JSON config"
            ) from None
        try:
            data = yaml.safe_load(raw)
        except Exception as exc:  # yaml.YAMLError
            raise TrajectoryError(f"config file {path} is not valid YAML: {exc}") from None
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TrajectoryError(
                f"config file {path} is not valid JSON: {exc}"
            ) from None
    if not isinstance(data, dict) or not data:
        raise TrajectoryError(
            f"config file {path} must contain a non-empty mapping of "
            "family -> model"
        )
    judges: list[Judge] = []
    for family, model in data.items():
        if not isinstance(family, str) or not family.strip():
            raise TrajectoryError(
                f"config file {path}: family names must be non-empty strings"
            )
        if not isinstance(model, str) or not model.strip():
            raise TrajectoryError(
                f"config file {path}: model for family {family!r} must be "
                "a non-empty string"
            )
        judges.append(LLMCLIJudge(model=model.strip(), family=family.strip()))
    return judges


def _warn_duplicate_families(judges: list[Judge]) -> None:
    seen: set[str] = set()
    for judge in judges:
        if judge.family in seen:
            print(
                f"warning: family {judge.family!r} has more than one judge; "
                "cross-family review is most useful with one judge per family",
                file=sys.stderr,
            )
            return
        seen.add(judge.family)


def format_report_text(report: ReviewReport) -> str:
    """Render a report as human-readable text."""
    lines = [
        f"Cross-family review: {report.trajectory_id}",
        f"Agreement: {report.agreement.upper()}",
        "",
    ]
    for verdict in report.verdicts:
        who = verdict.family or verdict.model
        label = f"  {who} ({verdict.model}): {verdict.verdict.upper()}"
        if verdict.verdict != "clean":
            label += f" ({verdict.severity})"
        lines.append(label)
        if verdict.flagged_steps:
            lines.append(f"    flagged steps: {verdict.flagged_steps}")
        for reason in verdict.reasons:
            lines.append(f"    - {reason}")
    lines.append("")
    lines.append("Where judges disagree:")
    for line in report.disagreements:
        lines.append(f"  - {line}")
    lines.append("")
    if report.exit_code == EXIT_VIOLATION:
        lines.append("Exit code: 1 (a judge flagged a violation)")
    elif report.exit_code == EXIT_UNCERTAIN:
        lines.append("Exit code: 2 (no violation, but a judge was uncertain)")
    else:
        lines.append("Exit code: 0 (all judges found the trajectory clean)")
    return "\n".join(lines)


def format_report_json(report: ReviewReport) -> str:
    """Render a report as machine-readable JSON."""
    return json.dumps(
        {
            "trajectory_id": report.trajectory_id,
            "agreement": report.agreement,
            "exit_code": report.exit_code,
            "verdicts": [
                {
                    "family": verdict.family,
                    "model": verdict.model,
                    "verdict": verdict.verdict,
                    "severity": verdict.severity,
                    "flagged_steps": verdict.flagged_steps,
                    "reasons": verdict.reasons,
                }
                for verdict in report.verdicts
            ],
            "disagreements": report.disagreements,
            "per_step_flags": {
                str(step): families
                for step, families in report.per_step_flags.items()
            },
        },
        indent=2,
    )


def _emit(report: ReviewReport, fmt: str, note: str = "") -> None:
    if fmt == "json":
        print(format_report_json(report))
    else:
        print(format_report_text(report))
    if note:
        print()
        print(note)


def _cmd_review(args: argparse.Namespace) -> int:
    try:
        trajectory = load_trajectory(args.trajectory)
    except TrajectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    try:
        if args.config:
            judges = load_judges_from_config(args.config)
        elif args.model:
            judges = [judge_from_spec(spec, timeout=args.timeout) for spec in args.model]
        else:
            judges = [
                LLMCLIJudge(model=model, family=family, timeout=args.timeout)
                for family, model in DEFAULT_MODELS.items()
            ]
    except (TrajectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    _warn_duplicate_families(judges)
    report = review_trajectory(trajectory, judges)
    _emit(report, args.format)
    return report.exit_code


def _cmd_demo(args: argparse.Namespace) -> int:
    report = run_demo()
    _emit(report, args.format, note=SYSTEM_CARD_NOTE)
    return report.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "review":
        return _cmd_review(args)
    if args.command == "demo":
        return _cmd_demo(args)
    parser.error("a subcommand is required")
    return EXIT_ERROR  # pragma: no cover - argparse exits first


if __name__ == "__main__":
    raise SystemExit(main())
