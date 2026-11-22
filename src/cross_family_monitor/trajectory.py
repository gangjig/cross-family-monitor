"""Trajectory loading and validation.

A trajectory is a JSON document describing one agent run:

    {
      "id": "run-2026-11-22-001",
      "steps": [
        {"kind": "model_output", "text": "..."},
        {"kind": "tool_call", "text": "read_file('shared/report.md')"},
        {"kind": "tool_result", "text": "..."},
        {"kind": "action", "text": "Sent the summary to the user."}
      ],
      "final_action": "Sent the summary to the user."
    }

Step kinds are fixed: ``model_output``, ``tool_call``, ``tool_result``,
``action``. ``final_action`` is optional and defaults to ``""``. Unknown
top-level keys are ignored so trajectories can carry extra metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STEP_KINDS = ("model_output", "tool_call", "tool_result", "action")


class TrajectoryError(ValueError):
    """Raised when a trajectory file fails validation."""


@dataclass(frozen=True)
class Step:
    """One ordered step of an agent trajectory."""

    kind: str
    text: str


@dataclass(frozen=True)
class Trajectory:
    """A validated agent trajectory."""

    id: str
    steps: tuple[Step, ...]
    final_action: str = ""

    def render(self) -> str:
        """Render the trajectory as numbered plain text for a judge prompt."""
        lines = [f"Trajectory id: {self.id}"]
        for index, step in enumerate(self.steps):
            lines.append(f"Step {index} [{step.kind}]: {step.text}")
        lines.append(f"Final action: {self.final_action or '(none)'}")
        return "\n".join(lines)


def parse_trajectory(data: Any) -> Trajectory:
    """Validate a decoded JSON value and return a Trajectory.

    Raises TrajectoryError with a specific message on the first problem
    found, so a bad input file fails with a clear error instead of a
    traceback.
    """
    if not isinstance(data, dict):
        raise TrajectoryError("trajectory must be a JSON object")
    trajectory_id = data.get("id")
    if not isinstance(trajectory_id, str) or not trajectory_id.strip():
        raise TrajectoryError("trajectory 'id' must be a non-empty string")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise TrajectoryError("trajectory 'steps' must be a non-empty list")
    steps: list[Step] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise TrajectoryError(f"step {index} must be an object")
        kind = raw_step.get("kind")
        if kind not in STEP_KINDS:
            raise TrajectoryError(
                f"step {index} has unknown kind {kind!r}; "
                f"expected one of {', '.join(STEP_KINDS)}"
            )
        text = raw_step.get("text")
        if not isinstance(text, str):
            raise TrajectoryError(f"step {index} 'text' must be a string")
        steps.append(Step(kind=kind, text=text))
    final_action = data.get("final_action", "")
    if not isinstance(final_action, str):
        raise TrajectoryError("trajectory 'final_action' must be a string")
    return Trajectory(
        id=trajectory_id.strip(),
        steps=tuple(steps),
        final_action=final_action,
    )


def load_trajectory(path: str | Path) -> Trajectory:
    """Load and validate a trajectory from a JSON file."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise TrajectoryError(f"trajectory file not found: {path}") from None
    except OSError as exc:
        raise TrajectoryError(f"cannot read trajectory file {path}: {exc}") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TrajectoryError(
            f"trajectory file {path} is not valid JSON: {exc}"
        ) from None
    return parse_trajectory(data)
