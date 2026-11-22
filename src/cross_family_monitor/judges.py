"""Judge implementations.

A judge is any object with ``family``, ``model``, and a ``review`` method
that takes a trajectory plus the system and user prompts and returns a
:class:`~cross_family_monitor.review.Verdict`.

Two implementations ship:

- :class:`LLMCLIJudge` shells out to the ``llm`` CLI
  (https://llm.datasette.io), so any model family the user has configured
  in ``llm`` can judge. The family is inferred from the model id, or given
  explicitly with the ``family:model`` spec form.
- :class:`StubJudge` returns a canned verdict. It exists for tests, for
  the ``--demo`` run, and for offline use; it never calls a model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Protocol

from .review import Verdict, parse_verdict
from .trajectory import Trajectory

# Model-id prefixes mapped to the family that owns them. Only used when
# the user gives a bare model id; an explicit ``family:model`` spec always
# wins.
FAMILY_PREFIXES = {
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "gemma": "google",
    "llama": "meta",
    "mistral": "mistral",
    "mixtral": "mistral",
    "grok": "xai",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "kimi": "moonshot",
}


def infer_family(model: str) -> str:
    """Infer a model family from a model id, or ``"unknown"``."""
    lowered = model.lower()
    for prefix, family in FAMILY_PREFIXES.items():
        if lowered.startswith(prefix):
            return family
    return "unknown"


class Judge(Protocol):
    """Protocol every judge implementation follows."""

    family: str
    model: str

    def review(
        self, trajectory: Trajectory, system_prompt: str, user_prompt: str
    ) -> Verdict: ...


# Canned verdicts for stub judges, keyed by verdict word.
_STUB_CANNED: dict[str, dict] = {
    "clean": {
        "verdict": "clean",
        "severity": "low",
        "flagged_steps": [],
        "reasons": ["No misalignment behavior detected in this trajectory."],
    },
    "violation": {
        "verdict": "violation",
        "severity": "high",
        "flagged_steps": [],
        "reasons": ["Stub judge configured to return a violation verdict."],
    },
    "uncertain": {
        "verdict": "uncertain",
        "severity": "low",
        "flagged_steps": [],
        "reasons": ["Stub judge configured to return an uncertain verdict."],
    },
}


def stub_verdict(verdict_word: str) -> Verdict:
    """Build the canned Verdict for a stub judge."""
    if verdict_word not in _STUB_CANNED:
        raise ValueError(
            f"unknown stub verdict {verdict_word!r}; "
            "expected one of clean, violation, uncertain"
        )
    data = _STUB_CANNED[verdict_word]
    return Verdict(
        verdict=data["verdict"],
        severity=data["severity"],
        flagged_steps=list(data["flagged_steps"]),
        reasons=list(data["reasons"]),
        raw=json.dumps(data),
    )


@dataclass
class StubJudge:
    """A judge that returns a canned verdict; no model is ever called."""

    family: str
    model: str = "stub"
    canned: Verdict | None = None

    def review(
        self, trajectory: Trajectory, system_prompt: str, user_prompt: str
    ) -> Verdict:
        verdict = self.canned or stub_verdict("clean")
        return Verdict(
            verdict=verdict.verdict,
            severity=verdict.severity,
            flagged_steps=list(verdict.flagged_steps),
            reasons=list(verdict.reasons),
            raw=verdict.raw,
            family=self.family,
            model=self.model,
        )


@dataclass
class LLMCLIJudge:
    """A judge that shells out to the ``llm`` CLI.

    Runs ``llm -m <model> --system <system prompt> <user prompt>`` and
    parses the reply as a verdict. A missing ``llm`` binary, a non-zero
    exit, or a timeout all become ``uncertain`` with the cause recorded in
    reasons, so a broken judge can never silently pass a trajectory.
    """

    model: str
    family: str = ""
    timeout: int = 120

    def __post_init__(self) -> None:
        if not self.family:
            self.family = infer_family(self.model)

    def _uncertain(self, reason: str) -> Verdict:
        return Verdict(
            verdict="uncertain",
            severity="low",
            reasons=[reason],
            family=self.family,
            model=self.model,
        )

    def review(
        self, trajectory: Trajectory, system_prompt: str, user_prompt: str
    ) -> Verdict:
        llm = shutil.which("llm")
        if llm is None:
            return self._uncertain(
                "The `llm` CLI is not installed or not on PATH; "
                "install it (https://llm.datasette.io) or use a stub judge."
            )
        try:
            proc = subprocess.run(
                [llm, "-m", self.model, "--system", system_prompt, user_prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return self._uncertain(
                f"Judge model {self.model} timed out after {self.timeout}s."
            )
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip()
            return self._uncertain(
                f"Judge model {self.model} exited with code {proc.returncode}"
                + (f": {detail}" if detail else ".")
            )
        return parse_verdict(proc.stdout, family=self.family, model=self.model)


def judge_from_spec(spec: str, timeout: int = 120) -> Judge:
    """Build a judge from one ``--model`` spec.

    Accepted forms:

    - ``gpt-4o`` — an ``llm`` model id; the family is inferred.
    - ``family:model`` — an ``llm`` model id with an explicit family.
    - ``stub:family:verdict`` — a stub judge; verdict is one of
      ``clean``, ``violation``, ``uncertain`` (default ``clean``).
    """
    if spec.startswith("stub:"):
        parts = spec.split(":")
        family = parts[1] if len(parts) > 1 and parts[1] else "stub"
        verdict_word = parts[2] if len(parts) > 2 and parts[2] else "clean"
        return StubJudge(
            family=family,
            model=f"stub:{family}",
            canned=stub_verdict(verdict_word),
        )
    if ":" in spec:
        family, model = spec.split(":", 1)
        if not family or not model:
            raise ValueError(
                f"invalid judge spec {spec!r}; expected 'family:model'"
            )
        return LLMCLIJudge(model=model, family=family, timeout=timeout)
    if not spec:
        raise ValueError("judge spec must not be empty")
    return LLMCLIJudge(model=spec, timeout=timeout)
