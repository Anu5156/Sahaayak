"""Deterministic eligibility rules engine.

Each scheme declares its rules as data (see data/schemes.json). A rule is either
a condition on one profile field, or an ``any`` group that passes when at least
one of its conditions passes. Every decision comes with human-readable reasons,
so citizens can see exactly why a scheme matched or not.

Three outcomes per scheme:
  * eligible     - every rule passed
  * not_eligible - at least one rule failed
  * maybe        - nothing failed, but some fields were not provided
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"


def _compare(op: str, actual: Any, expected: Any) -> bool:
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "not_in":
        return actual not in expected
    if op == "gte":
        return actual >= expected
    if op == "lte":
        return actual <= expected
    if op == "gt":
        return actual > expected
    if op == "lt":
        return actual < expected
    if op == "between":
        low, high = expected
        return low <= actual <= high
    if op == "is_true":
        return actual is True
    if op == "is_false":
        return actual is False
    raise ValueError(f"Unknown operator: {op}")


def evaluate_rule(rule: dict, profile: dict) -> tuple[str, str]:
    """Return (status, reason) for one rule against a profile."""
    if "any" in rule:
        results = [evaluate_rule(r, profile) for r in rule["any"]]
        statuses = [s for s, _ in results]
        if PASS in statuses:
            passed = next(reason for s, reason in results if s == PASS)
            return PASS, f"{rule['reason']} (satisfied: {passed})"
        if UNKNOWN in statuses:
            return UNKNOWN, rule["reason"]
        return FAIL, rule["reason"]

    value = profile.get(rule["field"])
    if value is None:
        return UNKNOWN, rule["reason"]
    return (PASS if _compare(rule["op"], value, rule.get("value")) else FAIL), rule["reason"]


@dataclass
class EligibilityResult:
    scheme_id: str
    status: str
    met: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = len(self.met) + len(self.failed) + len(self.missing)
        return len(self.met) / total if total else 0.0


def check_scheme(scheme: dict, profile: dict) -> EligibilityResult:
    result = EligibilityResult(scheme_id=scheme["id"], status="eligible")
    for rule in scheme.get("rules", []):
        status, reason = evaluate_rule(rule, profile)
        {PASS: result.met, FAIL: result.failed, UNKNOWN: result.missing}[status].append(reason)
    if result.failed:
        result.status = "not_eligible"
    elif result.missing:
        result.status = "maybe"
    return result


_ORDER = {"eligible": 0, "maybe": 1, "not_eligible": 2}


def check_all(schemes: list[dict], profile: dict) -> list[EligibilityResult]:
    """Check every scheme; eligible first, then 'maybe' (most rules met first)."""
    clean = {k: v for k, v in profile.items() if v is not None and v != ""}
    results = [check_scheme(s, clean) for s in schemes]
    return sorted(results, key=lambda r: (_ORDER[r.status], -r.score))
