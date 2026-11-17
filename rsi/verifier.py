"""Deterministic external verifier for self-generated training samples.

This is the component that prevents the pipeline from becoming an
unverified autophagous (self-consuming) loop. Every synthetic sample that
reaches `cpm tune` MUST pass through `verify_candidate`; unverified samples
are discarded by the controller.

Domain: Python algorithmic synthesis. A problem pins down a function name
and an input/output test suite. A candidate solution is:

  1. Static-AST checked (no imports outside an allowlist, no dunder access,
     no exec/eval/open/__import__, no I/O of any kind).
  2. Executed in a subprocess with CPU/memory/file-size/process-count
     limits and wall-clock timeout, talking only to a pipe.
  3. Scored against the problem's full test suite.

No human, LLM, or heuristic judge is involved anywhere in the verdict.
"""

from __future__ import annotations

import ast
import json
import resource
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Imports allowed inside candidate solutions. Everything else is rejected at
# the AST stage, before any code executes.
_ALLOWED_IMPORTS = frozenset(
    {
        "math",
        "collections",
        "itertools",
        "functools",
        "heapq",
        "bisect",
        "string",
        "re",
        "operator",
        "typing",
        "dataclasses",
    }
)

# Call names that are never permitted, even from allowed modules.
_BANNED_CALLS = frozenset({"exec", "eval", "open", "__import__", "input", "compile", "globals", "locals", "vars"})


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Verdict:
    passed: bool
    ast_ok: bool
    ast_error: str | None
    tests: list[TestResult] = field(default_factory=list)
    error: str | None = None
    elapsed_s: float = 0.0

    @property
    def pass_fraction(self) -> float:
        if not self.tests:
            return 0.0
        return sum(1 for t in self.tests if t.passed) / len(self.tests)


def ast_check(source: str) -> str | None:
    """Return None if the source passes the static policy, else a reason."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    return f"import not in allowlist: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORTS:
                return f"import-from not in allowlist: {node.module}"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BANNED_CALLS:
                return f"banned call: {func.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"dunder attribute access: {node.attr}"
    return None


_RUNNER = textwrap.dedent(
    """\
    import json, sys, resource

    # Hard resource ceilings (best-effort; enforced by OS where supported).
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))

    payload = json.load(sys.stdin)
    ns: dict = {}
    try:
        exec(compile(payload["solution"], "<candidate>", "exec"), ns)
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        print(json.dumps({"fatal": f"exec failure: {exc!r}"}))
        sys.exit(0)

    fn = ns.get(payload["entry_point"])
    if not callable(fn):
        print(json.dumps({"fatal": f"entry point not found or not callable: {payload['entry_point']}"}))
        sys.exit(0)

    results = []
    for test in payload["tests"]:
        try:
            got = fn(*test["args"])
            ok = got == test["expected"]
            results.append({"name": test["name"], "passed": bool(ok),
                            "detail": "" if ok else f"expected {test['expected']!r}, got {got!r}"})
        except Exception as exc:  # noqa: BLE001
            results.append({"name": test["name"], "passed": False, "detail": f"exception: {exc!r}"})
    print(json.dumps({"results": results}))
    """
)


def verify_candidate(problem: dict[str, Any], candidate_code: str, timeout_s: float = 6.0) -> Verdict:
    """Verify one candidate solution against a problem's test suite.

    `problem` schema:
        {
          "id": str,
          "entry_point": str,          # function name to call
          "tests": [{"name": str, "args": [...], "expected": ...}, ...]
        }
    """
    import time

    start = time.monotonic()
    ast_error = ast_check(candidate_code)
    if ast_error is not None:
        return Verdict(passed=False, ast_ok=False, ast_error=ast_error, error=ast_error,
                       elapsed_s=time.monotonic() - start)

    payload = {
        "solution": candidate_code,
        "entry_point": problem["entry_point"],
        "tests": problem["tests"],
    }
    with tempfile.TemporaryDirectory(prefix="rsi-verify-") as td:
        runner = Path(td) / "runner.py"
        runner.write_text(_RUNNER)
        try:
            proc = subprocess.run(
                [sys.executable, str(runner)],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return Verdict(passed=False, ast_ok=True, ast_error=None,
                           error=f"timeout after {timeout_s}s",
                           elapsed_s=time.monotonic() - start)

    out = (proc.stdout or "").strip()
    try:
        data = json.loads(out.splitlines()[-1]) if out else {}
    except json.JSONDecodeError:
        return Verdict(passed=False, ast_ok=True, ast_error=None,
                       error=f"runner produced no parseable output (stderr: {(proc.stderr or '')[:400]})",
                       elapsed_s=time.monotonic() - start)

    if "fatal" in data:
        return Verdict(passed=False, ast_ok=True, ast_error=None, error=data["fatal"],
                       elapsed_s=time.monotonic() - start)

    tests = [TestResult(name=r["name"], passed=bool(r["passed"]), detail=r.get("detail", ""))
             for r in data.get("results", [])]
    passed = bool(tests) and all(t.passed for t in tests)
    return Verdict(passed=passed, ast_ok=True, ast_error=None, tests=tests,
                   elapsed_s=time.monotonic() - start)
