"""Model clients: OpenAI-compatible HTTP endpoint + deterministic mock.

The real pipeline targets any OpenAI-compatible server (the CognitiveOS
`code-inference` server at http://api:8000/v1, vLLM, llama.cpp server, ...).
The MockModelClient exists for CI and for `scripts/mock_e2e.py`; it is a
test double and its outputs are NEVER training data for a real run.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAICompatibleClient:
    """Minimal stdlib client for /v1/chat/completions."""

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def complete(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.8,
                 seed: int | None = None) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"inference endpoint unreachable at {self.base_url}: {exc}") from exc
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


class MockModelClient:
    """Deterministic test double for CI. NOT a model. Outputs are canned."""

    def __init__(self, solving: bool = True):
        self.solving = solving

    def complete(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.8,
                 seed: int | None = None) -> Completion:
        if "Solve this Python problem" in prompt:
            # Solve request for the canned trivial problem; return a correct
            # solution for the canned problem shape only.
            match = re.search(r'"entry_point":\s*"([A-Za-z_][A-Za-z0-9_]*)"', prompt)
            entry = match.group(1) if match else "solve"
            if self.solving:
                return Completion(text=f"```python\ndef {entry}(*args):\n    return sum(args[0])\n```")
            return Completion(text=f"```python\ndef {entry}(*args):\n    return -1\n```")
        # Problem synthesis request: return a canned trivial problem.
        return Completion(
            text=json.dumps(
                {
                    "id": f"mock-{seed or 0}",
                    "entry_point": "total",
                    "prompt": "Return the sum of the list.",
                    "tests": [
                        {"name": "t1", "args": [[1, 2, 3]], "expected": 6},
                        {"name": "t2", "args": [[0]], "expected": 0},
                        {"name": "t3", "args": [[-1, 1]], "expected": 0},
                    ],
                }
            )
        )


class ProblemSynthesiser:
    """Ask the model to propose new (problem, tests) pairs."""

    TEMPLATE = (
        "Invent ONE novel Python algorithm problem in the domain {domain!s}.\n"
        "Respond with ONLY a JSON object with keys: id (string), entry_point (string), "
        "prompt (string), tests (array of {{name, args (array), expected}}).\n"
        "Requirements: deterministic, <= 6 tests, no I/O, no randomness, no imports beyond math.\n"
        "Domain: {domain}\n"
    )

    def __init__(self, client: OpenAICompatibleClient | MockModelClient, domain: str):
        self.client = client
        self.domain = domain

    def synthesize(self, *, seed: int, temperature: float = 0.9) -> dict[str, Any] | None:
        prompt = self.TEMPLATE.format(domain=self.domain)
        out = self.client.complete(prompt, max_tokens=700, temperature=temperature, seed=seed)
        return extract_json(out.text)


class SolutionGenerator:
    def __init__(self, client: OpenAICompatibleClient | MockModelClient):
        self.client = client

    def solve(self, problem: dict[str, Any], *, seed: int, temperature: float = 0.7) -> str | None:
        prompt = (
            "Solve this Python problem. Respond with ONLY a python code block.\n"
            f"problem: {json.dumps({k: problem[k] for k in ('id', 'entry_point', 'prompt', 'tests')}, default=str)}\n"
        )
        out = self.client.complete(prompt, max_tokens=900, temperature=temperature, seed=seed)
        return extract_code(out.text)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model output."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def extract_code(text: str) -> str | None:
    """Extract python code from a fenced code block (or raw source fallback)."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "def " in text:
        # Fallback: strip leading prose lines before the first def.
        idx = text.find("def ")
        return text[idx:].strip()
    return None
