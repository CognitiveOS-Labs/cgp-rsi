#!/usr/bin/env python3
"""Fail the build unless every anchor in data/anchors.jsonl passes the
deterministic verifier. Stdlib-only; runnable in the Alpine CI image."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi.verifier import verify_candidate  # noqa: E402

failures = 0
for line in Path("data/anchors.jsonl").read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    item = json.loads(line)
    verdict = verify_candidate(item["problem"], item["solution"])
    if not verdict.passed:
        print(f"FAIL {item['id']}: {verdict.error}")
        failures += 1
if failures:
    print(f"{failures} anchor(s) failed verification")
    raise SystemExit(1)
print("all anchors pass the deterministic verifier")
