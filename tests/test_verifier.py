import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi.verifier import ast_check, verify_candidate

PROBLEM = {
    "id": "t-sum",
    "entry_point": "total",
    "tests": [
        {"name": "t1", "args": [[1, 2, 3]], "expected": 6},
        {"name": "t2", "args": [[0]], "expected": 0},
        {"name": "t3", "args": [[-1, 1]], "expected": 0},
    ],
}

GOOD = "def total(xs):\n    return sum(xs)\n"
WRONG = "def total(xs):\n    return 0\n"
BOMB = "import os\ndef total(xs):\n    return os.system('id')\n"
EVALER = "def total(xs):\n    return eval('1+1')\n"
DUNDER = "def total(xs):\n    return xs.__len__()\n"
LOOP = "def total(xs):\n    while True:\n        pass\n"


def test_correct_solution_passes():
    v = verify_candidate(PROBLEM, GOOD)
    assert v.passed and v.ast_ok and len(v.tests) == 3


def test_wrong_answer_fails_without_crash():
    v = verify_candidate(PROBLEM, WRONG)
    assert not v.passed and v.ast_ok
    assert any(not t.passed for t in v.tests)


def test_banned_import_rejected_at_ast_layer():
    v = verify_candidate(PROBLEM, BOMB)
    assert not v.passed and not v.ast_ok
    assert "allowlist" in (v.ast_error or "")


def test_eval_rejected_at_ast_layer():
    assert ast_check(EVALER) is not None


def test_dunder_access_rejected_at_ast_layer():
    assert ast_check(DUNDER) is not None


def test_timeout_enforced():
    v = verify_candidate(PROBLEM, LOOP, timeout_s=1.0)
    assert not v.passed and (v.error or "").startswith("timeout")


def test_syntax_error_reported():
    v = verify_candidate(PROBLEM, "def total(:\n")
    assert not v.passed and not v.ast_ok
