"""The README staleness check: strict on words, tolerant on the last digit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_docs.py"
spec = importlib.util.spec_from_file_location("build_docs", SCRIPT)
build_docs = importlib.util.module_from_spec(spec)
sys.modules["build_docs"] = build_docs
assert spec.loader is not None
spec.loader.exec_module(build_docs)

BLOCK = """| **shrunk** | 5.73% | 0.939 | 0.86 | 8.2% | -1.79% (significant) |
| _irreducible floor_ | _5.48%_ | _1.000_ | _--_ | _--_ | _floor_ |

**Overfitting.** PBO across 54 configurations: **24%**, reference 58%.

**Consistency.** Weight error decays as T^-0.41, reaching L1 = 0.083 at 8,064 observations.

<sub>Generated from run `26a3f3559164935e` (fingerprint `8a1a5c8e7582afed`), 2,887 days.</sub>"""


def test_identical_blocks_match():
    ok, reason = build_docs.tables_match(BLOCK, BLOCK)
    assert ok
    assert "deviation 0" in reason


def test_last_digit_drift_is_tolerated():
    drifted = BLOCK.replace("0.083", "0.084").replace("5.73%", "5.74%").replace("T^-0.41", "T^-0.42")
    ok, _ = build_docs.tables_match(BLOCK, drifted)
    assert ok


def test_a_material_numeric_change_fails():
    ok, reason = build_docs.tables_match(BLOCK, BLOCK.replace("5.73%", "6.50%"))
    assert not ok
    assert "moved from 5.73 to 6.5" in reason


def test_a_changed_word_fails_even_with_identical_numbers():
    ok, reason = build_docs.tables_match(BLOCK, BLOCK.replace("(significant)", "(not distinguishable)"))
    assert not ok
    assert "wording" in reason


def test_a_changed_identifier_fails():
    """Hashes are identifiers, not measurements: a different run id is stale."""
    ok, _ = build_docs.tables_match(BLOCK, BLOCK.replace("26a3f3559164935e", "ffffffffffffffff"))
    assert not ok


def test_a_missing_number_fails():
    ok, _ = build_docs.tables_match(BLOCK, BLOCK.replace(", 2,887 days", ""))
    assert not ok
