"""Fold a large tool result, then reach back into it three different ways.

    python examples/02_folding_a_tool_result.py

This is the whole protocol in one file: a result goes in, a digest comes out,
and the body stays on this machine where it can still be addressed.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rangerkit.fold import gate, ledger  # noqa: E402

state = Path(tempfile.mkdtemp(prefix="rangerkit-example-"))
try:
    ledger.configure(state)

    # A tool result of the size that actually causes the problem: a CSV that a
    # model would otherwise carry in its window for the rest of the run.
    rows = ["id,customer,region,amount"]
    for i in range(1, 4001):
        rows.append(f"{i},cust{i:05d},{['EMEA', 'AMER', 'APAC'][i % 3]},{(i * 37) % 991}")
    body = "\n".join(rows)

    print("=" * 70)
    print("1. What the tool returned")
    print("=" * 70)
    print(f"{len(body):,} characters · {len(body.splitlines()):,} lines")

    digest, removed, handle, _structure = gate.fold("read_file", body, threshold=2000)

    print()
    print("=" * 70)
    print("2. What the model sees instead")
    print("=" * 70)
    print(digest)
    print()
    print(f"{len(digest):,} characters — {len(digest) / len(body):.2%} of the body.")
    print(f"{removed:,} characters never entered the window, and never will:")
    print("the conversation is re-sent on every subsequent step, so this is a")
    print("saving on every one of them, not just this one.")

    print()
    print("=" * 70)
    print("3. peek -- a slice, addressed by line")
    print("=" * 70)
    print(gate.peek(handle, start=1, count=4))

    print()
    print("=" * 70)
    print("4. grep -- the lines that match")
    print("=" * 70)
    print(gate.grep(handle, "cust03999", max_hits=3))

    print()
    print("=" * 70)
    print("5. stats -- a total over ALL of it, without reading it back")
    print("=" * 70)
    print(gate.stats(handle, column="amount", op="sum"))
    print()
    print(gate.stats(handle, column="amount", op="mean", where="APAC"))
    print()
    print(gate.stats(handle, column="region", op="top"))

    print()
    print("This is the verb the other two cannot replace. A sum is a property of")
    print("every row, so `peek` and `grep` could only answer it by paging the")
    print("whole body back -- which costs more than never folding it. The body is")
    print("already here, where arithmetic is free, so only the scalar crosses.")

    print()
    print("=" * 70)
    print("6. And it refuses rather than guessing")
    print("=" * 70)
    print(gate.stats(handle, column="customer", op="sum"))
    print()
    print("Not `0`. A zero with a confident face on is a wrong answer;")
    print("a refusal is not.")
finally:
    shutil.rmtree(state, ignore_errors=True)
