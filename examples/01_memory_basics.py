"""Store facts, get them back, and see what the store does that a dict does not.

    python examples/01_memory_basics.py

Everything here writes to a temp directory and cleans up after itself, so it is
safe to run repeatedly without touching a real store.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rangerkit import memory  # noqa: E402

state = Path(tempfile.mkdtemp(prefix="rangerkit-example-"))
try:
    memory.configure(state)

    print("=" * 70)
    print("1. Writing facts")
    print("=" * 70)
    for fact in [
        "Dave runs the training scripts on the RTX 4090 in the workstation",
        "The workstation has 64 GB of memory and a 2 TB NVMe drive",
        "The billing API is written in Go and the frontend is TypeScript",
        "Production deploys need a manual approval from Priya",
        "Standup is at 9:15 every weekday morning",
    ]:
        print(memory.remember(fact))

    print()
    print("=" * 70)
    print("2. Recall -- keyword overlap, plus a spread of related terms")
    print("=" * 70)
    for q in ["who approves production deploys",
              "how much memory does the workstation have",
              "what time is standup"]:
        print(f"\nQ: {q}")
        print(memory.recall(q))

    print()
    print("=" * 70)
    print("3. The write-side check: a fact that names nobody is reported")
    print("=" * 70)
    print(memory.remember("He prefers the blue one"))
    print()
    print("Nothing was refused and nothing was reworded. The extra line is the")
    print("whole intervention -- it arrives at the one moment the gap can still")
    print("be closed by the one party who can close it.")

    print()
    print("=" * 70)
    print("4. Dedup: a re-wording updates the row instead of adding one")
    print("=" * 70)
    memory.remember("Dave prefers dark mode in every editor and terminal")
    before = memory.count()
    memory.remember("Dave prefers dark mode in every editor and terminal window")
    print(f"facts before: {before} · after re-wording an existing one: {memory.count()}")
    print()
    print("The bar is Jaccard overlap above 0.75 on the scoring tokens, and it is")
    print("a bar rather than a guess: 'Standup is at 9:15 every weekday morning'")
    print("against '...without fail' scores 0.67 and stays two rows, because two")
    print("added words are two added words. What the merge keeps is the earned")
    print("tally -- unless the re-wording REVERSES the claim, in which case the")
    print("newer words still win the slot and the evidence resets to zero.")

    print()
    print("=" * 70)
    print("5. What rides the system prompt")
    print("=" * 70)
    print("preload() returns only sector-agnostic facts -- the ones that apply")
    print("whatever the user is doing -- so the prompt's bytes stay stable and")
    print("the KV prefix keeps matching:\n")
    print(memory.preload() or "(nothing sector-agnostic yet)")
    print()
    print(memory.sector_index() or "(no subjects yet)")
finally:
    shutil.rmtree(state, ignore_errors=True)
