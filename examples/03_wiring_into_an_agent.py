"""The whole integration, in the order a real agent loop would do it.

    python examples/03_wiring_into_an_agent.py

There is no model here and none is needed: the "model" below is a scripted list
of the calls one would make, so the file shows the WIRING rather than pretending
to demonstrate a behaviour it cannot. Six places to touch, and that is all:

    1. configure the stores
    2. put `memory.preload()` in your system prompt
    3. offer the read-back tools once folding is on
    4. fold every tool result on the way back to the model
    5. tell the memory layer which run a turn belongs to
    6. grade the turn when it ends

Step 5 is the one that is easy to skip and expensive to skip. Reinforcement and
seed grading both need to know which facts a turn served and which it wrote, and
both read that from the run identity the registry carries.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rangerkit import brain, memory  # noqa: E402
from rangerkit.fold import gate, ledger, tools as fold_tools  # noqa: E402
from rangerkit.registry import registry  # noqa: E402

FOLD_THRESHOLD = 2000

state = Path(tempfile.mkdtemp(prefix="rangerkit-example-"))
try:
    # ---------------------------------------------------------------- step 1
    print("1. configure the stores")
    memory.configure(state)
    brain.configure(state)
    ledger.configure(state)
    print(f"   memory.db and vector.db under {state}\n")

    memory.remember("Dave runs the training scripts on the RTX 4090")
    memory.remember("Production deploys need a manual approval from Priya")

    # ---------------------------------------------------------------- step 2
    print("2. build the system prompt")
    system = "\n".join(filter(None, [
        "You are a helpful assistant.",
        "",
        "## What you already know",
        memory.preload(),
        memory.sector_index(),
        brain.preload(),
    ]))
    print("   " + "\n   ".join(system.splitlines()) + "\n")

    # ---------------------------------------------------------------- step 3
    print("3. switch the read-back tools on")
    fold_tools.set_active(True)
    print(f"   offered to the model: {[n for n in registry.names() if n.startswith('fold_')]}\n")

    # ---------------------------------------------------------------- step 5
    # A run identity, shared down any delegation chain. `remember` uses it to
    # note which rows this turn wrote, and `recall` to note which it served.
    print("5. open a turn (do this BEFORE the model runs, not after)")
    stop = threading.Event()
    registry.set_current_stop(stop)
    print("   registry.set_current_stop(<per-run event>)\n")

    try:
        # ------------------------------------------------------------ step 4
        print("4. fold each tool result on its way back to the model")
        result = "\n".join(
            f"{i},cust{i:05d},{(i * 37) % 991}" for i in range(1, 3001))
        to_model, removed, handle, _ = gate.fold("read_file", result, FOLD_THRESHOLD)
        print(f"   tool returned {len(result):,} chars; "
              f"the model is handed {len(to_model):,} ({removed:,} held back)")

        # The model then calls a read-back tool by name, like any other tool.
        answer = registry.dispatch("fold_stats", {"handle": handle, "column": "3",
                                                  "op": "max"})
        print(f"   model called fold_stats -> {answer.splitlines()[1]}\n")

        # And recall is an ordinary tool call too.
        served = memory.recall("who approves deploys")
        print(f"   model called recall -> {served.splitlines()[0]}\n")

        # ------------------------------------------------------------ step 6
        print("6. grade the turn when it ends")
        reinforced = memory.reinforce("ok")
        brain.reinforce("who approves deploys", "ok")
        print(f"   memory.reinforce('ok') credited rows {reinforced}")
        print("   a fact that fed a turn which WORKED earns a use/win tally, and")
        print("   that familiarity breaks ties at the next recall. Pass 'error'")
        print("   instead and any fact this turn WROTE is held out of the prompt")
        print("   digest until it earns its place -- kept in full, still")
        print("   recallable, just not promoted on the strength of a failed run.")
    finally:
        # Always clear it. A stale run id makes the next turn's writes look like
        # this one's, which is the one way this mechanism can credit the wrong row.
        registry.set_current_stop(None)
        fold_tools.set_active(False)

    print()
    print("That is the entire integration. Everything else in this repository is")
    print("either one of these mechanisms, or a harness that measures it.")
finally:
    shutil.rmtree(state, ignore_errors=True)
