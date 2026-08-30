"""Fold A/B without a GPU: measure the plumbing, never the model.

The question "does folding actually do anything" splits cleanly into two
claims, and they need very different evidence:

    MECHANICAL   with the same tools called in the same order, does ON put
                 fewer characters through the window, and is the elided body
                 still reachable? Deterministic. No model needed. Settled here.

    BEHAVIOURAL  does a real model still get the right answer, in no more round
                 trips? Depends entirely on the model. NOT settled here, and no
                 arrangement of this file can settle it.

Conflating the two is the standard way this class of work fools itself: a
harness proves the bytes moved and then reports it as though the answers held.
So this file measures the first claim precisely and says nothing about the
second.

The scripted driver replays a FIXED tool-call trace, identical in both arms, so
the only variable is what the fold does to each result. It then answers each
question by searching the text ACTUALLY VISIBLE in its window -- it cannot use
knowledge it was not shown. That is the whole trick: correctness here is a
property of the plumbing (was the answer-bearing line still reachable?) rather
than of a policy this file invented for it.

**The column that is usually left out.** A conversation is re-sent in full on
every subsequent step, so a result read at step 3 is paid for again at steps 4
through N. Peak window size understates the cost of not folding by that factor;
cumulative prefill is the number that grows quadratically, and it is reported
here beside the peak rather than instead of it.

**The arm that makes this honest.** `readback` prices the case the fold loses.
`peek` and `grep` are bets that the model wants a few lines out of many; a
question about EVERY row (a sum, a count, the commonest value) can only be
answered by paging the whole body back a screen at a time -- which costs more
than never folding, because the conversation is re-sent at every hop. That arm
is run and reported, not hidden, and `stats` is measured against it.

    python bench/foldsim.py                  # small corpus, all arms
    python bench/foldsim.py --scale medium   # 8x, where OFF starts losing
    python bench/foldsim.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rangerkit.fold import gate, ledger  # noqa: E402

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m")

# Characters per token. A ratio, not a tokenizer: every arm is divided by the
# same constant, so it cancels out of every comparison this file makes and only
# ever sets the units on the token columns.
CHARS_PER_TOKEN = 3.8

SCALES = {"small": 1, "medium": 8, "large": 40}

# Fold anything above this. The same threshold in both arms; OFF simply never
# calls the gate.
THRESHOLD = 2000


# ------------------------------------------------------------------- corpus
def build_corpus(scale: int, seed: int = 20260830) -> dict[str, str]:
    """A deterministic corpus. Same seed, same bytes, on every machine.

    The three shapes are the three the fold actually meets: a line-oriented log
    (grep's case), a delimited table (stats' case), and prose with one buried
    fact (peek's case).
    """
    rng = random.Random(seed)
    regions = ["EMEA", "AMER", "APAC"]
    levels = ["INFO", "WARN", "ERROR"]

    log_lines = []
    for i in range(1, 400 * scale + 1):
        lvl = levels[0] if i % 17 else (levels[2] if i % 101 == 0 else levels[1])
        log_lines.append(f"2026-08-{(i % 28) + 1:02d} 11:{i % 60:02d}:{(i * 7) % 60:02d} "
                         f"{lvl:<5} worker-{i % 12} handled job {i} in {rng.randint(2, 900)}ms")
    # The one line every question about this file is really about.
    log_lines.insert(len(log_lines) // 2,
                     "2026-08-14 03:12:55 FATAL worker-9 checkpoint corrupted at offset 88214")

    rows = ["id,customer,region,amount"]
    total = apac = 0
    for i in range(1, 500 * scale + 1):
        amt = rng.randint(1, 999)
        region = regions[i % 3]
        total += amt
        if region == "APAC":
            apac += amt
        rows.append(f"{i},cust{i:05d},{region},{amt}")

    prose = []
    for i in range(1, 120 * scale + 1):
        prose.append(f"Section {i}. " + " ".join(
            rng.choice(["retention", "throughput", "ledger", "quota", "shard", "replica",
                        "checkpoint", "backfill", "index", "tenant"]) for _ in range(18)))
    prose.insert(len(prose) // 3,
                 "The production database password rotation window is the first Tuesday of each month.")

    csv_text = "\n".join(rows)
    corpus = {
        "app.log": "\n".join(log_lines),
        "sales.csv": csv_text,
        "handbook.txt": "\n".join(prose),
        "_sum_all": str(total),
        "_sum_apac": str(apac),
    }

    # The checker below searches for these strings in whatever the arm could
    # see. That is only a fair test if the string cannot ALSO turn up by
    # coincidence -- "500" as a row count is a substring of row id 500, and an
    # arm that computed nothing would score the point. So the property the
    # checker depends on is asserted here rather than assumed: an expected
    # answer that appears anywhere in the raw data is not a valid aggregate
    # probe, and this refuses to run rather than report a number it cannot
    # stand behind.
    for key in ("_sum_all", "_sum_apac"):
        if corpus[key] in csv_text:
            raise AssertionError(
                f"aggregate probe {key}={corpus[key]} occurs in the raw corpus, so an arm "
                f"that computed nothing would still match it. Change the seed or the shape.")
    return corpus


# -------------------------------------------------------------------- tasks
# Each task is a question, the tool result it concerns, and a deterministic
# checker over the text the arm could actually see. `needs_all` marks the ones
# that are a property of EVERY row -- the case the byte-returning verbs lose.
TASKS = [
    {"id": 1, "file": "app.log", "q": "at what offset was the checkpoint corrupted?",
     "find": "88214", "needs_all": False},
    {"id": 2, "file": "app.log", "q": "which worker logged the FATAL?",
     "find": "worker-9", "needs_all": False},
    {"id": 3, "file": "handbook.txt", "q": "when is the password rotation window?",
     "find": "first Tuesday", "needs_all": False},
    {"id": 4, "file": "sales.csv", "q": "what is the first customer id?",
     "find": "cust00001", "needs_all": False},
    {"id": 5, "file": "sales.csv", "q": "what is the total of the amount column?",
     "find": None, "needs_all": True, "op": "sum", "column": "amount",
     "where": "", "want": "_sum_all"},
    {"id": 6, "file": "sales.csv", "q": "what do the APAC rows add up to?",
     "find": None, "needs_all": True, "op": "sum", "column": "amount",
     "where": "APAC", "want": "_sum_apac"},
]


class Window:
    """The active context. Everything the arm is allowed to answer from.

    `steps` records the size after each tool result lands, which is what makes
    the cumulative column computable: a conversation is re-sent in full on every
    later step, so the total prefill is the sum of those sizes, not the last one.
    """

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.steps: list[int] = []
        self.round_trips = 0

    def add(self, text: str) -> None:
        self.parts.append(text)
        self.round_trips += 1
        self.steps.append(self.chars)

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.parts)

    @property
    def visible(self) -> str:
        return "\n".join(self.parts)

    @property
    def cumulative(self) -> int:
        """Characters the model must re-read across the whole run."""
        return sum(self.steps)


def _tok(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN)


# --------------------------------------------------------------------- arms
def arm_off(corpus: dict[str, str]) -> dict[str, Any]:
    """No fold. Every result lands in the window whole and stays there."""
    w = Window()
    retrieval = aggregate = 0
    for task in TASKS:
        w.add(corpus[task["file"]])
        if _answered(task, w.visible, corpus):
            aggregate += 1 if task["needs_all"] else 0
            retrieval += 0 if task["needs_all"] else 1
    return _result("off", w, retrieval, aggregate)


def arm_on(corpus: dict[str, str], work: Path) -> dict[str, Any]:
    """Fold, then read back with whichever verb the question actually needs."""
    ledger.configure(work)
    ledger.clear()
    w = Window()
    retrieval = aggregate = 0
    handles: dict[str, str] = {}

    for task in TASKS:
        name = task["file"]
        if name not in handles:
            digest, _removed, handle, _st = gate.fold("read_file", corpus[name], THRESHOLD)
            w.add(digest)
            handles[name] = handle

        handle = handles[name]
        if not handle:                       # below threshold: nothing was folded
            if _answered(task, w.visible, corpus):
                aggregate += 1 if task["needs_all"] else 0
                retrieval += 0 if task["needs_all"] else 1
            continue

        if task["needs_all"]:
            # The scalar case. One call, one number, no bytes.
            w.add(gate.stats(handle, column=task["column"], op=task["op"],
                             where=task.get("where", "")))
        elif task["find"]:
            # The slice case. grep is the verb a model reaches for when it knows
            # a distinguishing string, which is exactly what these questions have.
            w.add(gate.grep(handle, task["find"], max_hits=5))

        if _answered(task, w.visible, corpus):
            aggregate += 1 if task["needs_all"] else 0
            retrieval += 0 if task["needs_all"] else 1
    return _result("on", w, retrieval, aggregate)


def arm_readback(corpus: dict[str, str], work: Path) -> dict[str, Any]:
    """The fold's losing case, priced rather than hidden.

    Same fold, but the scalar questions are answered the only way the
    byte-returning verbs allow: page the whole body back. This is what
    `gate.stats` exists to replace, and the gap between this arm and `on` is
    the whole of what that verb is worth.
    """
    ledger.configure(work)
    ledger.clear()
    w = Window()
    retrieval = aggregate = 0
    handles: dict[str, str] = {}
    PAGE = 200

    for task in TASKS:
        name = task["file"]
        if name not in handles:
            digest, _r, handle, _s = gate.fold("read_file", corpus[name], THRESHOLD)
            w.add(digest)
            handles[name] = handle
        handle = handles[name]
        if not handle:
            if _answered(task, w.visible, corpus):
                aggregate += 1 if task["needs_all"] else 0
                retrieval += 0 if task["needs_all"] else 1
            continue

        if task["needs_all"]:
            total_lines = len(corpus[name].splitlines())
            start = 1
            while start <= total_lines:
                w.add(gate.peek(handle, start=start, count=PAGE, max_chars=20000))
                start += PAGE
        elif task["find"]:
            w.add(gate.grep(handle, task["find"], max_hits=5))

        if _answered(task, w.visible, corpus):
            aggregate += 1 if task["needs_all"] else 0
            retrieval += 0 if task["needs_all"] else 1
    return _result("readback", w, retrieval, aggregate)


def _answered(task: dict[str, Any], visible: str, corpus: dict[str, str]) -> bool:
    """Could the answer be read off what was actually in the window?

    Deliberately a search over visible text and nothing else. The checker has no
    access to the corpus except to know what the right answer IS -- it cannot
    supply one the arm was never shown.
    """
    if task["needs_all"]:
        # Digit separators only: the ledger prints large numbers grouped, and a
        # comma is also the CSV delimiter, so both sides are compared ungrouped.
        return corpus[task["want"]] in visible.replace(",", "")
    return bool(task["find"]) and task["find"] in visible


N_RETRIEVAL = sum(1 for t in TASKS if not t["needs_all"])
N_AGGREGATE = sum(1 for t in TASKS if t["needs_all"])


def _result(name: str, w: Window, retrieval: int, aggregate: int) -> dict[str, Any]:
    """Two answer columns, deliberately not one.

    Summing them would report a false equivalence. A RETRIEVAL question is
    answered when the line carrying the answer is visible -- a plumbing property,
    directly comparable across arms. An AGGREGATE question is a property of every
    row: having the bytes in the window is not having the answer, because
    somebody still has to do the arithmetic. Only `stats` actually answers one
    here; OFF and `readback` hold every byte and produce no total, which is the
    honest reading and the reason the two are counted apart.
    """
    return {
        "arm": name,
        "peak_chars": w.chars,
        "peak_tokens": _tok(w.chars),
        "cumulative_chars": w.cumulative,
        "cumulative_tokens": _tok(w.cumulative),
        "round_trips": w.round_trips,
        "retrieval": retrieval,
        "retrieval_of": N_RETRIEVAL,
        "aggregate": aggregate,
        "aggregate_of": N_AGGREGATE,
    }


# --------------------------------------------------------------------- gates
def _gates(off: dict[str, Any], on: dict[str, Any], rb: dict[str, Any],
           reach: dict[str, bool]) -> list[tuple[bool, str]]:
    """Every claim this file makes, as a pass/fail. Exits non-zero on any red."""
    return [
        (on["peak_chars"] < off["peak_chars"],
         "folding shrinks the active context"),
        (on["cumulative_chars"] < off["cumulative_chars"],
         "...and shrinks cumulative prefill, which is the number that compounds"),
        (on["retrieval"] == off["retrieval"] == N_RETRIEVAL,
         "every retrieval answer is still reachable -- folding loses nothing"),
        (on["aggregate"] == N_AGGREGATE and off["aggregate"] == 0,
         "...and `stats` answers the aggregates the raw window never did"),
        (all(reach.values()),
         "every folded body is still addressable by handle"),
        (rb["cumulative_chars"] > on["cumulative_chars"],
         "paging a body back costs more than one scalar (what `stats` is worth)"),
        (on["round_trips"] < rb["round_trips"],
         "...and costs more round trips"),
    ]


def run_sim(scale: str = "small", quiet: bool = False) -> dict[str, Any]:
    corpus = build_corpus(SCALES[scale])
    work = Path(tempfile.mkdtemp(prefix="foldsim-"))
    try:
        off = arm_off(corpus)
        on = arm_on(corpus, work)
        rb = arm_readback(corpus, work)

        # Reachability is asserted directly, not inferred from the answers: a
        # body that is unreachable but happened to have its answer in the head
        # would otherwise pass.
        ledger.configure(work)
        ledger.clear()
        reach: dict[str, bool] = {}
        for name in ("app.log", "sales.csv", "handbook.txt"):
            _d, _r, handle, _s = gate.fold("read_file", corpus[name], THRESHOLD)
            body = ledger.get_blob(handle)
            reach[name] = bool(handle) and body is not None and body["body"] == corpus[name]

        gates = _gates(off, on, rb, reach)
        out = {
            "scale": scale,
            "chars_per_token": CHARS_PER_TOKEN,
            "threshold_chars": THRESHOLD,
            "arms": {"off": off, "on": on, "readback": rb},
            "reachable": reach,
            "gates": [{"ok": ok, "claim": claim} for ok, claim in gates],
            "passed": all(ok for ok, _ in gates),
        }
        if not quiet:
            _report(out)
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _report(r: dict[str, Any]) -> None:
    off, on, rb = r["arms"]["off"], r["arms"]["on"], r["arms"]["readback"]
    print(f"\n{BOLD}  Fold mechanical A/B{RESET}")
    print(f"{DIM}  {len(TASKS)} scripted tasks · {r['scale']} corpus · "
          f"fold above {THRESHOLD:,} chars · no model{RESET}\n")

    print(f"{BOLD}  Active context{RESET}")
    print(f"    {'':<28}{'OFF':>12}{'ON':>12}{'readback':>12}")
    for label, key in (("peak chars", "peak_chars"),
                       ("peak tokens", "peak_tokens"),
                       ("cumulative chars", "cumulative_chars"),
                       ("cumulative tokens", "cumulative_tokens"),
                       ("round trips", "round_trips")):
        print(f"    {label:<28}{off[key]:>12,}{on[key]:>12,}{rb[key]:>12,}")

    cut = 1.0 - on["cumulative_chars"] / max(1, off["cumulative_chars"])
    print(f"\n    {DIM}cumulative prefill cut by {cut:.1%} against OFF{RESET}")

    print(f"\n{BOLD}  Answers{RESET}")
    print(f"    {'':<28}{'OFF':>12}{'ON':>12}{'readback':>12}")
    print(f"    {'retrieval (a line)':<28}"
          f"{off['retrieval']:>7}/{off['retrieval_of']:<4}"
          f"{on['retrieval']:>7}/{on['retrieval_of']:<4}"
          f"{rb['retrieval']:>7}/{rb['retrieval_of']:<4}")
    print(f"    {'aggregate (every row)':<28}"
          f"{off['aggregate']:>7}/{off['aggregate_of']:<4}"
          f"{on['aggregate']:>7}/{on['aggregate_of']:<4}"
          f"{rb['aggregate']:>7}/{rb['aggregate_of']:<4}")
    print(f"\n    {DIM}An aggregate is a property of every row, so holding the bytes is not\n"
          f"    holding the answer -- OFF and readback carry the whole table and still\n"
          f"    produce no total. `readback` spends {rb['cumulative_chars']:,} characters to end up\n"
          f"    exactly where OFF started on that question.{RESET}")

    print(f"\n{BOLD}  Gates{RESET}")
    for g in r["gates"]:
        mark = f"{GREEN}ok  {RESET}" if g["ok"] else f"{RED}FAIL{RESET}"
        print(f"    {mark}  {g['claim']}")

    print(f"\n{BOLD}  What this does and does not show{RESET}")
    print(f"{DIM}    Shown:     with an identical trace, folding puts fewer characters "
          f"through\n"
          f"               the window and far fewer through cumulative prefill, every\n"
          f"               elided body is still addressable, and every scripted answer\n"
          f"               is still reachable. The readback arm prices the case the\n"
          f"               byte-returning verbs lose -- paging a whole body back costs\n"
          f"               more than never folding it -- which is the case `stats`\n"
          f"               exists to answer instead.\n"
          f"    NOT shown: whether a real model CHOOSES the right verb, copies a handle\n"
          f"               back correctly, or reaches the same answer in no more turns.\n"
          f"               This driver is scripted to pick well; a model is not. Those\n"
          f"               are behavioural questions and need a model and your own\n"
          f"               workload to answer.{RESET}")
    if not r["passed"]:
        print(f"\n  {RED}One or more gates failed.{RESET}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fold A/B harness -- no model needed.")
    ap.add_argument("--scale", choices=sorted(SCALES), default="small")
    ap.add_argument("--json", metavar="PATH", help="write the full result bundle here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    r = run_sim(args.scale, quiet=args.quiet)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json}")
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
