<div align="center">

<img src="assets/ranger-icon.png" alt="Ranger" width="360">

**Cross-session memory and context folding for local agents.**
SQLite and the standard library. No embedding model, no vector store, no network.

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/self--test-33%2F33-brightgreen)](tests/selftest.py)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](#requirements)

</div>

---

## What it is

Two mechanisms an agent needs and most agent frameworks do not have, extracted
from [Ranger](https://github.com/kryptographer/ranger) and packaged to stand on
their own. One idea connects them: **an agent's useful context is far larger
than the window it is allowed to think in, and both halves of that gap can be
closed on the machine rather than in the prompt.**

**`rangerkit.memory` — what survives the session.** Facts written by the agent
or the user, stored in SQLite, ranked by keyword overlap and a spread of related
terms derived from your own store. What makes it different from a notes table is
that it *grades itself*: a fact that fed a turn which worked earns a tally that
breaks ties at the next recall, a fact written during a turn that failed is kept
in full but held out of the system prompt until it earns its place, and a fact
written in a form recall could never find is reported at the moment it is
written — the one moment the gap can still be closed.

**`rangerkit.fold` — what never enters the window at all.** A large tool result
goes into a local ledger under a handle, the model receives a shape digest, and
three read-back verbs reach the parts it turns out to need. On a 4,000-line CSV
that is 89,352 characters replaced by 657 — and because a conversation is
re-sent on every subsequent step, it is a saving on every one of them.

**`bench/` — the part that makes either claim checkable.** Four harnesses, all
deterministic, none needing a model. They report what the mechanism does *and*
what it costs, including the case the fold loses.

Nothing here reaches the network, loads a model, or requires one to be running.

---

## Install

```bash
pip install rangerkit
```

Or from source, which is also how you get the benchmarks and examples:

```bash
git clone https://github.com/kryptographer/ranger.git
cd "ranger/Ranger Release Repo"
pip install -e .
```

There is nothing else to install. See [Requirements](#requirements).

---

## Quickstart

### Memory

```python
from pathlib import Path
from rangerkit import memory

memory.configure(Path.home() / ".rangerkit")

memory.remember("Dave runs the training scripts on the RTX 4090")
memory.remember("Production deploys need a manual approval from Priya")

print(memory.recall("who approves deploys"))
# #2 [fact] Production deploys need a manual approval from Priya  (2026-08-30)

print(memory.preload())      # the sector-agnostic facts, for your system prompt
```

### Folding

```python
from pathlib import Path
from rangerkit.fold import gate, ledger

ledger.configure(Path.home() / ".rangerkit")

digest, removed, handle, _ = gate.fold("read_file", big_csv, threshold=2000)

print(digest)                                    # 657 chars, not 89,352
print(gate.grep(handle, "FATAL"))                # the lines that match
print(gate.peek(handle, start=2500, count=10))   # a slice, by line
print(gate.stats(handle, column="amount", op="sum"))   # a total, no bytes back
```

Three runnable examples are in [`examples/`](examples/) — including
[the full agent-loop wiring](examples/03_wiring_into_an_agent.py), which is six
function calls.

---

## The memory layer

A store is only as good as what goes into it and what comes back out, and the
mechanisms below grade both.

| Mechanism | What it does |
|---|---|
| **Scattershot recall** | Fires one query as a spread of related terms — conjugations, words your own facts keep in the same sentence, the routed subject's vocabulary, and words that *led* to a fact in a turn that worked. A question worded differently from the fact still reaches it. |
| **Reinforcement** | Recall is a write. A fact that served a turn which succeeded earns a use/win tally; a Laplace-shrunk win share breaks ties at the next recall, and halves for every 30 days unrecalled. Bounded strictly below one token of real overlap, so familiarity **orders** what already matched and can never outvote a better match. |
| **Seed grading** | The same idea pointed at what goes *in*. A fact written during a turn that then failed is kept in full, still recalled, still listed — but withheld from the system-prompt digest until it feeds a turn that works or the user vouches for it. |
| **Cue check** | Encoding specificity (Tulving & Thomson, 1973) stated in SQL. A fact whose subject is "he" has no "he" to match against, so no question about Dave will ever reach it. Reported at write time, on the tool result. Nothing is refused and nothing is reworded. |
| **Provenance** | A fact records the lineage it grew on — `tree/branch/stem`. A restatement folds back into its own row; a claim taking over a row grown on another tree wins the slot but inherits none of the proof that tree earned. Standing is *counted*, never declared. |
| **Sector routing** | Facts are filed by subject on write, and recall adds a bonus to the routed subject — deliberately worth less than one token of overlap, so it **ranks and never filters**. `preload()` ships only sector-agnostic facts, which is what keeps the prompt's bytes stable across sessions. |
| **Dedup** | Jaccard overlap above 0.75 updates the existing row instead of adding a near-duplicate — keeping its earned tally, *unless* the re-wording reverses the claim. One inserted "not" scores over 0.9 while meaning the opposite, so polarity is checked separately and a reversal resets the evidence. |

Full detail, including why each bound is where it is: **[docs/MEMORY.md](docs/MEMORY.md)**.

### Semantic recall is a seam, not a dependency

An embedding model competes with your main model for VRAM, which is the whole
reason keyword scoring is the default. So none is bundled. Register any
retriever you like and its hits are **fused** with the keyword hits by
reciprocal rank fusion — both compete for the same `recall_limit` slots, so
semantic recall can *shrink* the tokens sent to the model by displacing a weak
keyword hit, rather than only ever adding to them.

```python
from rangerkit import semantic

class MyBackend:
    def search(self, query, limit=5):
        return [(mem_id_or_None, "the matching text"), ...]

semantic.register(MyBackend())
```

SQLite stays the source of truth in every configuration, and every failure path
degrades silently to keyword-only.

---

## The fold

The claim is deliberately narrow, because the tempting version of it is false.
This does **not** turn a 32K model into a 320K model. It separates two things
that are usually identical:

| | |
|---|---|
| **Logical context** | Everything the agent can reach — files, command output, folded results, the ledger. Unbounded. Free. |
| **Active context** | What is actually inside the model's attention window on this request. Bounded. Expensive. |

Three verbs, and the split between them is the design:

| | | |
|---|---|---|
| `fold_peek` | a slice of the body | bytes back |
| `fold_grep` | the lines that match | bytes back |
| `fold_stats` | a total over **all** of it | a number back |

The first two are the same bet — that the model wants a few lines out of many.
Where it holds, folding wins enormously. Where it fails it fails *unboundedly*:
a sum, a row count, the commonest value is a property of every row, so the only
way to answer with the first two is to page the whole body back with the
conversation re-sent at every hop. `fold_stats` is the case that bet cannot
cover — the body is already on your machine, where arithmetic is free, so the
arithmetic happens there and only the scalar crosses.

It reports what it actually did (rows scanned, rows matched, cells that would
not parse as numbers), and it **refuses rather than guesses**: a sum over no
numeric values returns *"no numeric values in that column"*, never `0`.

Full detail, including the fail-open discipline and the injection surface
folding creates: **[docs/FOLDING.md](docs/FOLDING.md)**.

---

## Benchmarks

Every harness is deterministic, runs with no model and no network, and reports
what the mechanism costs alongside what it saves. Run them all with one command:

```bash
python bench/report.py            # writes bench-results/rangerkit-bench.{json,md}
```

### The fold, measured

Same scripted tool-call trace in every arm; the only variable is what the fold
does to each result.

| | fold off | **fold on** | read-back only |
|---|---:|---:|---:|
| peak active context | 824,270 | **5,251** | 212,422 |
| cumulative prefill | 3,313,226 | **33,301** | 4,780,994 |
| round trips | 6 | **9** | 49 |
| retrieval answers | 4/4 | **4/4** | 4/4 |
| aggregate answers | 0/2 | **2/2** | 0/2 |

<sub>`python bench/foldsim.py --scale medium`, characters. Cumulative prefill is
the number that compounds: a conversation is re-sent in full on every later step,
so a result read at step 3 is paid for again at steps 4 through N.</sub>

**Cumulative prefill cut by 99.0%**, and it holds as the corpus grows —
92.1% at `small`, 99.8% at `large`, where the fold-on column barely moves
(32,346 → 34,207 characters) while fold-off climbs to 16.8 million.

The third column is the fold's losing case, priced rather than hidden. Answering
a question about every row by paging the body back costs **more than never
folding it** — 4.8M characters against 3.3M, and 109M against 16.8M at `large`,
because the conversation is re-sent at every hop. That is what `fold_stats`
exists to replace, and the gap between columns two and three is what it is worth.

The aggregate row is counted separately from the retrieval row on purpose:
holding the bytes is not holding the answer. Fold-off carries the whole table and
still produces no total.

### Recall, measured on an unrigged set

40 facts written first, 40 questions written afterwards as a person would type
them — *not* reverse-engineered from the mechanism — plus 5 questions the store
genuinely cannot answer, as false-positive controls.

| | scattershot off | scattershot on | on + earned bonds |
|---|---:|---:|---:|
| correct fact somewhere in results | 31/40 | 35/40 | **36/40** |
| correct fact ranked first | 27/40 | 29/40 | 29/40 |
| answered an unanswerable question | 1/5 | 1/5 | 1/5 |
| total result lines (prompt cost) | 60 | 89 | 101 |

<sub>`python bench/recallrate.py`. Hit rate 78% → 88% → **90%**.</sub>

The last two rows are the honest half. The false-positive control does not move,
so the spread is not buying its hit rate by answering more often. And the spread
is **not free**: it fills slots that came back empty, so an average result gets
longer even though the per-recall cap never moves.

### What none of this shows

These are **mechanical** results: with a fixed workload, the mechanisms do what
they claim. They say nothing about whether a real model asks recall better
questions, writes better facts when told one cannot be found, or picks the right
read-back verb. Those are behavioural, they depend on the model and on your own
data, and no arrangement of these harnesses can settle them. Watch the tallies on
your own store.

More on method and how to read the numbers: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

---

## Testing

```bash
python tests/selftest.py     # 33 checks, standalone, no dependencies
pytest tests/selftest.py     # the same checks as pytest cases
```

Real SQLite in a temp directory, nothing mocked, under a second. The suite
asserts the invariants the mechanisms are only useful if they hold — that
familiarity stays bounded below a token of overlap, that a held fact is still
recallable, that the prompt digest is byte-stable across an ordinary write, that
a fold which would inflate is refused *and* un-holds its blob, that handles are
never reused, that an expired handle says it was held rather than answering with
silence, and that a corrupt database is quarantined rather than fatal.

---

## Requirements

**Python 3.10 or newer. Nothing else.**

No runtime dependencies, and that is a design constraint rather than an
accident: the mechanisms exist to make a *local* agent workable, and a memory
layer that pulls in a model server has given back what it was supposed to save.
Everything is `sqlite3`, `re` and `pathlib` from the standard library.

`pytest` is optional and only for running the test suite under pytest; the same
suite runs standalone with no dependencies at all.

---

## Docs

| | |
|---|---|
| [docs/MEMORY.md](docs/MEMORY.md) | The memory layer: schema, every grading mechanism, and why each bound is where it is |
| [docs/FOLDING.md](docs/FOLDING.md) | The fold: the ledger, the three verbs, fail-open discipline, and the injection surface |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | What each harness measures, what it deliberately does not, and how to read the output |
| [examples/](examples/) | Three runnable files, ending with the full agent-loop wiring |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: a change to a
mechanism needs a check in `tests/selftest.py` that fails without it, and a
change to a *bound* needs a number from `bench/`.

Security policy: [SECURITY.md](SECURITY.md).

---

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

## About

Extracted from [Ranger](https://github.com/kryptographer/ranger), a local-first
desktop agent, and released on its own because these two mechanisms are useful
well outside it. The app is not included here and is not needed — this is the
memory layer, the fold, and the harnesses that measure them.
