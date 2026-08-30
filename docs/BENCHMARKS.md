# Benchmarks

Four harnesses. All deterministic, all stdlib-only, none needing a model or a
network. Run them all with one command:

```bash
python bench/report.py                 # writes bench-results/rangerkit-bench.{json,md}
python bench/report.py --scale medium  # the fold bench at 8x
python bench/report.py --quick         # self-test and the fold bench only
```

Or individually:

```bash
python bench/foldsim.py --scale medium   # the fold, mechanical A/B
python bench/memsim.py                   # the memory mechanisms, gated A/B
python bench/recallrate.py               # an unrigged recall-rate measurement
python tests/selftest.py                 # 33 invariant checks
```

---

## The split every harness here observes

The claims these mechanisms make are of two kinds, and they need very different
evidence:

> **MECHANICAL** — with the same workload, does the mechanism do what it says?
> Deterministic. No model needed. **Settled here.**
>
> **BEHAVIOURAL** — does a real model ask recall better questions, write better
> facts when told one cannot be found, pick the right read-back verb, get the
> answer in no more turns? Depends on the model and on your own data. **Not
> settled here, and no arrangement of these files can settle it.**

Conflating the two is the standard way this class of work fools itself: a harness
proves the bytes moved and then reports it as though the answers held. Every
harness in this directory prints what it does and does not show, at the bottom of
its own output, for that reason.

---

## The fold

`bench/foldsim.py` replays a **fixed** tool-call trace, identical in every arm, so
the only variable is what the fold does to each result. It answers each question
by searching the text *actually visible* in its window — it cannot use knowledge
it was not shown. Correctness here is therefore a property of the plumbing (was
the answer-bearing line still reachable?) rather than of a policy the harness
invented for itself.

### Three arms

| arm | what it is |
|---|---|
| `off` | no fold. Every result lands in the window whole and stays there |
| `on` | fold, then read back with whichever verb the question needs |
| `readback` | the same fold, but scalar questions answered *only* by paging the body back — the fold's losing case, priced rather than hidden |

### Results

| | fold off | **fold on** | read-back only |
|---|---:|---:|---:|
| peak active context | 824,270 | **5,251** | 212,422 |
| cumulative prefill | 3,313,226 | **33,301** | 4,780,994 |
| round trips | 6 | **9** | 49 |
| retrieval answers | 4/4 | **4/4** | 4/4 |
| aggregate answers | 0/2 | **2/2** | 0/2 |

<sub>`--scale medium`, characters.</sub>

Across scales, the fold-on column barely moves while fold-off tracks the corpus:

| scale | fold off (cumulative) | fold on | cut |
|---|---:|---:|---:|
| `small` | 407,811 | 32,346 | 92.1% |
| `medium` | 3,313,226 | 33,301 | 99.0% |
| `large` | 16,781,778 | 34,207 | 99.8% |

### Why cumulative prefill, and not peak

A conversation is re-sent in full on every subsequent step, so a result read at
step 3 is paid for again at steps 4 through N. Peak window size understates the
cost of *not* folding by that factor. Cumulative prefill is the number that
compounds, and it is reported beside the peak rather than instead of it.

Offline this is more load-bearing than in the cloud, not less. A runner that
keeps a KV cache per slot re-uses the longest matching prefix, and tokens inside
it are not re-evaluated — so a conversation whose prefix stopped matching pays
for it in prompt evaluation on the GPU, on every step, reported by nothing. It
does not look like a bug. It looks like "local models are slow."

### The arm that keeps this honest

`readback` costs **more than never folding**: 4.8M characters against fold-off's
3.3M at `medium`, and 109M against 16.8M at `large`, because the conversation is
re-sent at every one of its 209 hops. That is the real shape of the failure, and
the gap between the `on` and `readback` columns is exactly what `fold_stats` is
worth.

### Why the two answer rows are counted separately

Summing them would report a false equivalence. A **retrieval** question is
answered when the line carrying the answer is visible — a plumbing property,
directly comparable across arms. An **aggregate** question is a property of every
row: having the bytes in the window is not having the answer, because somebody
still has to do the arithmetic. Fold-off carries the whole table and produces no
total.

### The checker cannot be satisfied by coincidence

An aggregate probe whose expected answer already occurs somewhere in the raw data
would score a point for an arm that computed nothing — a row count of "500" is a
substring of row id 500. So the corpus builder **asserts** that no expected answer
appears in the corpus text, and refuses to run rather than report a number it
cannot stand behind.

---

## The memory mechanisms

`bench/memsim.py` is a gated A/B: the workload is fixed and identical in both
arms, and the only variable is the mechanism under test. Every claim is a
**gate** — the harness exits non-zero when one fails, so a regression turns the
bench red rather than shading a number nobody reads.

It covers reinforcement (do ties reorder toward the fact actually used; is the
guard real; does it fade), the sleep pass (is every planted habit promoted and
nothing else), the cue check (is a fact recall could never find caught, without
flagging ones it can), provenance (does a cross-tree takeover stop inheriting
proof), and scattershot (do all four pellet kinds bridge, and does the choke
hold).

The scattershot section reports the cost as well as the benefit: the spread fills
slots that came back empty, so an average result gets **longer** even though the
per-recall cap never moves.

### The arm that needs restating

The encoding arms are scripted to write the vague fact and then write it again
spelled out — a model that always takes the correction. **That is not evidence a
model does.** It is the setup under which the mechanical question becomes
answerable at all: what the correction costs when it lands, and where it lands
when nothing asked for it.

---

## Recall rate

`bench/recallrate.py` is the least flattering harness here, by construction.

1. Facts written first, in the voice a real store accumulates them in.
2. Questions written **afterwards** as a person would type them — *not* reverse
   engineered from the pellet kinds. Some share words with their fact, some do
   not; that mix is the thing being measured.
3. Each question tagged with the id of the fact that answers it, or `None` when
   the store genuinely cannot answer — those are the false-positive controls.
4. Same store, same questions, one variable.

| | scattershot off | scattershot on | on + earned bonds |
|---|---:|---:|---:|
| returned anything at all | 34/40 | 37/40 | 38/40 |
| correct fact somewhere in results | 31/40 | 35/40 | **36/40** |
| correct fact ranked first | 27/40 | 29/40 | 29/40 |
| answered an unanswerable question | 1/5 | 1/5 | 1/5 |
| total result lines (prompt cost) | 60 | 89 | 101 |

Hit rate 78% → 88% → **90%**. Top-1 68% → 72%.

**The last two rows are the point.** The false-positive control does not move, so
the spread is not buying its hit rate by answering more often. And the cost row
says plainly that a result gets longer.

> **The bias that remains:** one author wrote both halves. Treat this as an
> indication, not a measurement of anyone's real store. The honest version of
> this measurement is the tally on your own facts.

---

## The self-test

`tests/selftest.py` is not a benchmark and has no numbers to argue with. It
asserts the invariants the mechanisms are only useful if they hold:

- familiarity stays bounded strictly below one token of overlap, at the
  *achievable* maximum rather than the constants;
- a held fact is still stored in full and still recallable;
- the prompt digest is byte-identical across an ordinary write;
- a fold that would inflate is refused **and** un-holds its blob;
- handles are never reused, even after their blob is dropped;
- an expired handle says it was held, rather than answering with silence;
- `fold_stats` refuses rather than returning a confident zero;
- a catastrophic regex is refused, not run;
- a corrupt database is quarantined, with its WAL sidecars, rather than fatal.

33 checks, real SQLite, under a second.

```bash
python tests/selftest.py     # standalone
pytest tests/selftest.py     # same checks as pytest cases
```

---

## Reading someone else's report

`bench/report.py` bundles results with the platform, Python version, package
version and git revision — including a `-dirty` marker when the checkout has
uncommitted changes. A benchmark whose provenance is unknown is worth less than
one that says so.

The report contains no file contents from your machine: only benchmark numbers, a
platform string, and two version strings. Nothing is uploaded; both files are
written locally and that is the end of it.
