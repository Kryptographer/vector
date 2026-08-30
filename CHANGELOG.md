# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-30

First public release. The memory layer and the fold, extracted from Ranger and
packaged to stand on their own.

### Added

**Memory (`rangerkit.memory`)**
- SQLite fact store with weighted keyword recall and a recency tiebreak.
- **Scattershot recall** — one query fired as a spread of stem, neighbour,
  sector and earned-bond pellets, with a bounded fill that only ever appends to
  a short result and never exceeds the per-recall cap.
- **Reinforcement** — a use/win tally earned by facts that fed turns which
  succeeded, read back as a Laplace-shrunk win share bounded strictly below one
  token of overlap and halved every 30 days unrecalled.
- **Seed grading** — a fact written during a failed turn is kept in full and
  still recallable, but withheld from the prompt digest until it earns its place
  or the user vouches for it.
- **Cue check** — a write-time report on a fact whose referent is missing.
  Reports; never refuses or rewords.
- **Provenance** — `tree/branch/stem` lineage, restatement folding, and
  cross-tree takeovers that win the slot without inheriting the proof.
- **Sector routing** — facts filed by subject on write; recall ranks by the
  routed subject and never filters by it. `preload()` ships only sector-agnostic
  facts, which keeps the system prompt byte-stable across sessions.
- **Dedup with polarity** — Jaccard above 0.75 updates rather than inserts,
  keeping the earned tally, unless the re-wording reverses the claim.

**Second brain (`rangerkit.brain`)**
- Learned task shortcuts, episodes and associations in the same database, with a
  consolidation pass that promotes repeated habits.

**Fold (`rangerkit.fold`)**
- `gate.fold` — large tool results replaced by a shape digest, body held locally
  under a monotonic never-reused handle.
- `fold_peek`, `fold_grep`, `fold_stats` — the three read-back verbs, registered
  disabled and switched on with the mode.
- `codec` — single-source shape detection, including gutter stripping that
  discriminates on whether the numbers *count*.
- `ledger` — TTL sweep that demolishes the body and keeps the survey, so an
  expired handle answers "held, then cleared" rather than with silence.
- Fail-open on every path, including refusing a digest that would inflate — and
  un-holding its blob when it does.
- `sanitize`, `nested_quantifier`, `Redactor`, `trim`, `evict_recalls` for the
  surface folding creates.

**Semantic seam (`rangerkit.semantic`)**
- A pluggable backend protocol with reciprocal rank fusion. Empty by default;
  nothing is bundled and nothing reaches the network.

**Harnesses**
- `bench/foldsim.py` — mechanical A/B of the fold with three arms, including the
  read-back arm that prices the fold's losing case.
- `bench/memsim.py` — gated A/B of every memory mechanism.
- `bench/recallrate.py` — unrigged recall-rate measurement with false-positive
  controls.
- `bench/report.py` — one command, every harness, bundled with provenance.
- `tests/selftest.py` — 33 invariant checks, standalone or under pytest.

### Notes

- Requires Python 3.10+. **No runtime dependencies.**
- Read-back tools are named `fold_peek` / `fold_grep` / `fold_stats` in this
  package; they ship as `vector_*` inside the Ranger application.
