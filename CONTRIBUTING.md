# Contributing

Thanks for looking. This is a small, deliberately dependency-free library, and
the rules below are what keep it that way.

## The short version

- **A change to a mechanism needs a check in `tests/selftest.py` that fails
  without it.** Not a check that passes with it — one that catches its absence.
- **A change to a *bound* needs a number from `bench/`.** The constants in this
  library (0.75 for dedup, the sector bonus, the reinforcement weight and
  half-life) are load-bearing, and each one has a reason written next to it.
  Moving one without a measurement replaces a reason with a preference.
- **No runtime dependencies.** See below.

## Running everything

```bash
python tests/selftest.py       # 33 checks, no dependencies, under a second
python bench/report.py         # every harness, bundled
```

Both must be green before a change is worth reviewing.

## No runtime dependencies

This is a constraint, not a preference. The mechanisms exist to make a *local*
agent workable, and a memory layer that pulls in a model server has given back
what it was supposed to save. `sqlite3`, `re` and `pathlib` are the budget.

If you want semantic recall, register a backend through
[`rangerkit/semantic.py`](rangerkit/semantic.py) — that seam exists precisely so
the dependency lives in your project rather than in this one.

## Things that will be asked about in review

**Does the failure path fail open?** A folding layer that drops a tool result on
a database hiccup is worse than no folding layer. A memory layer that refuses to
start on a corrupt database makes the whole host unlaunchable. Both modules take
this seriously and a change should too.

**Does it report, or does it refuse?** The cue check reports and stores anyway.
`fold_stats` refuses rather than returning `0`. Those are different answers to
different questions and both were deliberate — a zero with a confident face on is
a wrong answer, and a rejected fact is a fact the user lost.

**Does it count, or does it declare?** Nothing here knows that a user is more
reliable than a model; standing is measured from outcomes. A ranking asserted
rather than measured lets any writer claim its way to the top.

**What does it cost?** Every mechanism in this library has a cost written down
next to its benefit — scattershot makes results longer, the fold adds round
trips. A change that reports only the benefit is incomplete.

**Is a comment explaining a decision, or narrating the code?** The comments here
are dense because they record *why* a bound is where it is and what broke before
it was. Keep that; drop anything that restates what the line already says.

## Style

- Match the surrounding code. It is stdlib Python with full type annotations and
  `from __future__ import annotations` everywhere.
- Exemptions state their reason. A silenced check without one is a standing
  permission for the next occurrence.
- No new files unless a mechanism genuinely needs one.

## Reporting a bug

Include the output of `python bench/report.py` if the bug is behavioural — it
carries your platform, Python version and the revision, which is most of what a
first reply would otherwise have to ask for.

Security issues: see [SECURITY.md](SECURITY.md) — please do not open a public
issue for those.
