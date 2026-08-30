# The memory layer

SQLite, one table, no embedding model. What makes it more than a notes table is
that it grades both ends: what comes back out, and what went in.

```python
from pathlib import Path
from rangerkit import memory

memory.configure(Path.home() / ".rangerkit")
memory.remember("Dave runs the training scripts on the RTX 4090")
memory.recall("what GPU is in the workstation")
```

---

## Storage

`memories(id, category, fact, context, created_at, folder, uses, wins, last_used, unproven, lineage, grafted)`
in `<state_dir>/memory.db`.

The database is opened in WAL so readers do not block the writer, and every
helper opens its own connection. Schema setup runs **once per process** behind a
module flag rather than on every connection; `configure()` resets the flag,
because a new state directory is a different database. Before that, ordinary
recall re-ran `CREATE TABLE` and the migration check against a file other
processes held open — which surfaced as intermittent stalls rather than as an
error.

### Why not embeddings by default

An embedding model competes with the main model for VRAM. For a few hundred
personal facts, keyword scoring is adequate and instant. If you want semantic
recall anyway, register a backend — see *Semantic recall* below — and it is
fused rather than bolted on.

---

## Recall

Weighted keyword overlap over a stop-word-filtered token set, with `id / 1e6` as
a recency tiebreak. `_tokens` drops anything under three letters, which is worth
knowing because it is what the cue check exists to police.

### Scattershot — the read-side answer to encoding specificity

`overlap = len(query_tokens & fact_tokens)` is a literal set intersection, so
*"what GPU do I have"* scores zero against *"Dave runs the training scripts on
the RTX 4090"*. Nothing is wrong with that fact. It is simply not phrased the way
the question was.

The write-side instruction ("include the terms a future question would actually
use") asks the model to guess, at write time, the vocabulary of a question
nobody has asked yet. Scattershot closes the same gap from the other end: one
query is fired as a **spread** of related terms, every one derived from the store
itself, and any fact a pellet touches becomes a candidate.

| pellet | what it is | where it comes from |
|---|---|---|
| **stem** | a stored word differing only by an ending | string shape, with digits explicitly never counting as an ending, so `rtx4090` and `rtx4080` stay two cards |
| **neighbour** | words your own facts keep in the same sentence | corpus co-occurrence, gated at a minimum count |
| **sector** | the routed subject's vocabulary | widest and weakest |
| **bond** | a word that *led* to a specific fact in a turn that then succeeded | earned, not guessed — the only pellet kind that is not a theory about language |

Only words the fact does not itself contain are bonded; a word already in the
sentence needs no bridge to it.

**The choke matters as much as the spread.** A spread wide enough to reach
anything finds nothing, so the fill is bounded: it only ever *appends* to a
result that came back short, never reorders one, never exceeds the per-recall
cap, and is floored relative to the best pellet — because the eighth-best pellet
hit is noise wearing a hit's format. The benchmark asserts all four
([`bench/memsim.py`](../bench/memsim.py)).

### Sector routing

Every fact is filed by subject on write. Routing reads that back at *retrieval*
time, which turns one flat store into a set of sector brains sharing a database.

**Recall ranks by sector; it never filters by one.** The keyword pass scores
every fact and adds a bonus to those in a routed subject. The bonus is
deliberately worth less than a single token of overlap, so it settles ties toward
the routed subject and can never outrank a fact that simply answers better.

> This was a filter until it was measured. Scoring only the routed sectors and
> retrying "if that yields nothing" meant one weak in-sector match suppressed
> every fact outside the routed subjects, however much better it scored. Two
> kinds of fact fell through and the retry could not catch either: a **core**
> fact, in no routed list by definition, and anything **hand-filed** outside the
> standard subjects. Ask "how much VRAM does my GPU have" with the GPU fact
> moved to `Hardware/GPU` and recall answered with a monitor's refresh rate.

`preload()` ships only sector-agnostic facts — the ones that matched no
vocabulary, so they apply whatever the user is doing. `sector_index()` adds one
line naming the subjects that hold the rest, which is what tells the model there
is something to `recall`.

> **Why this is about cache stability.** "The newest N" changes whenever a fact
> is saved, so the next session opened with different bytes and the KV prefix —
> which matches from position 0 — missed on all of it, re-prefilling the whole
> prompt on a runner that was still warm. Core facts and the subject line change
> rarely, so those bytes hold. The test suite asserts the byte-identity directly,
> because it is the kind of property that regresses in silence.

---

## Grading what comes back — reinforcement

Recall is a write. The final fused list marks the row ids it served, and
`reinforce(outcome)` drains that set and bumps `uses`/`wins`/`last_used` when the
turn was not cut short.

`_familiarity` reads the tally back at ranking: a Laplace-shrunk win share
(`wins / (uses + 2)`, strictly under 1), weighted, and halved for every 30 days
unrecalled.

**The bound is structural, not a tuning choice.** Familiarity plus the sector
bonus stays strictly under one token of genuine overlap, so a famous in-sector
fact **orders ties** and never outvotes a better match. And it fades: rank earned
last month is not rank held forever. The test suite asserts the achievable
maximum rather than the constants, because that is the number the guarantee is
actually about.

Deliberately narrow about evidence. Only an explicit `recall` that fed a finished
turn counts: `list_memories` is browsing, the prompt digest rides every turn, and
a stopped run says nothing.

---

## Grading what goes in — seed grading

Reinforcement grades what memory gives back. Nothing graded what it is fed, and a
store is only as good as what was put into it. A fact the model writes mid-turn
was stored at full strength whatever became of that turn, and a sector-less one
then rode the system prompt into every turn after it — a claim from a run that
went wrong, read back as settled truth.

So a write made inside a turn that then **fails** is held back: kept in full,
still returned by `recall`, still listed — but out of the prompt digest until it
has earned its place. `preload` is the only reader.

It releases itself the first time it feeds a turn that works, or the user vouches
for it with `trust_memory`.

**Narrow in four directions**, each for its own reason:

- a **stopped** run holds nothing back — a stop is not evidence;
- a write made **outside a turn** is never held — it is not the model's guess;
- a fact recall merely *served* through a failure is not blamed for it — the
  shrinking win share is the gentle answer there;
- with learning off the flag is inert in **both** directions, because with the
  turn-end hook dark nothing would ever release a held fact.

> **Why this does not churn the KV prefix:** `unproven` moves only when a turn
> fails on a write or a held fact is released — rare, and rarer still twice in
> the same direction. Ordering the digest by anything the tally moves (`wins`,
> say) would rewrite the prompt every turn; the flag is a *filter* for that
> reason.

---

## Grading the form — the cue check

The question that comes before both grades above. Neither asks whether the fact
was written in a form recall can find at all.

That is not a matter of taste. Recall ranks by shared scoring tokens and drops
words under three letters — so a fact whose subject is "he" has no "he" to match
against, and no query about Dave will ever reach it. It is stored, it is counted,
it can ride the prompt digest, and it is invisible to the one mechanism meant to
surface it on merit.

This is the encoding specificity principle (Tulving & Thomson, 1973) stated in
SQL: retrieval works to the extent that the cues present when a fact is read back
were encoded when it was written. The model writing the fact is the only party
that ever holds both ends of that, and it holds them for one turn — which is why
the check runs at **write** time, where the referent is still on screen, rather
than at recall, where the evidence of what went wrong is exactly what is missing.

**It reports; it does not refuse.** Nothing is rejected, nothing is reworded
behind the model's back, and a flagged fact is stored, recallable and preloaded
like any other. The whole intervention is one extra line on the tool result —
which is also the strongest instruction available, because it arrives at the one
moment the gap can still be closed by the one party who can close it.

---

## Dedup, polarity, and provenance

**Dedup.** Jaccard overlap above 0.75 with an existing fact updates that row
instead of inserting. The update **keeps** the earned tally, so a re-wording does
not cost a fact the rank it earned.

**Unless the re-wording reverses it.** High token overlap means the same *words*,
not the same *claim*: inserting one `not` into a long sentence flips it while
still scoring over 0.9, and that was the path by which the exact opposite of a
fact inherited its evidence and arrived at the top of the next recall carrying
the maximum boost the system can grant. Polarity is counted separately, on the
raw text (so `no` survives the three-letter filter, `doesn't` folds onto
`doesnt`, and a double negative reads as an assertion). When it differs the newer
wording still wins the slot — a correction is meant to correct — but the tally
resets and the new claim earns its rank from scratch.

**Provenance.** Overlap alone cannot tell *where* a claim came from, and three
failures follow from that one gap: a mid-turn guess overwriting a fact the user
vouched for and inheriting its proven turns; the same agent restating a fact in
fewer words landing a *second* row; and two independent witnesses collapsed on
word overlap alone.

So a fact records the lineage it grew on, as a **path** read off what the process
already knows rather than off anything the writer says about itself:

| part | what it is |
|---|---|
| `tree` | what bore it at the root — `user`, `machine`, `model`, or `import` |
| `branch` | the run of work it grew on. A sub-agent **shares** its parent's branch |
| `stem` | which agent on that branch bore it |

Two rules act on it. **A restatement folds back into its own row** when the new
fact's tokens are a proper subset of a stored one's *and* the two share a stem or
branch — and the fuller text is the one kept. Containment alone never licenses
the fold: without kinship, a broad fact and a narrow one are two witnesses. And
**a cross-tree takeover inherits no proof**: the words win the slot, but the
tally survives only if the newcomer's tree is standing at least as well.

Standing **counts, never declares**: a Laplace-shrunk win share over the tree's
facts, scaled by the share not currently held out of the prompt. Nothing in the
module knows that a user is more reliable than a model — a standing asserted
rather than measured would let any writer claim its way to the top. A tree with
no finished fruit is *absent* from the result rather than scored zero: *has borne
nothing yet* is not *has borne badly*, and a check that cannot tell them apart
demotes every first write in a fresh store.

---

## Semantic recall

None is bundled and the seam is empty by default. Register a backend and its hits
are **fused** with the keyword hits by reciprocal rank fusion:

```python
from rangerkit import semantic

class MyBackend:
    def configure(self, state_dir, cfg): ...          # optional
    def mirror_fact(self, mem_id, category, fact, created_at): ...  # optional
    def remove_fact(self, mem_id): ...                # optional
    def search(self, query, limit=5):                 # required
        return [(mem_id_or_None, "matching text"), ...]

semantic.register(MyBackend())
```

RRF works on ranks, so an overlap count and a cosine score never need a shared
scale, and a backend that returns garbage costs at most its share of the slots.
Both retrievers compete for the same `recall_limit` slots, which is what lets
semantic recall *shrink* what is sent to the model rather than only ever adding
to it.

The semantic pass is deliberately **not** sector-scoped, unlike the keyword pass.
It is already relevance-ranked, so it does not have the problem scoping solves —
and letting it cross sectors is what makes recall associative.

SQLite stays the source of truth in every configuration. Every call is wrapped
and every failure degrades silently to keyword-only.

---

## The second brain

`rangerkit.brain` stores what the agent *does*, as opposed to what it was told:
trigger phrases, the resolution that worked, and a use/win tally that becomes a
confidence score. Three tables in the same database, so the two can be linked.

| table | what it holds |
|---|---|
| `patterns` | learned task shortcuts — reinforced when a matching request succeeds, pinned when taught by hand |
| `episodes` | a compact log of past turns, so a similar task can be looked back at |
| `associations` | weighted links between patterns that fire together |

`preload()` folds the top shortcuts into the system prompt at session start,
which is where the saving comes from: the model is handed the answer instead of
re-deriving it every time.

---

## Configuration

`memory.configure(state_dir, cfg)` takes a plain dict. Every key has a default
and every mechanism can be switched off.

| key | default | what it does |
|---|---|---|
| `recall_limit` | `8` | hard cap on facts returned per recall |
| `scattershot` | `True` | the spread. Reads only the store, so it is *not* gated on learning |
| `scatter_bonds` | `True` | the earned pellet kind — gated on reinforcement, since the same hook writes them |
| `sector_routing` | `True` | rank by routed subject; `False` searches everything and preloads the newest N |
| `reinforcement` | `True` | the use/win tally |
| `hold_unproven` | `True` | seed grading. Inert with reinforcement off, in both directions |
| `cue_check` | `True` | the write-time referent check |
| `provenance` | `True` | lineage, restatement folding, cross-tree takeover |
| `fusion` | `"rrf"` | `"keyword_only"` disables the semantic pass entirely |
| `preload_core` | `8` | sector-agnostic facts in the prompt digest |

---

## See also

- [FOLDING.md](FOLDING.md) — the other half of the kit
- [BENCHMARKS.md](BENCHMARKS.md) — how these claims are measured
- [`examples/01_memory_basics.py`](../examples/01_memory_basics.py) — runnable
