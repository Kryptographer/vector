"""Memory recall A/B without a model: measure the dream-recall layer, never the model.

Same split `foldsim.py` draws for the fold, for the same reason. The
reinforcement and consolidation work makes claims of two different kinds, and
they need very different evidence:

    MECHANICAL   with the same store and the same scripted workload, does the
                 reinforcement knob reorder ties toward the fact actually
                 used, stay strictly under one token of real overlap, fade
                 with disuse -- does the sleep pass promote every planted
                 habit and nothing else -- and does the cue check catch a fact
                 recall could never find, without flagging ones it can?
                 Deterministic. No model needed. Settled here.

    BEHAVIOURAL  does a real model, over your real facts, ask recall better
                 questions, WRITE better facts when told one cannot be found,
                 and give better answers because of either? Depends on the
                 model and on what your store actually holds. NOT settled
                 here, and no arrangement of this file can settle it -- watch
                 the tallies on your own store (`memory.all_memories()`,
                 `memory.orchard()`) for that.

The encoding arm below is the one place that split needs restating, because
its workload PRESUMES the behavioural half. Both arms are scripted to write
the vague fact and then write it again spelled out -- a model that always
takes the correction. That is not evidence a model does; it is the setup
under which the mechanical question becomes answerable at all, which is what
the correction COSTS when it lands, and where it lands when nothing asked
for it.

The workload is FIXED and identical in both arms; the only variable is
`reinforcement` in the config. Every claim below is a gate: this exits 0 only
when all of them hold, so a regression in the mechanism turns the bench red
rather than shading a number nobody reads.

Scattershot adds a third question to that split, and it is the one that comes
FIRST: reinforcement decides which matching fact ranks highest, the cue check
decides whether a fact was written so it can match at all -- and neither helps
when the fact is fine and the QUESTION arrived in different words. Its arms are
mechanical in the same sense: fixed workload, one variable (`scattershot`), and
a guard beside every recovery number, because a spread that answers everything
has not found anything.

Provenance is the fourth question, and it is the only one about a COLLISION.
The three above each grade a fact on its own: which of the matching ones ranks
first, whether it was written so anything could match it, whether a differently
worded question reaches it. None of them says which of two near-identical facts
is kept, or what the survivor inherits from the one it replaced. Its arms are
mechanical in the same sense -- fixed workload, one variable (`provenance`) --
and both fixtures are built to sit where the ratio bar cannot help: one pair
below it (a restatement the threshold will never fold at any setting) and one
above it (a takeover that merges in both arms, so the only thing left to
measure is the tally).

    python memsim.py                  # small store, both arms
    python memsim.py --scale medium   # 4x pairs and habits
    python memsim.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rangerkit import brain as brainmod  # noqa: E402
from rangerkit import memory as memmod  # noqa: E402
from rangerkit.registry import registry  # noqa: E402

DIM, GREEN, YELLOW, RED, BOLD, RESET = (
    "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[0m")

SCALES: dict[str, dict[str, int]] = {
    "small": {"pairs": 12, "credits": 4, "habits": 5, "noise": 6},
    "medium": {"pairs": 48, "credits": 4, "habits": 12, "noise": 24},
}


def _top(query: str) -> str:
    """The single line recall ranks first, with its served mark drained.

    Probes must be non-events: `reinforce("stopped")` empties the served set
    without crediting anything, the same contract a cut-short turn has, so
    measuring the ranking can never change it.
    """
    line = memmod.recall(query, limit=1).splitlines()[0]
    memmod.reinforce("stopped")
    return line


def _build_pairs(n: int) -> None:
    """n sibling pairs: an older fact the workload uses, a newer rival.

    Per pair the two facts tie exactly on the tie probe (three shared tokens
    each), so with reinforcement off the recency tiebreak picks the newer
    rival every time -- the pre-feature behaviour the ON arm is measured
    against.
    """
    for i in range(n):
        memmod.remember(f"the site{i:02d} panel{i:02d} checklist lives in the blue binder", "fact")
        memmod.remember(f"the site{i:02d} panel{i:02d} manual lives in the gray drawer", "fact")


def _credit_pairs(n: int, rounds: int) -> None:
    """The scripted habit: each round recalls every pair's checklist fact
    (limit=1, so exactly that fact is served) and ends the turn well."""
    for _ in range(rounds):
        for i in range(n):
            memmod.recall(f"site{i:02d} panel{i:02d} checklist blue binder", limit=1)
        memmod.reinforce("ok")


def _rank_arm(n: int) -> dict[str, int]:
    """Tie and guard probes against whichever store is configured."""
    tie = sum(1 for i in range(n)
              if "checklist" in _top(f"site{i:02d} panel{i:02d} lives"))
    guard = sum(1 for i in range(n)
                if "manual" in _top(f"site{i:02d} panel{i:02d} manual gray"))
    return {"tie_habitual_first": tie, "guard_better_match_first": guard}


def _vague(i: int) -> str:
    """A fact as it falls out of a conversation: true, useful, and about nobody.

    Exactly three scoring tokens, which is not decoration. A pure addition
    scores |old| / (|old| + 1) on `remember`'s overlap ratio, so at three
    tokens the named rewrite lands on 0.75 -- and the dedup bar is `> 0.75`.
    Three tokens is therefore the widest case the ratio misses by the smallest
    margin, i.e. the one where a correction quietly becomes a duplicate. Longer
    facts are caught by the ratio and never reach the containment path at all.
    """
    return f"he prefers the sans{i:02d} typeface"


def _named(i: int) -> str:
    """The same claim after the model is told the first version names nobody."""
    return f"marlow{i:02d} prefers the sans{i:02d} typeface"


def _control(i: int) -> str:
    """A well-formed fact. Must draw no complaint in either arm.

    The false-flag guard, and the counterpart to the reinforcement arm's
    "better match beats fame": a check that fires on everything measures
    nothing, and would teach a real model to skip the line.
    """
    return f"the sans{i:02d} typeface renders badly on the studio projector"


def _encoding_arm(work: Path, n: int, check: bool) -> dict[str, int]:
    """The identical write workload under one setting of `cue_check`.

    Probe families first (they must not crowd the newest-N digest window),
    then the pairs interleaved -- vague, then named, i by i -- because that is
    the order the correction actually arrives in: the model saves the fact as
    it came up and fixes it in the same breath, not in a batch weeks later.
    Ordering matters to exactly one metric here, and it is the one the digest
    slots are counted in.
    """
    memmod.configure(work, {"backend": "sqlite", "cue_check": check})
    poor = sum(1 for i in range(n)
               if "NOTE:" in memmod.remember(f"agreed{i:02d}", "decision"))
    control = sum(1 for i in range(n)
                  if "NOTE:" in memmod.remember(_control(i), "fact"))
    notes = 0
    for i in range(n):
        if "NOTE:" in memmod.remember(_vague(i), "preference"):
            notes += 1
        memmod.remember(_named(i), "preference")

    with sqlite3.connect(work / "memory.db") as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE fact LIKE '%prefers%'").fetchone()[0]
        left = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE fact LIKE 'he prefers%'").fetchone()[0]

    # What the prompt digest spends its (few, fixed) slots on. Distinct claims
    # rather than lines: in the arm where the correction duplicates, every
    # claim in the window is present twice, so the same slots carry half as
    # much -- and half of what they do carry names nobody.
    lines = [ln for ln in memmod.preload().splitlines() if "typeface" in ln]
    claims = {ln.split("sans")[1][:2] for ln in lines if "sans" in ln}

    # Reachable by the one word a later question would actually carry: the
    # name. It shares no token with the vague wording, which is the whole
    # point -- this is the query the vague fact is invisible to.
    reachable = sum(1 for i in range(n)
                    if f"marlow{i:02d}" in memmod.recall(f"marlow{i:02d}", limit=1))
    memmod.reinforce("stopped")  # probes are non-events, same as `_top`
    return {
        "notes": notes, "poor": poor, "control": control, "rows": rows,
        "left": left, "slots": len(lines), "claims": len(claims),
        "unfindable": sum(1 for ln in lines if "] he prefers" in ln),
        "reachable": reachable,
    }


def _in_turn(fn: Any, outcome: str = "ok") -> Any:
    """Run one write or recall the way the model does -- inside a live turn.

    The loop publishes its stop event on the thread for the length of a run,
    and that event is what `memory._lineage` reads to tell the model's claim
    from the app's measurement, and what identifies the branch it grew on.
    Called outside one, every write in the provenance arm would be the
    machine's and the arm would measure nothing.
    """
    ev = threading.Event()
    registry.set_current_stop(ev)
    try:
        return fn()
    finally:
        registry.set_current_stop(None)
        memmod.reinforce(outcome, key=id(ev))


# ---------------------------------------------------------------- provenance
# The question none of the arms above can answer. They each grade a fact by its
# own fruit; none of them settles a COLLISION -- when two facts say nearly the
# same thing, which is kept and what may the survivor inherit from the other.
#
# Two families, because the fix has two halves and they fail differently.
#
#   restatement  one agent saying its own fact again in fewer words. Dropping a
#                word from a four-token sentence scores exactly 0.75 and the
#                merge bar is `> 0.75`, so this is the duplicate that survives
#                precisely at the ratio's own edge -- the pile-up is not a gap
#                in the threshold, it is the threshold.
#   takeover     a claim from one tree replacing a row grown on another. The
#                words win the slot either way; what is measured is whether the
#                rank the replaced fact earned goes with them.


def _prov_full(i: int) -> str:
    """Eight scoring tokens, three of them unique to `i` so pairs never collide."""
    return f"the jig{i:02d} press holds bolt{i:02d} in the north tray{i:02d} before dusk"


def _prov_short(i: int) -> str:
    """The same claim with two words dropped: a proper subset of `_prov_full`.

    Five tokens of the full eight, so the ratio scores 0.625 and the merge bar
    (`> 0.75`) is nowhere near it. That is the point of the fixture: this pair
    is a duplicate the threshold cannot see at any setting, and only kinship
    plus containment resolves it.
    """
    return f"the jig{i:02d} press holds bolt{i:02d} in tray{i:02d}"


def _prov_rival(i: int) -> str:
    """A re-wording of `_prov_full` ABOVE the ratio bar -- one word swapped.

    Seven shared tokens of nine, so it scores 0.778 and merges in both arms.
    The row is taken over either way; what the arm measures is the tally.
    """
    return f"the jig{i:02d} press holds bolt{i:02d} in the north tray{i:02d} before dawn"


def _prov_noise(i: int) -> str:
    """Model-tree background: claims from turns that fell over.

    Load-bearing, and not padding. A tree is graded on the fruit it HAS, and
    in the takeover store every original is grafted onto the user's tree when
    it is vouched for -- leaving the model's tree holding nothing, ungraded,
    and therefore never demoted. Which is correct behaviour and measures
    nothing, so the arm gives the model tree the fruit a real store would have
    given it by then.
    """
    return f"the spare drum{i:02d} was left in the annex{i:02d} loft"


def _prov_arm(work: Path, n: int, on: bool) -> dict[str, int]:
    """The identical workload under one setting of `provenance`.

    The restatement pairs are written INSIDE one live turn, because that is the
    only way the two sentences share a stem -- and sharing a stem is the whole
    of what licenses the fold. Written outside a turn they would be two writes
    by the machine and the arm would measure nothing.
    """
    memmod.configure(work, {"backend": "sqlite", "provenance": on})
    _in_turn(lambda: [(memmod.remember(_prov_full(i), "fact"),
                       memmod.remember(_prov_short(i), "fact"))
                      for i in range(n)], "ok")
    with sqlite3.connect(work / "memory.db") as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE fact LIKE '%press holds%'").fetchone()[0]
        # The poorer wording left sitting beside the fuller one. Counted rather
        # than inferred from the row count, because a fold that kept the SHORT
        # sentence would also show n rows -- and would be a duplicate resolved
        # by discarding the half worth keeping.
        short_left = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE fact NOT LIKE '%north%' "
            "AND fact LIKE '%press holds%'").fetchone()[0]

    # The takeover: a fact vouched for by hand and proven across turns, then
    # re-worded by the model. A different tree, above the ratio bar.
    take = work / "take"
    memmod.configure(take, {"backend": "sqlite", "provenance": on})
    for i in range(n):
        _in_turn(lambda i=i: memmod.remember(_prov_full(i), "fact"), "ok")
    for rid in range(1, n + 1):
        memmod.trust_memory(rid)   # grafted onto the user's own tree
    for i in range(n):
        _in_turn(lambda i=i: memmod.remember(_prov_noise(i), "fact"), "error")
    for _ in range(6):
        for i in range(n):
            _in_turn(lambda i=i: memmod.recall(f"jig{i:02d} bolt{i:02d} tray{i:02d}"), "ok")
    proven = sum(1 for r in memmod.all_memories() if r["wins"] >= 6)
    standing_start = next(
        (t["standing"] for t in memmod.orchard() if t["tree"] == "user"), -1.0)
    # Run in order, and the ORDER is what the last two numbers are about. Each
    # takeover moves one more row off the user's tree, and the question is what
    # that does to the tree's MEASURED standing -- because that standing is the
    # bar every later takeover has to clear, so a measurement that drains as
    # the tree is stripped lowers the bar for exactly the behaviour it exists
    # to catch. `standing_end` is the honest way to price it: the demotion
    # count alone does not discriminate here, since the tree being taken from
    # still holds the row in question when the comparison is made.
    demoted = 0
    for i in range(n):
        msg = _in_turn(lambda i=i: memmod.remember(_prov_rival(i), "fact"), "ok")
        demoted += 1 if "different tree" in msg else 0
    _end = {t["tree"]: t for t in memmod.orchard()}
    held_at_end = _end.get("user", {}).get("facts", 0)
    standing_start = round(standing_start, 6)
    standing_end = round(_end.get("user", {}).get("standing") or -1.0, 6)
    with sqlite3.connect(take / "memory.db") as conn:
        take_rows = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE fact LIKE '%press holds%'").fetchone()[0]
        # The measurement: how many took over a proven row and kept its tally.
        inherited = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE fact LIKE '%dawn%' AND wins > 0"
        ).fetchone()[0]
    return {"rows": rows, "short_left": short_left, "proven": proven,
            "take_rows": take_rows, "inherited": inherited,
            "demoted": demoted, "held_at_end": held_at_end,
            "standing_start": standing_start, "standing_end": standing_end}


# --------------------------------------------------------------- scattershot
# The prior question to the cue check, the way the cue check is the prior
# question to reinforcement. `_vague`/`_named` measure a fact that carries no
# cues of its own; these measure a fact that carries perfectly good ones and is
# still missed, because the QUESTION arrived in different words. Nothing about
# "Dave runs the training scripts on the RTX 4090" is poorly encoded, and
# "what GPU do I have" reaches none of it.
#
# Every probe below shares ZERO tokens with the fact it is meant to find, which
# is what makes it a probe: the tight shot scores on literal overlap, so a
# probe with one shared word would measure nothing. Four ways for the spread to
# bridge that gap, one per pellet kind, plus the two guards that decide whether
# a wider spread is safe to fire at all.


def _shot_store(n: int) -> None:
    """Four facts per subject: one to find, two to reach it through, one to bond.

    Written the way facts fall out of a conversation rather than the way a
    later question will ask for them -- which is the whole premise. Note the
    numbering runs INSIDE the words (`scaffold07ing`), so the stem probe has a
    per-subject answer and a wrong-subject near-miss to be rejected against.
    """
    for i in range(n):
        memmod.remember(
            f"the atlas{i:02d} scaffold{i:02d}ing was rebuilt last spring", "fact")
        memmod.remember(
            f"berth{i:02d} renewal was filed against atlas{i:02d} in march", "fact")
        memmod.remember(
            f"berth{i:02d} inspection covers atlas{i:02d} every winter", "fact")
        memmod.remember(
            f"the kiln{i:02d} at atlas{i:02d} fires at eleven hundred degrees", "fact")


def _shot_arm(work: Path, n: int, on: bool) -> dict[str, int]:
    """The identical workload under one setting of `scattershot`."""
    memmod.configure(work, {"backend": "sqlite", "scattershot": on})
    _shot_store(n)

    # STEM -- the question conjugates the word differently. Deliberately also a
    # rejection test: `scaffold07ed` must reach subject 07 and no other, so a
    # spread that folds every "scaffold..." together scores on the recovery
    # line and immediately loses the guard line below.
    stem = sum(1 for i in range(n)
               if f"scaffold{i:02d}ing" in _probe(f"scaffold{i:02d}ed", 2))
    stem_wrong = sum(1 for i in range(n)
                     for other in (f"scaffold{(i + 1) % n:02d}ing",)
                     if n > 1 and other in _probe(f"scaffold{i:02d}ed", 2))

    # NEIGHBOUR -- the question names something the store keeps in the same
    # sentence as the answer. `berth07` matches its own two facts literally;
    # what it cannot match, and what only the spread reaches, is the scaffolding
    # fact those two both point at.
    near = sum(1 for i in range(n)
               if f"scaffold{i:02d}ing" in _probe(f"berth{i:02d}", 4))

    # BOND -- cold, then after one turn that used the fact and ENDED WELL. The
    # cold half is the control: a bond that fires before it is earned is a
    # bond that was never evidence.
    bond_cold = sum(1 for i in range(n)
                    if f"kiln{i:02d}" in _probe(f"pottery{i:02d}", 2))
    for i in range(n):
        memmod.recall(f"pottery{i:02d} kiln{i:02d}", limit=1)
        memmod.reinforce("ok")
    bond_warm = sum(1 for i in range(n)
                    if f"kiln{i:02d}" in _probe(f"pottery{i:02d}", 2))
    bond_wrong = sum(1 for i in range(n)
                     if f"scaffold{i:02d}ing" in _probe(f"pottery{i:02d}", 2))

    # GUARD -- a subject the store does not hold. A spread that answers this
    # is not finding facts, it is manufacturing them, and it would teach a
    # model to stop believing the ones it does find.
    absent = sum(1 for i in range(n)
                 if "No memories matching" not in _probe(f"submarine{i:02d} periscope", 4))

    # THE CHOKE -- exactly four facts name `atlas07`, so at limit=4 the tight
    # shot fills every slot and the spread must have nowhere to go. Byte
    # equality against the OFF arm is checked by the caller; what is counted
    # here is the cap, which nothing is allowed past.
    over_cap = 0
    filled: list[str] = []
    wide: list[list[str]] = []
    for i in range(n):
        got = _probe(f"atlas{i:02d}", 4)
        filled.append(got)
        if len(got.splitlines()) > 4:
            over_cap += 1
        # ...and where the spread does have room AND something to put in it.
        # `berth07` matches its own two facts literally and leaves six slots
        # empty at this limit, which is the case the choke actually has to
        # survive: the weaker claim holds, and it is the one that matters in
        # practice -- whatever the spread adds, it adds AFTER, so the hits the
        # tight shot found keep their places and their order.
        room = _probe(f"berth{i:02d}", 8)
        wide.append(room.splitlines())
        if len(room.splitlines()) > 8:
            over_cap += 1
    return {"stem": stem, "stem_wrong": stem_wrong, "near": near,
            "bond_cold": bond_cold, "bond_warm": bond_warm,
            "bond_wrong": bond_wrong, "absent": absent, "over_cap": over_cap,
            "filled": filled, "wide": wide, "bonds": memmod.bond_count()}


def _probe(query: str, limit: int) -> str:
    """A recall that changes nothing -- same non-event contract as `_top`."""
    out = memmod.recall(query, limit=limit)
    memmod.reinforce("stopped")
    return out


def _recency_arm(work: Path, on: bool) -> str:
    """One store, built to make the fallback's old failure visible.

    Both facts are filed under Development: the first because it is genuinely
    about building software, the second because "code" is in that theme's
    vocabulary and a door code is not. The question routes to Development and
    shares no word with either. With nothing but recency to separate them the
    door wins, purely for being newer, and arrives in a real hit's format --
    which is what a wider spread has to beat to be worth firing.
    """
    memmod.configure(work, {"backend": "sqlite", "scattershot": on})
    memmod.remember("the atlas build pipeline compiles the rust binary each night", "fact")
    memmod.remember("the garage door code is 4417", "fact")
    return _probe("docker kubernetes deploy", 2)


def _habit_short(h: int) -> str:
    """A habit's compact form: three tokens no other habit shares.

    Fully distinct on purpose. Habits that share most of their trigger tokens
    are ONE request shape to the brain's coverage matcher, and the sleep pass
    honours that by design -- the twin family below measures that collapse as
    its own claim, so the main habits must not trip it by accident.
    """
    return f"tend{h:02d} the plot{h:02d} beds{h:02d}"


def _sleep_store(habits: int, noise: int, db: Path) -> None:
    """Plant the episode log the sleep pass will be measured on."""
    with sqlite3.connect(db) as conn:
        def ep(ts: str, req: str, outcome: str = "ok") -> None:
            conn.execute(
                "INSERT INTO episodes (ts, request, summary, tools, outcome) "
                "VALUES (?,?,?,?,?)",
                (ts, req, "Tended and logged", "run_shell", outcome))

        # Real habits: three phrasings across three days, shortest first so
        # the promoted trigger is the compact form.
        for h in range(habits):
            short = _habit_short(h)
            ep("2026-08-18T09:00:00", short)
            ep("2026-08-19T09:00:00", f"{short} and log the result")
            ep("2026-08-20T09:00:00", f"please {short} for me")
        # Near-twins: three habits sharing two of three trigger tokens. The
        # coverage matcher reads them as one request shape, so exactly ONE
        # shortcut must come out -- a pile of near-identical shortcuts would
        # crowd the preload digest with copies.
        for t in range(3):
            ep("2026-08-18T09:30:00", f"restock the shelf{t:02d} rack")
            ep("2026-08-19T09:30:00", f"restock the shelf{t:02d} rack and log it")
            ep("2026-08-20T09:30:00", f"please restock the shelf{t:02d} rack")
        # Noise that must NOT promote: unique one-offs (disjoint vocabulary),
        # a task done only twice, one afternoon of retries, and a habit-shaped
        # run of failures.
        for u in range(noise):
            ep(f"2026-08-{18 + u % 3}T10:00:00", f"unique{u:02d} chore{u:02d} once")
        ep("2026-08-18T11:00:00", "defrost the sample freezer")
        ep("2026-08-19T11:00:00", "defrost the sample freezer again")
        for t in ("12:00", "12:10", "12:20"):
            ep(f"2026-08-20T{t}:00", "reseat the loose bench camera")
        for d in ("18", "19", "20"):
            ep(f"2026-08-{d}T13:00:00", "flash the beta firmware build", outcome="error")


def run_sim(scale: str = "small", quiet: bool = False) -> dict[str, Any]:
    p = SCALES[scale]
    n, habits = p["pairs"], p["habits"]
    work = Path(tempfile.mkdtemp(prefix="memsim-"))
    for arm in ("off", "on", "sleep", "enc-off", "enc-on", "enc-vague",
                "prov-off", "prov-off/take", "prov-on", "prov-on/take",
                "shot-off", "shot-on", "rec-off", "rec-on"):
        (work / arm).mkdir(parents=True, exist_ok=True)

    def say(text: str = "") -> None:
        if not quiet:
            print(text)

    say(f"\n{BOLD}  Memory recall mechanical A/B{RESET}")
    say(f"{DIM}  {n} tie pairs x {p['credits']} credited rounds · "
        f"{habits} habits + {p['noise']}+ noise episodes · "
        f"{n} encoding pairs · {n} scattershot subjects · no model{RESET}")

    # -- OFF arm: identical workload, reinforcement disabled --------------
    memmod.configure(work / "off", {"backend": "sqlite", "reinforcement": False})
    _build_pairs(n)
    _credit_pairs(n, p["credits"])  # marks nothing, credits nothing
    off = _rank_arm(n)

    # -- ON arm -----------------------------------------------------------
    memmod.configure(work / "on", {"backend": "sqlite"})
    _build_pairs(n)
    _credit_pairs(n, p["credits"])
    on = _rank_arm(n)

    # -- Decay, ON store only ---------------------------------------------
    # Both fresh: the heavier habit keeps the tie. Then the habit goes stale
    # (last_used backdated 120 days, four half-lives) while the rival earns
    # one fresh credit: recent light use must now beat heavy stale use.
    for i in range(n):
        memmod.recall(f"site{i:02d} panel{i:02d} manual gray drawer", limit=1)
    memmod.reinforce("ok")
    fresh_habit = sum(1 for i in range(n)
                      if "checklist" in _top(f"site{i:02d} panel{i:02d} lives"))
    stale = (datetime.now() - timedelta(days=120)).isoformat(timespec="seconds")
    with sqlite3.connect(work / "on" / "memory.db") as conn:
        conn.execute("UPDATE memories SET last_used = ? WHERE fact LIKE '%checklist%'",
                     (stale,))
    recent_over_stale = sum(1 for i in range(n)
                            if "manual" in _top(f"site{i:02d} panel{i:02d} lives"))

    # -- Encoding: can the fact be found at all? ---------------------------
    # The prior question to everything above. Reinforcement decides which
    # matching fact ranks first and the sleep pass decides which habits become
    # shortcuts; neither can help a fact that matches nothing. Only variable:
    # `cue_check`.
    enc_off = _encoding_arm(work / "enc-off", n, check=False)
    enc_on = _encoding_arm(work / "enc-on", n, check=True)

    # -- Provenance: which tree did the fruit come from? ------------------
    # The question the arm above cannot reach. `cue_check` decides whether a
    # fact can be found; this decides which of two findable ones is kept, and
    # what the survivor is allowed to inherit. Only variable: `provenance`.
    prov_off = _prov_arm(work / "prov-off", n, on=False)
    prov_on = _prov_arm(work / "prov-on", n, on=True)

    # The claim underneath both arms, measured on its own: with only the vague
    # wording in the store, the name a later question would carry reaches
    # nothing. No rewrite here -- this is the store as it stands the moment
    # before the correction, which is the state the OFF arm leaves behind.
    memmod.configure(work / "enc-vague", {"backend": "sqlite"})
    for i in range(n):
        memmod.remember(_vague(i), "preference")
    vague_reachable = sum(1 for i in range(n)
                          if f"sans{i:02d}" in memmod.recall(f"marlow{i:02d}", limit=1))
    memmod.reinforce("stopped")

    # -- Scattershot: can the fact be found when the QUESTION differs? -----
    # Prior to the cue check, which measures a fact that carries no cues of its
    # own. These facts carry perfectly good ones and are still missed, because
    # recall matches literally. Only variable: `scattershot`.
    shot_off = _shot_arm(work / "shot-off", n, on=False)
    shot_on = _shot_arm(work / "shot-on", n, on=True)
    # Byte equality where the tight shot filled every slot, and the prefix
    # property where it did not. Together these are THE CHOKE: the spread can
    # add, and it can do nothing else.
    choke_identical = sum(1 for a, b in zip(shot_off["filled"], shot_on["filled"]) if a == b)
    choke_prefix = sum(1 for a, b in zip(shot_off["wide"], shot_on["wide"])
                       if b[:len(a)] == a)
    shot_added = sum(len(b) - len(a) for a, b in zip(shot_off["wide"], shot_on["wide"]))
    shot_base = sum(len(a) for a in shot_off["wide"])

    # The fallback, measured on its own. Both facts are filed under Development
    # and the question routes there sharing no word with either, so with only
    # recency to separate them the newest wins -- see `_recency_arm`.
    rec_off = _recency_arm(work / "rec-off", on=False)
    rec_on = _recency_arm(work / "rec-on", on=True)
    rec_off_first = rec_off.splitlines()[0] if rec_off.splitlines() else ""
    rec_on_first = rec_on.splitlines()[0] if rec_on.splitlines() else ""

    # -- The sleep pass ----------------------------------------------------
    sleep = work / "sleep"
    memmod.configure(sleep, {"backend": "sqlite"})
    brainmod.configure(sleep, {"learning": True})
    _sleep_store(habits, p["noise"], sleep / "memory.db")
    per_sweep: list[int] = []
    while True:
        got = brainmod.consolidate(force=True)
        if not got:
            break
        per_sweep.append(len(got))
    triggers = {q["trigger"] for q in brainmod.all_patterns()}
    wanted = {_habit_short(h) for h in range(habits)}
    promoted = len(wanted & triggers)
    twin_shortcuts = sum(1 for t in triggers if t.startswith("restock the shelf"))
    false_promotions = len(triggers - wanted) - twin_shortcuts
    short_answered = sum(
        1 for h in range(habits)
        if any(q.get("trigger") == _habit_short(h)
               for q in brainmod.predict(_habit_short(h))))

    metrics: dict[str, Any] = {
        "scale": scale, "pairs": n, "credits": p["credits"], "habits": habits,
        "tie_off": off["tie_habitual_first"], "tie_on": on["tie_habitual_first"],
        "guard_off": off["guard_better_match_first"],
        "guard_on": on["guard_better_match_first"],
        "guard_divergence": abs(on["guard_better_match_first"]
                                - off["guard_better_match_first"]),
        "fresh_habit_holds": fresh_habit, "recent_over_stale": recent_over_stale,
        "promoted": promoted, "false_promotions": false_promotions,
        "twin_shortcuts": twin_shortcuts,
        "short_answered": short_answered, "promotions_per_sweep": per_sweep,
        "enc_vague_reachable": vague_reachable,
        "enc_named_reachable": enc_on["reachable"],
        "enc_notes_on": enc_on["notes"], "enc_notes_off": enc_off["notes"],
        "enc_poor_flagged": enc_on["poor"],
        "enc_false_flags": enc_on["control"] + enc_off["control"],
        "enc_rows_on": enc_on["rows"], "enc_rows_off": enc_off["rows"],
        "enc_vague_left_on": enc_on["left"], "enc_vague_left_off": enc_off["left"],
        "enc_digest_slots": enc_on["slots"],
        "enc_digest_claims_on": enc_on["claims"],
        "enc_digest_claims_off": enc_off["claims"],
        "enc_digest_unfindable_off": enc_off["unfindable"],
        "enc_digest_unfindable_on": enc_on["unfindable"],
        "prov_rows_on": prov_on["rows"], "prov_rows_off": prov_off["rows"],
        "prov_short_left_on": prov_on["short_left"],
        "prov_short_left_off": prov_off["short_left"],
        "prov_proven": prov_on["proven"],
        "prov_demoted_on": prov_on["demoted"], "prov_demoted_off": prov_off["demoted"],
        "prov_tree_emptied": prov_on["held_at_end"],
        "prov_standing_start": prov_on["standing_start"],
        "prov_standing_end": prov_on["standing_end"],
        "prov_take_rows_on": prov_on["take_rows"],
        "prov_take_rows_off": prov_off["take_rows"],
        "prov_inherited_on": prov_on["inherited"],
        "prov_inherited_off": prov_off["inherited"],
        "shot_stem_off": shot_off["stem"], "shot_stem_on": shot_on["stem"],
        "shot_near_off": shot_off["near"], "shot_near_on": shot_on["near"],
        "shot_bond_cold": shot_on["bond_cold"], "shot_bond_warm": shot_on["bond_warm"],
        "shot_bond_off": shot_off["bond_warm"], "shot_bonds": shot_on["bonds"],
        "shot_wrong_subject": shot_on["stem_wrong"] + shot_on["bond_wrong"],
        "shot_absent_off": shot_off["absent"], "shot_absent_on": shot_on["absent"],
        "shot_over_cap": shot_off["over_cap"] + shot_on["over_cap"],
        "shot_choke_identical": choke_identical, "shot_choke_prefix": choke_prefix,
        "shot_slots_added": shot_added, "shot_slots_base": shot_base,
        "shot_recency_off": rec_off_first, "shot_recency_on": rec_on_first,
    }

    # Every mechanical claim is a gate. A bench that shades a regression into
    # a slightly worse number is a bench nobody reads in time.
    gates = [
        ("ties reorder toward the fact actually used", metrics["tie_on"] == n),
        ("...and the tie is real: recency picks the rival with the knob off",
         metrics["tie_off"] == 0),
        ("a better match beats fame in both arms, identically",
         metrics["guard_on"] == n and metrics["guard_divergence"] == 0),
        ("a heavier habit keeps the tie while fresh", metrics["fresh_habit_holds"] == n),
        ("recent light use beats heavy stale use", metrics["recent_over_stale"] == n),
        ("every planted habit is promoted", metrics["promoted"] == habits),
        ("near-twins collapse into one shortcut, not several",
         metrics["twin_shortcuts"] == 1),
        ("nothing else is promoted", metrics["false_promotions"] == 0),
        ("each promoted habit answers its short form", metrics["short_answered"] == habits),
        ("a fact that names nobody is unreachable by its subject",
         metrics["enc_vague_reachable"] == 0),
        ("...and the same claim, named, is reachable by it",
         metrics["enc_named_reachable"] == n),
        ("every referent gap is reported as the fact is written",
         metrics["enc_notes_on"] == n),
        ("a fact with almost nothing to search on is reported too",
         metrics["enc_poor_flagged"] == n),
        ("no well-formed fact is flagged, in either arm",
         metrics["enc_false_flags"] == 0),
        ("the named rewrite replaces the vague row",
         metrics["enc_rows_on"] == n and metrics["enc_vague_left_on"] == 0),
        ("with the check off the same rewrite duplicates instead",
         metrics["enc_rows_off"] == 2 * n and metrics["enc_vague_left_off"] == n),
        ("...and the check off says nothing that would prompt it",
         metrics["enc_notes_off"] == 0),
        ("the digest then carries half the claims in the same slots",
         metrics["enc_digest_claims_on"] == metrics["enc_digest_slots"]
         and metrics["enc_digest_claims_off"] * 2 == metrics["enc_digest_slots"]),
        ("...half of them sentences naming nobody",
         metrics["enc_digest_unfindable_on"] == 0
         and metrics["enc_digest_unfindable_off"] * 2 == metrics["enc_digest_slots"]),
        ("one agent restating its own fact adds no second row",
         metrics["prov_rows_on"] == n),
        ("...and with provenance off the same pair duplicates, as it did",
         metrics["prov_rows_off"] == 2 * n),
        ("...and the fold keeps the fuller wording, never the poorer one",
         metrics["prov_short_left_on"] == 0 and metrics["prov_short_left_off"] == n),
        ("the replaced fact really had earned its rank first",
         metrics["prov_proven"] == n),
        ("a claim from another tree still takes the slot, not a new row",
         metrics["prov_take_rows_on"] == n and metrics["prov_take_rows_off"] == n),
        ("...but inherits none of the rank the tree it replaced had earned",
         metrics["prov_inherited_on"] == 0),
        ("every takeover is demoted, down to the one that empties the tree",
         metrics["prov_demoted_on"] == n and metrics["prov_tree_emptied"] == 0),
        ("...and none is with provenance off", metrics["prov_demoted_off"] == 0),
        ("a stripped tree keeps the standing its lost fruit earned",
         metrics["prov_standing_end"] == metrics["prov_standing_start"]
         and metrics["prov_standing_end"] > 0),
        ("...which is exactly what it used to inherit, every time",
         metrics["prov_inherited_off"] == n),
        ("a question that conjugates the word differently reaches the fact",
         metrics["shot_stem_on"] == n and metrics["shot_stem_off"] == 0),
        ("a question that names its neighbour reaches it too",
         metrics["shot_near_on"] == n and metrics["shot_near_off"] == 0),
        ("a word with nothing in common reaches it after one turn that worked",
         metrics["shot_bond_warm"] == n and metrics["shot_bond_off"] == 0),
        ("...and not before: an unearned bond never fires",
         metrics["shot_bond_cold"] == 0),
        ("no probe reaches the wrong subject", metrics["shot_wrong_subject"] == 0),
        ("a subject the store does not hold is still answered with nothing",
         metrics["shot_absent_on"] == 0 and metrics["shot_absent_off"] == 0),
        ("the choke holds: a filled result is byte-identical to the spread off",
         metrics["shot_choke_identical"] == n),
        ("...and where there was room, the spread only ever appended",
         metrics["shot_choke_prefix"] == n and metrics["shot_slots_added"] > 0),
        ("nothing exceeds the recall cap, in either arm",
         metrics["shot_over_cap"] == 0),
        ("the spread outranks the recency fallback it replaced",
         "pipeline" in metrics["shot_recency_on"]
         and "garage" in metrics["shot_recency_off"]),
    ]

    say(f"\n  {BOLD}Reinforcement (ties, guard, decay){RESET}")
    say(f"    habitual fact first on ties   OFF {metrics['tie_off']:>3}/{n}   "
        f"ON {metrics['tie_on']:>3}/{n}")
    say(f"    better match first (guard)    OFF {metrics['guard_off']:>3}/{n}   "
        f"ON {metrics['guard_on']:>3}/{n}   divergence {metrics['guard_divergence']}")
    say(f"    fresh habit holds the tie     {metrics['fresh_habit_holds']:>3}/{n}")
    say(f"    recent use beats stale habit  {metrics['recent_over_stale']:>3}/{n}"
        f"{DIM}   (120 days idle = four half-lives){RESET}")
    say(f"\n  {BOLD}Sleep pass (consolidation){RESET}")
    say(f"    habits promoted               {metrics['promoted']:>3}/{habits}"
        f"   in sweeps of {'+'.join(map(str, per_sweep)) or '0'}")
    say(f"    near-twins collapse to        {metrics['twin_shortcuts']:>3} shortcut"
        f"{DIM}   (3 phrasings, one request shape){RESET}")
    say(f"    false promotions              {metrics['false_promotions']:>3}"
        f"{DIM}   (one-offs, two-timers, retries, failures){RESET}")
    say(f"    short form answered           {metrics['short_answered']:>3}/{habits}")
    say(f"\n  {BOLD}Cue check (encoding){RESET}")
    say(f"    vague fact found by its name  {metrics['enc_vague_reachable']:>3}/{n}"
        f"{DIM}   (no name in the sentence, nothing to match){RESET}")
    say(f"    named rewrite found by it     {metrics['enc_named_reachable']:>3}/{n}")
    say(f"    gaps reported at write time   OFF {metrics['enc_notes_off']:>3}/{n}   "
        f"ON {metrics['enc_notes_on']:>3}/{n}"
        f"{DIM}   + {metrics['enc_poor_flagged']}/{n} cue-poor{RESET}")
    say(f"    well-formed facts flagged     {metrics['enc_false_flags']:>3}"
        f"{DIM}   (both arms; a check that fires on everything measures nothing){RESET}")
    say(f"    rows for {n} claims            OFF {metrics['enc_rows_off']:>3}     "
        f"ON {metrics['enc_rows_on']:>3}"
        f"{DIM}   vague wording left: {metrics['enc_vague_left_off']} / "
        f"{metrics['enc_vague_left_on']}{RESET}")
    say(f"    claims in {metrics['enc_digest_slots']} digest slots      "
        f"OFF {metrics['enc_digest_claims_off']:>3}     "
        f"ON {metrics['enc_digest_claims_on']:>3}"
        f"{DIM}   naming nobody: {metrics['enc_digest_unfindable_off']} / "
        f"{metrics['enc_digest_unfindable_on']}{RESET}")

    say(f"\n  {BOLD}Provenance (which tree the fruit came from){RESET}")
    say(f"    rows for {n} restated claims   OFF {metrics['prov_rows_off']:>3}     "
        f"ON {metrics['prov_rows_on']:>3}"
        f"{DIM}   poorer wording left: {metrics['prov_short_left_off']} / "
        f"{metrics['prov_short_left_on']}{RESET}")
    say(f"    cross-tree takeovers          OFF {metrics['prov_take_rows_off']:>3}     "
        f"ON {metrics['prov_take_rows_on']:>3}"
        f"{DIM}   (the words win the slot in both){RESET}")
    say(f"    ...inheriting a proven rank   OFF {metrics['prov_inherited_off']:>3}/{n}   "
        f"ON {metrics['prov_inherited_on']:>3}/{n}"
        f"{DIM}   (6 winning turns, earned by the fact it replaced){RESET}")
    say(f"    ...caught as the tree empties  OFF {metrics['prov_demoted_off']:>3}/{n}   "
        f"ON {metrics['prov_demoted_on']:>3}/{n}"
        f"{DIM}   (it holds {metrics['prov_tree_emptied']} of its own by the last one)"
        f"{RESET}")
    say(f"    stripped tree's standing      {metrics['prov_standing_start']:.3f} "
        f"-> {metrics['prov_standing_end']:.3f}"
        f"{DIM}   the bar every later takeover has to clear{RESET}")

    say(f"\n  {BOLD}Scattershot (the spread, and the choke on it){RESET}")
    say(f"    conjugated word (stem)        OFF {metrics['shot_stem_off']:>3}/{n}   "
        f"ON {metrics['shot_stem_on']:>3}/{n}")
    say(f"    neighbour named (co-occur)    OFF {metrics['shot_near_off']:>3}/{n}   "
        f"ON {metrics['shot_near_on']:>3}/{n}")
    say(f"    unrelated word (bond)         OFF {metrics['shot_bond_off']:>3}/{n}   "
        f"ON {metrics['shot_bond_warm']:>3}/{n}"
        f"{DIM}   cold {metrics['shot_bond_cold']}/{n}, "
        f"{metrics['shot_bonds']} bonds earned{RESET}")
    say(f"    wrong subject reached         {metrics['shot_wrong_subject']:>3}"
        f"{DIM}   (a spread that hits everything finds nothing){RESET}")
    say(f"    absent subject answered       OFF {metrics['shot_absent_off']:>3}     "
        f"ON {metrics['shot_absent_on']:>3}"
        f"{DIM}   (both must be 0: silence beats invention){RESET}")
    say(f"    filled result unchanged       {metrics['shot_choke_identical']:>3}/{n}"
        f"{DIM}   byte-identical with the spread off{RESET}")
    say(f"    partial result appended to    {metrics['shot_choke_prefix']:>3}/{n}"
        f"{DIM}   never reordered, only extended{RESET}")
    say(f"    lines returned on those        {metrics['shot_slots_base']:>3} "
        f"-> {metrics['shot_slots_base'] + metrics['shot_slots_added']}"
        f"{DIM}   what the spread COSTS: slots that were empty now carry a fact{RESET}")
    say(f"    results over the cap          {metrics['shot_over_cap']:>3}"
        f"{DIM}   (the per-recall cap is unchanged, so the worst case is too){RESET}")
    say(f"    routed query with no match    "
        f"{DIM}OFF{RESET} {metrics['shot_recency_off'][:52] or '(nothing)'}")
    say(f"                                  "
        f"{DIM}ON {RESET} {metrics['shot_recency_on'][:52] or '(nothing)'}")

    ok = True
    say(f"\n  {BOLD}Gates{RESET}")
    for label, held in gates:
        ok = ok and held
        say(f"    {GREEN + 'ok  ' if held else RED + 'FAIL'}{RESET}  {label}")

    say(f"\n  {BOLD}What this does and does not show{RESET}")
    say(f"  {DIM}  Shown:     the knob reorders ties toward use, is strictly bounded by")
    say("             relevance, fades with disuse, the sleep pass promotes exactly")
    say("             the planted habits, a fact recall could never find is caught")
    say("             as it is written -- while well-formed ones are not -- and a")
    say("             question worded differently from the fact now reaches it four")
    say("             ways, without displacing one hit the narrower search had.")
    say("             The spread is not free: it fills slots that came back empty,")
    say("             so an average result gets LONGER even though the per-recall")
    say("             cap never moves. Bounded at _SHOT_FILL slots and a floor")
    say("             relative to the best pellet, because the eighth-best pellet")
    say("             hit is noise wearing a hit's format.")
    say("             And a duplicate that the ratio bar cannot see at any")
    say("             setting is folded back into the row it restates, keeping")
    say("             the fuller wording -- while a claim taking over a row")
    say("             grown on another tree still wins the slot and no longer")
    say("             inherits the rank that tree earned.")
    say("    NOT shown: whether YOUR store has such ties, whether a real model asks")
    say("             recall better questions, or whether it takes the correction")
    say("             when told a fact names nobody. The encoding arms SCRIPT that")
    say("             correction in both directions to price it, which is not the")
    say("             same as evidence a model makes it. Scattershot's probes are")
    say("             likewise BUILT to be bridgeable -- each one is the gap its")
    say("             own pellet kind exists to cross, so the recovery columns")
    say("             price the mechanism working, not how often your questions")
    say("             miss by exactly that much. The guards are the honest half:")
    say("             a spread wide enough to reach a wrong subject, or to answer")
    say("             about a subject the store does not hold, would show up")
    say("             The provenance arm gives the model's tree its poor")
    say("             fruit rather than waiting for a real store to grow it,")
    say("             so it prices the rule FIRING -- not how often two of")
    say("             your own trees end up far enough apart to fire it.")
    say(f"             there and does not. Watch your own facts.{RESET}")

    metrics["ok"] = ok
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Memory recall mechanical A/B (no GPU)")
    ap.add_argument("--scale", choices=list(SCALES), default="small")
    ap.add_argument("--json", help="Write the metrics to this file")
    args = ap.parse_args(argv)
    metrics = run_sim(args.scale)
    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"{DIM}  wrote {args.json}{RESET}")
    return 0 if metrics["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
