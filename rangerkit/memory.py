"""Cross-session memory -- facts that survive the process that learned them.

SQLite-backed, stdlib only -- no embedding model, no vector store. Recall uses
weighted keyword scoring, which is the right trade here: an embedding model
would compete with the main model for VRAM, and for a few hundred personal
facts keyword search is both adequate and instant.

Keyword scoring alone is blind in one way, and the module answers it twice.
`scattershot` fires each query as a spread of related terms derived from the
store's own vocabulary, so a fact worded differently from the question can
still be reached. And when a semantic backend is registered (`semantic.py` --
none is bundled, and the seam is empty by default), its hits are fused with the
keyword hits into ONE ranked list by reciprocal rank fusion, hard-capped at the
recall limit. Both retrievers compete for the same slots, so enabling semantic
recall can SHRINK the tokens sent to the model -- by displacing a weak keyword
hit -- instead of only ever adding to them.

Facts live in `<state_dir>/memory.db` and survive restarts.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .registry import registry
from . import semantic as palace
from . import dbsafe

_DB: Path | None = None

# Recall fusion settings, set by configure(). RRF works on ranks, so the
# keyword overlap counts and a backend's cosine scores never need a shared
# scale.
_FUSION = "rrf"          # "rrf" | "keyword_only"
_RRF_K = 60
_MIN_SCORE = 0.0
_RECALL_LIMIT = 8

# Sector routing: narrow recall to the "brain" a request concerns, and ship
# only sector-agnostic facts in the system prompt. Off restores the old
# behaviour (search everything, preload the newest N).
_ROUTING = True
_PRELOAD_CORE = 8

# How much being in the request's sector is worth when ranking a fact. Keyword
# overlap is a whole-number count of shared tokens, so anything below 1.0 makes
# the sector a tiebreaker rather than a trump: a fact from the routed subject
# wins against an equally-good match from elsewhere, and loses to a fact that
# genuinely answers the question better. That ceiling is the point -- routing is
# a guess made from a handful of words, and it must not be able to outvote the
# evidence in the facts themselves.
_SECTOR_BONUS = 0.5

# Retrieval reinforcement -- recall is a write. A fact recall serves into a
# turn that then succeeds earns a use/win tally, the same evidence the brain
# keeps for learned shortcuts, and that familiarity breaks ties between
# equally-relevant facts at the next recall. Two bounds keep it honest. The
# boost is strictly under _REINFORCE_WEIGHT (wins <= uses < uses + 2 keeps the
# win share under 1), and _REINFORCE_WEIGHT + _SECTOR_BONUS < 1.0 holds by
# construction -- so even a famous, in-sector fact loses to one more token of
# genuine overlap: relevance stays the gate, familiarity only orders what
# already matched, and nothing can ratchet its way to the top. And it fades:
# the boost halves for every _REINFORCE_HALFLIFE_DAYS a fact goes unrecalled,
# so rank earned last month is not rank held forever.
_REINFORCE = True
_REINFORCE_WEIGHT = 0.5
_REINFORCE_HALFLIFE_DAYS = 30.0

# Seed grading -- the other half of the same idea. Reinforcement above grades
# what memory GIVES BACK; nothing graded what goes IN, and a store is only ever
# as good as what is fed to it. A fact the model saves mid-turn is written at
# full strength whatever becomes of that turn, and a sector-less one then rides
# the system prompt into every turn that follows -- so a fact invented in a run
# that fell over gets read back as settled truth from then on.
#
# So a write made inside a turn that then FAILS is held back: kept in full,
# still returned by `recall`, still listed by `all_memories`, but out of
# the prompt digest until it has earned its place by feeding a turn that
# succeeded (or the user vouches for it by hand). Nothing is hidden and nothing
# is deleted -- the only thing withheld is the automatic promotion into every
# future prompt, which is the one privilege an unproven fact has not earned.
#
# Deliberately narrow. Only a failed turn holds anything back: a stopped run is
# not evidence (same reason `reinforce` skips it), and a write made outside any
# turn -- `ui.scan_apps` folding the app index in from a worker thread -- is not
# the model's guess at all, so `_turn_is_live` gates the planting on this
# thread actually running one.
_HOLD_UNPROVEN = True

# Encoding quality -- the write-side counterpart to both of the above.
#
# Reinforcement grades what memory GIVES BACK; seed grading grades what a
# turn's outcome says about what went IN. Neither asks the question that comes
# before either: was the fact written in a form recall can find at all? That is
# not a matter of taste here. `recall` ranks by how many scoring tokens a query
# shares with the stored sentence, and `_tokens` drops words under three
# letters -- so a fact whose subject is "he" has no "he" to match against, and
# no query about Dave will ever reach it. It is stored, it is counted, it can
# ride the prompt digest, and it is invisible to the one mechanism meant to
# surface it on merit.
#
# This is the encoding specificity principle (Tulving & Thomson 1973) stated in
# SQL: retrieval works to the extent that the cues present when a fact is read
# back were encoded when it was written. The model writing the fact is the only
# party that ever holds both ends of that, and it holds them for one turn --
# which is why the check runs at write time, where the referent is still on
# screen, rather than at recall, where the evidence of what went wrong is
# exactly what is missing.
#
# It reports; it does not refuse. Nothing is rejected, nothing is reworded
# behind the model's back, and a flagged fact is stored, recallable and
# preloaded like any other. The whole intervention is one extra line on the
# tool result -- which is also the strongest instruction available, because it
# arrives at the one moment the gap can still be closed by the one party who
# can close it.
_CUE_CHECK = True

# Fewer distinct scoring tokens than this and recall can only reach the fact
# through a query that happens to use that one word. Two is the floor a real
# short fact clears ("Sarah is my wife" -> {sarah, wife}) and a filler one does
# not ("yes, do that" -> {yes}).
_CUE_MIN = 2

# Provenance -- where a fact came from, traced back stem to branch to tree.
#
# Everything above grades a fact by ITS OWN fruit: how often recall served it
# into a turn that worked, what became of the turn that wrote it, whether the
# sentence carries its own cues. None of it asks the question that decides a
# COLLISION -- when two facts say nearly the same thing, which of them is the
# one to keep, and what may the survivor inherit from the other.
#
# `remember` had one answer to that: the newer wording wins the row and keeps
# everything the older one earned. That is right when a fact is being re-worded
# by the same hand and wrong in every other case, and the store had no way to
# tell those apart, because a row recorded WHAT was claimed and never WHERE the
# claim came from. Three failures follow from the one gap:
#
#   * a guess the model made mid-turn overwrote a fact the user had vouched for
#     by hand, and inherited the twenty successful turns the vouched-for fact
#     had earned -- arriving at the top of the next recall carrying the maximum
#     boost the system can grant, backed by evidence gathered about the claim
#     it replaced;
#   * the same agent, in the same turn, restating a fact in fewer words landed
#     a SECOND row (the ratio bar is `> 0.75` and a restatement that drops one
#     word scores exactly 0.75), so one claim held two slots and the shorter,
#     poorer sentence competed with the fuller one at every recall;
#   * two genuinely independent witnesses -- the machine's own scan and the
#     model's reading of it -- were collapsed into one row on word overlap
#     alone, which is the one merge that actually destroys evidence.
#
# So a fact now records the lineage it grew on, and the lineage is a PATH, not
# a label: `tree/branch/stem`.
#
#   tree    what kind of thing bore it at the root -- the user's own hand, the
#           machine's measurement of itself, the model's claim, or `import` for
#           a row that predates this column.
#   branch  the run of work it grew on. A delegation chain SHARES a branch,
#           which is the botanically right answer: a sub-agent's work is wood
#           off its parent's branch, not a separate tree.
#   stem    the individual agent on that branch that bore it -- delegation
#           depth. The parent's claims and its sub-agent's claims hang on the
#           same branch and different stems.
#
# Two facts are then related by WHERE THEIR LINEAGES DIVERGE (`_kin`), and that
# is what settles the collision:
#
#   stem    the same agent, in the same run, saying it twice.
#   branch  the same run; a sub-agent and its parent.
#   tree    the same kind of source, some other session.
#   grove   different trees -- two independent witnesses.
#
# Matthew 7:16-20 is the whole of the reasoning, and it cuts in both
# directions. "Ye shall know them by their fruits" is a rule about EVIDENCE: a
# tree's standing is measured from what has come of what it bore, never
# declared from what it calls itself, so nothing here says the user's word
# outranks the model's -- `_standing` counts fruit and reports what it counts.
# "A corrupt tree cannot bring forth good fruit" is the rule this fixes: a
# claim from a tree whose fruit has gone bad cannot be the source of good
# evidence, so it does not get to inherit any. And "every tree that bringeth
# not forth good fruit is hewn down" is read the way this module reads every
# other consequence -- what is cut away is the RANK, never the fact. Nothing is
# deleted, nothing is hidden, and the claim stands in the store as its own.
#
# Set false and the two rules go inert -- the newest wording wins the row and
# keeps everything, exactly as before. The lineage is still RECORDED, because
# where a fact came from is an observation about the write and not a policy
# about it, and a store that stopped noticing would have nothing to work with
# on the day the knob went back on.
_PROVENANCE = True

# Trees. `import` is the honest reading of a blank lineage -- a row written
# before this column existed came from somewhere, and the store does not know
# where. It is never evidence for or against any tree.
_TREE_USER = "user"        # a person vouched for it by hand
_TREE_MACHINE = "machine"  # the app measured it (ui.scan_apps, and its like)
_TREE_MODEL = "model"      # the model claimed it inside a turn
_TREE_IMPORT = "import"    # written before lineage was recorded

# The fixed lineages. A user's hand and the machine's own scan are single
# points of origin rather than runs of work, so their branch and stem say so
# instead of minting an id that would never be compared against anything.
_LINEAGE_USER = f"{_TREE_USER}/hand/hand"
_LINEAGE_MACHINE = f"{_TREE_MACHINE}/scan/scan"

# How much worse a tree has to stand before its fruit is barred from
# inheriting another tree's evidence. A margin rather than a bare `<` because
# `_standing` is a ratio over a handful of turns: without it, one unlucky turn
# on either side would flip which of two equally-proven trees may inherit, and
# a rule that turns on noise is worse than no rule. 0.15 is a sixth of the
# scale, so the demotion fires on a real gap in the fruit and not on drift.
_TREE_MARGIN = 0.15

# How many grafts one row remembers. A row grafted more than a handful of
# times is already pathological, and the record has to be bounded or a row
# rewritten in a loop grows a column without limit. The OLDEST is dropped when
# the cap is reached: the recent grafts are the recent evidence, and a record
# from twenty rewrites ago is the one least worth the space.
_GRAFT_KEEP = 4

# Branch ids are minted per run and have to be unique across PROCESSES, not
# just within one: the desktop app, the sidecar and the CLI all write to the
# same file, and two of them minting "b1" for unrelated runs would read back as
# one branch -- which is precisely the false kinship that would let a stranger's
# claim fold into a row it never grew on. A per-process nonce makes that
# impossible without coordinating between the processes.
_PROC = uuid.uuid4().hex[:6]
_branches: dict[int, str] = {}
_branch_seq = 0
_branch_lock = threading.Lock()

# Words whose referent lives outside the sentence. Read off the RAW text like
# `_polarity` is, because `_tokens` drops two-letter words and "he" is exactly
# the word that must not be dropped here.
#
# Checked against the OPENING word only. That is where the difference lies: a
# pronoun in the subject slot points at something the conversation supplied and
# the store did not ("He prefers the dark theme"), while one further in usually
# has its antecedent inside the same sentence ("Dave keeps his notes in
# Obsidian") -- flagging those would fire on half the store and teach the model
# to ignore the line.
_PRONOUNS = {
    "he", "she", "they", "him", "her", "them", "his", "hers", "their",
    "theirs", "it", "its",
}

# Demonstratives are split off because they carry a referent only as bare
# pronouns. "That is the one we picked" names nothing; "That laptop has 32GB"
# names a laptop and scores four tokens. So these flag only when a verb follows
# rather than a noun, approximated by the copulas and auxiliaries below --
# which is the whole difference between a demonstrative pronoun and a
# demonstrative determiner.
_DEMONSTRATIVES = {"this", "that", "these", "those"}
_COPULAS = {
    "is", "was", "are", "were", "will", "would", "should", "has", "have",
    "had", "does", "did", "can", "means", "meant", "works", "worked", "goes",
    "went", "stays", "stayed", "needs", "needed", "seems", "seemed",
}

# Fact ids recall served this turn, drained by `reinforce` at turn end. A set,
# so serving one fact twice in a turn counts once.
#
# Keyed by RUN, not by process. These were single module-level sets, which was
# right when one conversation was live at a time -- but the desktop now runs up
# to four chats side by side plus a phone session, and reinforce() drained
# everything unconditionally at every turn end. Whichever chat finished first
# graded every other chat's facts with its own outcome: a background automation
# that failed marked a fact the user's chat had just saved as unproven, and a
# success released facts it had never seen. Silent, cumulative, and systematic
# under exactly the parallel use the product advertises.
#
# The key is the run's stop event, which the registry already publishes on the
# dispatching thread -- and which sub-agents deliberately SHARE with their
# parent, so the intended "the parent's turn end drains what its sub-agent
# recalled" behaviour survives unchanged.
_served: dict[int, set[int]] = {}
_served_lock = threading.Lock()

# ...and the ids this turn WROTE, drained by the same hook. Separate set: what
# recall served is evidence the fact was useful, what remember planted is a
# claim with no evidence behind it yet, and the two earn different things.
_planted: dict[int, set[int]] = {}
_planted_lock = threading.Lock()


def _run_key() -> int:
    """Which run the calling thread is working for.

    The stop event is per-run and shared down a delegation chain, so it is the
    identity reinforcement wants. Falling back to the thread id keeps callers
    outside a dispatched run (the CLI, tests) working as before.
    """
    try:
        from .registry import registry

        stop = registry.current_stop()
        if stop is not None:
            return id(stop)
    except Exception:  # noqa: BLE001 - bookkeeping must never break a recall
        pass
    return threading.get_ident()

# Schema setup runs once per process; configure() resets it for a new state dir.
_SCHEMA_READY = False
# Guards the migration below. invalidate_schema() resets the latch while the
# desktop's worker threads are live (a backup restore does exactly that), so
# two first-connects can race: both read the same PRAGMA table_info, both
# decide a column is missing, and the loser's ALTER fails with 'duplicate
# column name' -- landing precisely in the just-restored window this
# mechanism exists to make safe.
_SCHEMA_LOCK = threading.Lock()

# Words too common to be useful for scoring.
_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "at", "for", "with", "my", "i", "me", "it",
    "that", "this", "what", "when", "how", "do", "does", "did", "about",
}


def invalidate_schema() -> None:
    """Force the next `_connect` to re-run the migration.

    The once-per-process latch below is only safe while the file stays the one
    this process migrated. Restoring a backup replaces it in place -- see
    `backup.restore_backup`, which calls this -- and a backup taken by an older
    build has an older schema. Without this, the latch kept asserting a `folder`
    column that the restored file does not have, and every `recall` raised
    `no such column` until the app was restarted.
    """
    global _SCHEMA_READY
    _SCHEMA_READY = False


def _connect() -> sqlite3.Connection:
    global _SCHEMA_READY
    if _DB is None:
        raise RuntimeError("Memory not initialized; call configure() first.")
    conn = sqlite3.connect(_DB)
    # Schema work runs once per process. Recall opens a connection per call and
    # was paying for a CREATE TABLE plus a PRAGMA introspection each time, on a
    # file the sidecar and the desktop UI hold open too. Anything that swaps the
    # file underneath a running process must call `invalidate_schema`.
    if _SCHEMA_READY:
        return conn
    # The schema work below can raise -- a corrupt file fails on the first
    # statement -- and the connection has to be CLOSED when it does. On POSIX a
    # leaked handle is invisible because a file can be renamed while open; on
    # Windows it cannot, so dbsafe.quarantine's replace() failed with
    # PermissionError, the original error was re-raised, and the app died on
    # exactly the corrupt database the quarantine exists to survive.
    try:
        return _prepare(conn)
    except BaseException:
        conn.close()
        raise


def _prepare(conn: sqlite3.Connection) -> sqlite3.Connection:
    global _SCHEMA_READY
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:   # another thread migrated while we waited
            return conn
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                category   TEXT NOT NULL DEFAULT 'fact',
                fact       TEXT NOT NULL,
                context    TEXT,
                created_at TEXT NOT NULL,
                folder     TEXT NOT NULL DEFAULT '',
                uses       INTEGER NOT NULL DEFAULT 0,
                wins       INTEGER NOT NULL DEFAULT 0,
                last_used  TEXT NOT NULL DEFAULT '',
                unproven   INTEGER NOT NULL DEFAULT 0,
                lineage    TEXT NOT NULL DEFAULT '',
                grafted    TEXT NOT NULL DEFAULT ''
            )
            """
        )
        # Migrate older databases that predate the folder column.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        if "folder" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN folder TEXT NOT NULL DEFAULT ''")
        # ...and the reinforcement tallies, added the same way. TEXT '' rather than
        # NULL for last_used keeps every row comparable in one code path.
        if "uses" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN uses INTEGER NOT NULL DEFAULT 0")
        if "wins" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN wins INTEGER NOT NULL DEFAULT 0")
        if "last_used" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN last_used TEXT NOT NULL DEFAULT ''")
        # ...and the held-back flag. DEFAULT 0 is the load-bearing part of this
        # migration: every fact already on disk was written before anything graded
        # the seed, so there is no evidence to hold any of them back on, and they
        # all keep behaving exactly as they did. The flag only ever gets set by a
        # turn that fails from here on.
        if "unproven" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN unproven INTEGER NOT NULL DEFAULT 0")
        # ...and the lineage. DEFAULT '' is load-bearing in the same way the
        # flag above defaults to 0, and for a stricter reason: a row already on
        # disk grew on a tree nobody recorded, so the honest value is "not
        # known" and NOT a plausible guess at which writer it came from. Every
        # reader below treats a blank as evidence about nothing -- `_kin`
        # returns "", `_standing` skips the row -- so an old database keeps
        # behaving exactly as it did while new writes start carrying a trace.
        if "lineage" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN lineage TEXT NOT NULL DEFAULT ''")
        # ...and what earlier trees bore on this row before it changed hands.
        # Blank on every existing row for the same reason `lineage` is: nothing
        # recorded a takeover before this, so the store does not know of any,
        # and inventing one would be worse than knowing of none.
        if "grafted" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN grafted TEXT NOT NULL DEFAULT ''")
        # Routed recall and the core digest both select on folder.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_folder ON memories(folder)")
        # Scattershot's learned bonds: which words led to which fact in a turn
        # that then worked. A separate table rather than columns on `memories`
        # because the relation is many-to-many and it is pruned on its own
        # schedule -- see `_write_bonds`. Purely derived data: dropping the
        # table costs the store nothing but the vocabulary it had learned, and
        # every older database simply starts with none.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bonds (
                term   TEXT NOT NULL,
                mem_id INTEGER NOT NULL,
                weight INTEGER NOT NULL DEFAULT 0,
                last   TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (term, mem_id)
            )
            """
        )
        # Deleting a fact deletes its bonds, and that lookup is by id.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bonds_mem ON bonds(mem_id)")
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass  # a filesystem that cannot do WAL still works, just serialised
        conn.commit()
        _SCHEMA_READY = True
    return conn


def configure(state_dir: Path, cfg: dict[str, Any] | None = None) -> None:
    global _DB, _FUSION, _RRF_K, _MIN_SCORE, _RECALL_LIMIT, _ROUTING, _PRELOAD_CORE
    global _SCHEMA_READY, _REINFORCE, _HOLD_UNPROVEN, _CUE_CHECK, _PROVENANCE
    global _SCATTER, _SHOT_BONDS, _SHOT_NEIGHBOURS, _SHOT_BOND_MIN, _SHOT_FILL
    cfg = cfg or {}
    _DB = Path(state_dir) / "memory.db"
    _SCHEMA_READY = False  # a new state dir is a different database
    # Reinforcement rides the turn-end learning hook (loop._remember_run), so
    # `learning = false` has to disable it too -- with the brain off, nothing
    # would ever drain the served set.
    _REINFORCE = bool(cfg.get("reinforcement", True)) and bool(cfg.get("learning", True))
    # Seed grading rides the same hook, so it needs the same two off switches
    # plus its own. Held back facts are only ever released by that hook, so with
    # it dark the flag has to be inert in BOTH directions -- see `preload`.
    _HOLD_UNPROVEN = bool(cfg.get("hold_unproven", True)) and _REINFORCE
    # NOT gated on learning or reinforcement, unlike the two above. Those grade
    # a fact by what a finished turn did with it, so they need the turn-end
    # hook to exist; this one reads the sentence the model just typed and needs
    # nothing but the sentence. Turning the brain off is a choice about
    # learning, not a request to start writing facts that cannot be found.
    _CUE_CHECK = bool(cfg.get("cue_check", True))
    # Not gated on anything either, and for a sharper version of the same
    # reason. The two grades above read a finished turn; this one reads where
    # the write came FROM, which is knowable at the moment of the write and
    # from nothing else. It also has to keep working with the brain off,
    # because the failure it prevents -- a claim from one source silently
    # inheriting the proof earned by another -- is not a learning feature, it
    # is the store handing back evidence that was never gathered about the
    # sentence it is attached to.
    _PROVENANCE = bool(cfg.get("provenance", True))
    with _branch_lock:
        _branches.clear()  # a different database owes nothing to this run
    # The spread reads nothing but the store, so unlike the two settings above
    # it is NOT gated on learning: with the brain off, recall should still be
    # able to find a fact the question words differently. Its BONDS are gated,
    # because those are graded by the same turn-end hook -- with that hook dark
    # nothing would ever write one, and a bond half-written is worse than none.
    _SCATTER = bool(cfg.get("scattershot", True))
    _SHOT_BONDS = bool(cfg.get("scatter_bonds", True)) and _REINFORCE
    _SHOT_NEIGHBOURS = max(0, int(cfg.get("scatter_neighbours", 4)))
    _SHOT_BOND_MIN = max(1, int(cfg.get("scatter_bond_min", 1)))
    _SHOT_FILL = max(1, int(cfg.get("scatter_fill", 3)))
    with _served_lock:
        _served.clear()  # a different database owes nothing to this turn
    with _planted_lock:
        _planted.clear()
    with _bonded_lock:
        _bonded.clear()
    _corpus_changed()  # a different store has a different vocabulary
    _FUSION = str(cfg.get("fusion", "rrf")).lower()
    _RRF_K = max(1, int(cfg.get("rrf_k", 60)))
    _MIN_SCORE = float(cfg.get("min_score", 0.0))
    _RECALL_LIMIT = max(1, int(cfg.get("recall_limit", 8)))
    _ROUTING = bool(cfg.get("sector_routing", True))
    _PRELOAD_CORE = max(0, int(cfg.get("preload_core", 8)))
    dbsafe.open_or_quarantine(_DB, lambda: _connect().close())
    backfill_folders()
    palace.configure(Path(state_dir), cfg)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


# Words that carry a claim's polarity. Read on the RAW text rather than through
# `_tokens`, which drops anything under three letters -- "no" is exactly the
# word that must not be dropped here.
_NEGATORS = {
    "not", "no", "never", "none", "nor", "cannot", "cant", "dont", "doesnt",
    "didnt", "wont", "isnt", "arent", "wasnt", "werent", "shouldnt", "wouldnt",
    "couldnt", "hasnt", "havent", "aint", "without", "avoid", "avoids",
    "stop", "stops", "stopped", "dislike", "dislikes", "hate", "hates",
    "refuse", "refuses", "disabled", "disable", "off", "false",
}


def _polarity(text: str) -> int:
    """0 if a sentence asserts, 1 if it denies -- negation words counted mod 2.

    Parity rather than presence, so a double negative reads as an assertion and
    two ways of saying the same denial agree: "does not use X" and "doesn't use
    X" both come out 1, so re-wording one into the other is not mistaken for a
    reversal. Apostrophes are stripped first, which is what folds "doesn't" onto
    the bare "doesnt" in the set.

    A three-word heuristic, and it does not pretend otherwise: it catches the
    negation flip, which is the reversal that survives a near-total token
    overlap, and it does not catch a swapped number or an antonym pair with no
    negator in it. Those score far below the dedup threshold anyway (two door
    codes share every word but the digits and land near 0.67), so they never
    reach the path this guards -- they are stored as the separate claims they
    are.

    It leans toward calling a reversal, and the asymmetry is deliberate: a
    false positive costs a fact a tally it can earn back over the next few
    turns, while a false negative hands the maximum boost in the system to a
    claim that contradicts the evidence behind it. Note also that only a
    CHANGE in parity counts, so a negator sitting in both texts -- the noun in
    "the bus stop near the office" -- is invisible here.
    """
    words = re.findall(r"[a-z]+", str(text or "").lower().replace("'", "").replace("\u2019", ""))
    return sum(1 for w in words if w in _NEGATORS) % 2


def _cue_gap(fact: str) -> str:
    """Why recall would struggle to find this fact later, or "" if it is fine.

    Two ways a sentence fails to carry its own cues, checked in the order they
    cost the most:

      REFERENT  the claim opens with a pronoun, so whom or what it is about
                lived in a conversation that ends with this turn. Note that
                such a fact can still MATCH -- "prefers the dark theme" scores
                three tokens -- and still be useless when it comes back,
                because what returns is a settled preference belonging to
                nobody. This is the failure that survives every downstream
                mechanism: reinforcement will happily promote it.

      CUES      fewer than `_CUE_MIN` distinct scoring tokens, so there is
                nearly nothing for a query to overlap with. Exact rather than
                heuristic -- counted with the same `_tokens` that `recall`
                ranks by, so it measures the real thing and not a proxy for it.

    The referent test is a heuristic and does not pretend otherwise: "that door
    code is 4415" is flagged and would have been findable. The asymmetry is
    deliberate and the costs are lopsided -- a false flag spends one line of a
    tool result the model is free to ignore, while a miss leaves a sentence
    about nobody in the store for good, read back as settled truth every time
    it surfaces.
    """
    if not _CUE_CHECK:
        return ""
    text = str(fact or "").strip()
    words = re.findall(r"[a-z]+", text.lower().replace("'", "").replace("\u2019", ""))
    head = words[0] if words else ""
    bare_demonstrative = (
        head in _DEMONSTRATIVES and (len(words) < 2 or words[1] in _COPULAS))
    if head in _PRONOUNS or bare_demonstrative:
        return (
            f"it opens with {head!r} and never names the subject, so nothing "
            "here says who or what it is about"
        )
    cues = _tokens(text)
    if len(cues) < _CUE_MIN:
        found = ", ".join(sorted(cues)) or "none at all"
        return (
            f"it carries almost nothing to search on ({found}), so recall can "
            "reach it only through a query that happens to use that word"
        )
    return ""


def _cue_note(fact: str) -> str:
    """The gap report as a line appended to `remember`'s own result.

    Phrased as the fix rather than the fault, and it names the consequence
    ("replaces this one") so that following it does not read like a risk of
    duplicating the fact -- which, before `_elaborates` below, it was.
    """
    gap = _cue_gap(fact)
    if not gap:
        return ""
    return (
        f"\nNOTE: saved exactly as written, but {gap}. Recall searches these "
        "sentences by their words, so a word this conversation supplied is a "
        "word it cannot search for. Call `remember` again with the subject "
        "named and the specific terms spelled out -- the fuller sentence "
        "replaces this one rather than sitting beside it."
    )


# How long a cue-poor fact stays open to being replaced by its own corrected
# wording. The containment path below exists for ONE moment -- the model is
# told, on the tool result, that the sentence it just wrote names nobody, and
# writes it again spelled out while the referent is still on screen. Outside
# that moment containment is not evidence of anything: see `_elaborates`.
_ELABORATE_WINDOW_S = 3600.0


def _elaborates(new_tokens: set[str], old_fact: str, old_stamp: Any = None) -> bool:
    """True when a new fact is the same claim as `old_fact`, finally said properly.

    Without this, the note above is advice that makes the store worse. The
    dedup pass in `remember` treats two facts as one when they share more than
    75% of their tokens, and closing a referent gap is precisely the edit that
    misses that bar: adding one name to a three-token sentence scores exactly
    0.75, which is not greater than 0.75. So "Dave prefers the dark theme"
    landed BESIDE "He prefers the dark theme" -- two rows, one claim, and the
    vague original still holding its place in the prompt digest. A model that
    ignored the note left the store in better shape than one that followed it.

    Containment rather than ratio, and gated on the original being cue-poor:
    every word of the old fact survives in the new one (so this is the same
    claim, not a different one that happens to overlap), the new one adds at
    least one word (so it is not a re-save of the same text), and the old one
    had a gap worth closing (so an ordinary fact is never swallowed by a longer
    sentence that happens to contain its words). Outside that gate nothing
    changes and the ratio still decides everything.

    Note what the gate rules out on its own: a fact with NO scoring tokens is
    contained by every sentence in the language, so without the emptiness check
    the next unrelated write would absorb it.

    What containment does NOT establish is IDENTITY OF SUBJECT, and on this
    path that is the whole question. A REFERENT-gap fact's subject is precisely
    the word that is not among its tokens, so "he prefers the dark theme" is
    contained by "Priya prefers the dark theme in every editor" exactly as
    neatly as by "Dave prefers..." -- and if the first was about Dave, agreeing
    with Priya deletes it. Two real preferences, one row, no note, nothing on
    screen to say a fact was dropped.

    No cheap test resolves a pronoun, so the answer is the WINDOW. This path is
    justified by one specific moment: the model is told on the tool result that
    the sentence it just wrote names nobody, and writes it again spelled out
    while the referent is still on screen. `_cue_note` promises exactly that
    ("the fuller sentence replaces this one"). Outside `_ELABORATE_WINDOW_S` of
    that moment the justification is gone and only the coincidence of words is
    left, so the ratio decides it like anything else. A vague fact from last
    month stays a vague fact -- which is what it already was -- instead of
    becoming a licence to overwrite it with the next sentence containing its
    words.

    A reversal is deliberately NOT excluded here: naming the subject while
    flipping the claim is still one correction, and the reset in `remember`
    is what stops it inheriting evidence gathered for its opposite.
    """
    if not _CUE_CHECK:
        return False
    old_tokens = _tokens(old_fact)
    if not old_tokens or not _cue_gap(old_fact):
        return False
    if old_stamp is not None and _age_seconds(old_stamp) > _ELABORATE_WINDOW_S:
        return False
    return old_tokens < new_tokens  # proper subset: contained, and adds at least one


def _age_seconds(stamp: Any) -> float:
    """Seconds since an ISO stamp, or inf when it cannot be read.

    Unreadable reads as OLD rather than fresh, so a stamp this cannot parse
    closes the replacement window instead of opening it. The failure that
    matters here is deleting a fact that should have been kept.
    """
    try:
        return max(0.0, (datetime.now() - datetime.fromisoformat(str(stamp))).total_seconds())
    except (TypeError, ValueError):
        return float("inf")


def _turn_is_live() -> bool:
    """True when this thread is inside a running turn.

    The loop publishes its stop event on the thread for the length of a run
    (`Agent.run` sets it, `_remember_run` clears it), which makes it the one
    signal already on hand for "the model is writing this, mid-turn" as opposed
    to the app writing it from a worker thread -- `ui.scan_apps` folds the
    installed-app digest in that way, and an index the user asked for is not a
    claim that needs proving.
    """
    try:
        return registry.current_stop() is not None
    except Exception:  # noqa: BLE001 - a missing thread-local is just "no turn"
        return False


def _depth() -> int:
    """How far down a delegation chain this thread is. 0 is the parent agent.

    Imported lazily for the same reason `_run_key` imports the registry lazily:
    `tools.subagent` builds its catalogue from the registry this module
    registers into, and a top-level import would close the ring.
    """
    try:
        from .tools import subagent

        return int(subagent._depth())  # noqa: SLF001 - the depth is per-thread state
    except Exception:  # noqa: BLE001 - unknown depth is the parent's, which is the safe read
        return 0


def _branch_id(run: int) -> str:
    """A stable id for the run of work this write grows on.

    Minted rather than derived from the run key, which is `id(stop_event)` and
    therefore RECYCLED: CPython hands a freed object's address to the next one,
    so two unrelated runs an hour apart can share a key, and a stored branch id
    built from it would read back as the same branch. The ids here are handed
    out once each and never reused, so a branch means one run for good.

    Bounded, because a long-lived desktop session runs thousands of turns and a
    map that only grows is a leak with a nice name. `reinforce` drops a run's
    entry at turn end; the cap is what covers a run that never reaches one.
    """
    global _branch_seq
    with _branch_lock:
        got = _branches.get(run)
        if got is not None:
            return got
        if len(_branches) > 512:
            _branches.clear()  # cheaper than an LRU, and a lost id only costs kinship
        _branch_seq += 1
        _branches[run] = f"{_PROC}{_branch_seq:x}"
        return _branches[run]


def _drop_branch(run: int) -> None:
    """Forget a finished run's branch id. Called from `reinforce`."""
    with _branch_lock:
        _branches.pop(run, None)


def _lineage() -> str:
    """`tree/branch/stem` for the write happening on this thread.

    Read from what the process already knows rather than from anything the
    caller passes, which is the point: a writer cannot describe its own
    provenance any more than a wolf can describe its own coat. `_turn_is_live`
    separates the model's claim from the app's measurement, and `_depth`
    separates a sub-agent's stem from its parent's on the shared branch.
    """
    if not _turn_is_live():
        # Not the model's guess at all -- `ui.scan_apps` folding the installed
        # -app digest in from a worker thread is the app reporting what it
        # measured, which is the same reason `_plant` skips it.
        return _LINEAGE_MACHINE
    run = _run_key()
    return f"{_TREE_MODEL}/{_branch_id(run)}/d{_depth()}"


def _tree_of(lineage: Any) -> str:
    """The tree at the root of a stored lineage; `import` for a blank one."""
    head = str(lineage or "").split("/", 1)[0]
    return head if head in (_TREE_USER, _TREE_MACHINE, _TREE_MODEL) else _TREE_IMPORT


def _kin(new: Any, old: Any) -> str:
    """Where two lineages diverge, traced back from the stem.

    One of "stem", "branch", "tree", "grove", or "" when either side has no
    lineage to trace. The empty answer is not "unrelated" -- it is "not known",
    and every caller below treats it as a reason to change nothing, because a
    row written before this column existed is evidence about nothing.
    """
    a, b = str(new or "").split("/"), str(old or "").split("/")
    if len(a) != 3 or len(b) != 3 or "" in a or "" in b:
        return ""
    if a == b:
        return "stem"
    if a[0] == b[0] and a[1] == b[1]:
        return "branch"
    if a[0] == b[0]:
        return "tree"
    return "grove"


def _restates(new_tokens: set[str], old_tokens: set[str]) -> bool:
    """True when the new fact says strictly less than one already stored.

    A proper subset, both ends non-empty: every word the new sentence can be
    searched by is already in the old one, and the old one carries at least one
    more. There is nothing in the newer sentence that the store does not
    already hold, so writing it as its own row adds a second claim that says
    less and then competes with the fuller one for the same recall slots.

    This is the exact shape the ratio bar cannot see. Dropping one word from a
    four-token fact scores 3/4 = 0.75 and the bar is `> 0.75`, so the pile-up
    the ratio exists to prevent happens most reliably at its own edge.

    Direction matters and only this direction is safe. A new fact CONTAINING an
    old one is a different, richer claim ("...in the back studio" is a place the
    store did not know), and folding those together would lose the detail -- so
    that direction stays with `_elaborates`, which pays for it with a cue-poor
    gate and an hour-long window. Containment alone never licenses a merge here
    either: the caller also requires the two to share a stem or a branch, so
    this only ever folds an agent's own restatement back into its own claim.
    """
    return bool(new_tokens) and bool(old_tokens) and new_tokens < old_tokens


def _grafts(text: Any) -> list[tuple[str, int, int]]:
    """Read a row's graft record: what each earlier tree had borne on it.

    `tree:uses:wins`, semicolon-separated. Anything unparseable is skipped
    rather than raising -- this is derived bookkeeping, and a column a hand
    edit has mangled must cost the store a measurement, never a recall.
    """
    out: list[tuple[str, int, int]] = []
    for part in str(text or "").split(";"):
        bits = part.split(":")
        if len(bits) != 3:
            continue
        tree, uses, wins = bits
        if tree not in (_TREE_USER, _TREE_MACHINE, _TREE_MODEL):
            continue
        try:
            out.append((tree, int(uses), int(wins)))
        except ValueError:
            continue
    return out


def _graft_record(existing: Any, tree: str, uses: Any, wins: Any) -> str:
    """Add one graft to a row's record, oldest dropped at `_GRAFT_KEEP`.

    Written when a cross-tree takeover moves a row off the tree that grew it,
    and ONLY when that tree had actually borne something on it: a row with no
    finished fruit has nothing to remember, so the column stays empty in the
    ordinary case and fills only where a tree genuinely loses proven fruit.
    """
    if not int(uses or 0) and not int(wins or 0):
        return str(existing or "")
    if tree not in (_TREE_USER, _TREE_MACHINE, _TREE_MODEL):
        # A row of unknown origin has no tree to leave a record for. The caller
        # checks this too; refusing it here as well keeps the column free of
        # entries `_grafts` would only skip on the way back out.
        return str(existing or "")
    kept = _grafts(existing)[-(_GRAFT_KEEP - 1):] if _GRAFT_KEEP > 1 else []
    kept.append((tree, int(uses or 0), int(wins or 0)))
    return ";".join(f"{t}:{u}:{w}" for t, u, w in kept)


def _standing(rows: list[tuple]) -> dict[str, float]:
    """Each tree's standing in [0, 1), measured only from the fruit it bore.

    Takes `(lineage, uses, wins, unproven, grafted)` rows rather than a
    connection, so `remember` can grade the trees from the single table scan it
    already makes on its way to the dedup pass instead of paying for a second
    one.

    "By their fruits ye shall know them" is a rule about evidence before it is
    anything else, and this is the whole of what the rule permits: count what
    became of the facts a tree has borne, and report that. Nothing here knows
    that a user is more reliable than a model, and nothing here is allowed to
    -- a standing asserted rather than counted is the sheep's clothing the
    passage is warning about, and it would let any writer claim its way to the
    top of the store by declaring itself.

    Two terms, both already meaning something in this module:

      the win share  sum(wins) / (sum(uses) + 2), the same Laplace-shrunk ratio
                     `_familiarity` uses, so no new scale is invented and the
                     result is directly comparable to a per-fact one.
      what is held   the share of the tree's facts NOT currently held out of
                     the prompt. A tree half of whose fruit was written in
                     turns that fell over is standing on half the ground.

    A tree with no finished fruit is ABSENT from the result rather than scored
    zero. Those are different facts -- "has borne nothing yet" is not "has
    borne badly" -- and a caller that cannot tell them apart would demote every
    first write in a fresh store. This is Law 2 in the blueprint: a dash, never
    a guess.

    Counted from the fruit a tree HAS BORNE, which is not the same as the fruit
    it still holds, and the difference is the whole of what `grafted` exists
    for. The first version of this counted current rows only, and that made the
    measurement erasable by the very act it guards against: a takeover moves
    the row's lineage to the tree that now bears the sentence, so a tree whose
    facts were taken kept fewer and fewer, down to none and out of this result
    -- and the more a worse-standing tree overwrote, the less evidence was left
    that it was the worse tree. A rule that can be dismantled by the behaviour
    it exists to catch is not a rule.

    So a takeover leaves the record on the row it took: which tree had grown
    that fruit, and how much of it there was at the moment it changed hands
    (`_graft_record`). That is not the shadow ledger this docstring used to
    argue against -- there is no second table, nothing accrues anywhere a
    reader cannot see, and every number here is still read off the `memories`
    rows themselves. It is the row carrying its own history, exactly as `uses`,
    `wins` and `unproven` already do.

    And it is the honest reading besides. "Ye shall know them by their fruits"
    is past tense: the turns a tree's fruit won were really won, and another
    tree overwriting the sentence afterwards does not un-win them. What a tree
    is holding right now still dominates the number -- a graft is one row's
    worth of history against however many facts the tree currently bears, and
    its own bad fruit drags it down through the same `held` term as ever.
    """
    tally: dict[str, list[int]] = {}
    for lineage, uses, wins, held, grafted in rows:
        tree = _tree_of(lineage)
        if tree != _TREE_IMPORT:  # a row of unknown origin is evidence about no tree
            acc = tally.setdefault(tree, [0, 0, 0, 0])
            acc[0] += int(uses or 0)
            acc[1] += int(wins or 0)
            acc[2] += 1
            acc[3] += 1 if held else 0
        # ...and the fruit this row bore for the trees that held it before.
        # Never counted as held: a graft is only written when there was proven
        # fruit to remember, and being taken from a tree is not that tree
        # having borne badly.
        for gtree, guses, gwins in _grafts(grafted):
            gacc = tally.setdefault(gtree, [0, 0, 0, 0])
            gacc[0] += guses
            gacc[1] += gwins
            gacc[2] += 1
    out: dict[str, float] = {}
    for tree, (uses, wins, borne, held) in tally.items():
        if not uses and not held:
            continue  # nothing has come of it either way
        out[tree] = (wins / (uses + 2)) * ((borne - held) / borne)
    return out


# A category picks the top-level folder; the first keyword theme a fact hits
# picks a subfolder under it. So memories self-organize into folders and
# subfolders instead of piling into one flat list, and the user can still move
# anything by hand afterwards.
_CATEGORY_FOLDER = {
    "preference": "Preferences", "decision": "Decisions",
    "context": "Context", "fact": "Facts",
}
_THEME_FOLDERS: list[tuple[str, set[str]]] = [
    ("Development", {
        "code", "coding", "git", "github", "python", "node", "npm", "npx", "api",
        "docker", "kubernetes", "build", "compile", "repo", "repository", "javascript",
        "typescript", "rust", "golang", "java", "sql", "database", "terminal", "shell",
        "bash", "powershell", "vscode", "ide", "framework", "library", "deploy", "backend",
        "frontend", "server", "linux",
    }),
    ("Audio & music", {
        "audio", "music", "daw", "plugin", "vst", "ableton", "izotope", "mixing", "mix",
        "master", "mastering", "track", "song", "synth", "midi", "reaper", "cubase",
        "steinberg", "tonal",
    }),
    ("Apps & tools", {
        "app", "apps", "program", "programs", "launch", "install", "installed", "exe",
        "software", "tool", "tools", "shortcut",
    }),
    ("Files & paths", {
        "folder", "folders", "path", "paths", "directory", "drive", "disk", "file",
        "files", "download", "downloads", "documents",
    }),
    ("System & PC", {
        "computer", "windows", "gpu", "cpu", "ram", "driver", "drivers", "network",
        "wifi", "vpn", "monitor", "display", "hardware", "printer",
    }),
    ("Work", {
        "work", "project", "projects", "client", "clients", "deadline", "meeting",
        "company", "team", "job", "invoice",
    }),
    ("Personal", {
        "name", "birthday", "email", "phone", "address", "family", "friend", "hobby",
        "pet", "favorite", "favourite",
    }),
]


def auto_folder(fact: str, category: str) -> str:
    """Pick a folder path for a fact from its category and words."""
    top = _CATEGORY_FOLDER.get(category, "Facts")
    words = {w for w in re.findall(r"[a-z0-9]+", (fact or "").lower())}
    for name, keys in _THEME_FOLDERS:
        if words & keys:
            return f"{top}/{name}"
    return top


# ------------------------------------------------------------------ sectors
# The theme above a fact's folder is its SECTOR -- the "brain" it belongs to.
# The split already existed as data (every row is tagged on write, and
# backfill_folders() tags the old ones); until now nothing read it back, so
# recall scanned every fact and preload shipped 25 arbitrary ones to the model.
#
# Routing reads it. A request is scored against the same keyword sets that
# assigned the folders, and recall then searches the matching sector instead of
# the whole store. That is where the context saving comes from: the facts about
# your DAW stop competing for slots with a question about git.
#
# Deliberately keyword-scored rather than model-routed. Picking the sector must
# cost less than searching without it, and a dictionary lookup cannot stall the
# way a 7B classifier call can -- same reasoning as the rules in vector/route.py.


def sector_of(folder: str) -> str:
    """The sector a folder belongs to -- the part under the category, or ''.

    A fact that matched no theme keeps a bare top-level folder ("Facts"), and
    that empty sector is meaningful: it is what makes the fact CORE (see
    `preload`), not merely unclassified.
    """
    parts = [p for p in str(folder or "").split("/") if p]
    return parts[1] if len(parts) > 1 else ""


# The routing vocabulary: each theme's keywords PLUS the words of its own name.
# The filing side (`auto_folder`) deliberately keeps using the bare keyword sets
# -- widening it would re-file facts and invalidate folders already on disk --
# so the two vocabularies are no longer identical. That is safe now only because
# routing is a ranking signal rather than a filter (see `recall`): a theme the
# router misses costs a fact its bonus, never its place in the results.
#
# The names have to be in here because the system prompt PRINTS them. `preload`
# ships only core facts and `prompt.build` tells the model "more is filed by
# subject: Development, Personal, ..." -- so those exact strings come back as
# recall queries, and "Development", "Personal" and "System & PC" appear in no
# keyword set. Advertising a key the lookup cannot resolve is the failure this
# closes.
_THEME_ROUTING: list[tuple[str, set[str]]] = [
    (name, keys | {w for w in re.findall(r"[a-z0-9]+", name.lower())})
    for name, keys in _THEME_FOLDERS
]


def route_sectors(text: str, limit: int = 2) -> list[str]:
    """Sectors a request most likely concerns, strongest first.

    Scored on raw words rather than `_tokens`, to match `auto_folder`: the
    filing side counts every word, so the routing side has to as well or a fact
    filed under a theme loses its bonus through it.
    """
    words = {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())}
    if not words:
        return []
    scored = []
    for name, keys in _THEME_ROUTING:
        hits = len(words & keys)
        if hits:
            scored.append((hits, name))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [name for _, name in scored[: max(1, int(limit))]]


def sector_index() -> str:
    """One line naming the sectors that hold facts, for the system prompt.

    This is what replaces the fat digest of fact text. Names only, no counts:
    a count changes on every `remember`, and anything in the system prompt that
    changes per write invalidates the KV prefix -- the exact cache-miss the
    day-resolution clock in prompt.py exists to avoid. Sector names change only
    when a theme gets its first fact, so the line stays byte-identical for
    weeks at a time.

    Empty when routing is off: the line's whole claim is that those facts are
    filed away and reachable through `recall` rather than printed above it, and
    with routing off `preload` does print them. Asserting both would be a lie
    the model then acts on.
    """
    if not _ROUTING:
        return ""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT DISTINCT folder FROM memories WHERE folder LIKE '%/%'"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return ""
    names = sorted({sector_of(r[0]) for r in rows if sector_of(r[0])})
    if not names:
        return ""
    return ", ".join(names)


# ------------------------------------------------------------- scattershot
# Recall's tight shot is a slug: `_keyword_rank` scores a fact only when a word
# of the question appears literally in the sentence. It lands on the stored
# wording or it misses, and on a miss the store goes quiet about facts it
# plainly holds. Ask "what GPU do I have" of a store that knows "Dave runs the
# training scripts on the RTX 4090" and the answer is "No memories matching",
# because the question and the fact share no word. Worse, when the query names
# a subject the fallback below used to answer from the routed sector BY RECENCY
# -- so "how do we ship the backend" came back with the newest fact in
# Development, presented in the same format as a real hit, with nothing to mark
# it a guess.
#
# The `remember` tool description tries to close this from the writing side --
# "include the terms a future question would actually use" -- which asks the
# model to guess, at write time, the vocabulary of a question nobody has asked
# yet. That is the wrong end. This closes it from the READING side: one query
# is fired as a SPREAD of related terms, each derived from the store itself,
# and every fact a pellet touches becomes a candidate. Four kinds of pellet,
# strongest evidence first:
#
#   BOND       a word that has led to this exact fact in a turn that then
#              WORKED. Not a guess about language at all -- a record of what
#              your own successful turns did. This is what bridges "gpu" to
#              "RTX 4090": nothing in either string connects them, and one
#              good turn does. See `_bond_hits`.
#   STEM       a stored word sharing a prefix with a query word, which folds
#              "allergy" onto "allergic" and "deploy" onto "deployment".
#   NEIGHBOUR  a word the store's own facts keep in the same sentence as a
#              query word. Corpus-derived, so it learns your vocabulary and
#              nobody else's: "terraform" pulls "pipeline" because a fact YOU
#              wrote put the two together.
#   SECTOR     the routed theme's vocabulary -- the widest, weakest pellet.
#
# THE CHOKE, which is the whole reason a wider spread is safe to switch on. A
# shotgun is only useful if it does not also fill the room, and recall's cap is
# what keeps the prompt honest. So the spread does not get to touch it. It is
# strictly a FILLER: the fused keyword+semantic result is computed exactly as
# before, kept in exactly its old order, and the spread is only ever allowed
# into slots that came back EMPTY. Switching scattershot on therefore cannot
# reorder, displace or evict one hit the tight shot found, and cannot add one
# token past the `limit` that already bound it. What it can do is answer the
# miss -- where the alternative was silence, or that recency guess wearing a
# hit's clothes.
#
# Deliberately corpus-derived rather than dictionary-derived, for the same
# reason `route_sectors` is keyword-scored rather than model-routed: a spread
# built from a shipped thesaurus knows English, and what a recall needs to know
# is THIS store. Nothing here reaches outside ~/.ranger.
_SCATTER = True
_SHOT_STEM_MIN = 4      # shortest prefix that can ever count as the same word
_SHOT_STEM_SUFFIX = 2   # ...and the longest ending two such words may differ by
_SHOT_STEM_GROW = 4     # ...and the most the longer of the two may add in total
_SHOT_NEIGHBOURS = 4    # co-occurring terms pulled in per query word
_SHOT_MIN_COOC = 2      # facts two words must share before they count as near
_SHOT_BONDS = True      # learned query-term -> fact bonds
_SHOT_BOND_MIN = 1      # bond weight before it fires -- one good turn
_SHOT_FILL = 3          # most empty slots the spread may fill in one recall
_SHOT_FLOOR = 0.5       # ...and the least a pellet hit may score against the best
_BOND_CAP = 4000        # rows kept in `bonds`; the weakest are pruned first

# Pellet weights. These order the spread's OWN candidates and nothing else --
# no pellet score is ever compared against a keyword overlap, because the two
# never compete for a slot (see THE CHOKE). Ordering by KIND is the point: a
# fact reached through a bond your own successful turns recorded should arrive
# ahead of one reached through a shared word-stem, which should arrive ahead of
# one reached only by belonging to the same broad theme. The absolute numbers
# mean nothing outside this dict; the order between them means everything.
_SHOT_W = {"bond": 1.0, "stem": 0.55, "neighbour": 0.35, "sector": 0.15}

# The corpus maps are rebuilt when the store changes, not when it is read.
# Recall is hot and already materialises every row, so building the maps costs
# no extra database work -- but building them on EVERY recall would cost the
# O(tokens^2) pair walk every time, and the answer is identical until a write
# lands. `_corpus_changed` is called from every path that adds, edits or
# removes a fact.
_CORPUS_GEN = 0
_CORPUS_CACHE: tuple[int, dict[str, set[str]], dict[str, dict[str, int]]] | None = None
_CORPUS_LOCK = threading.Lock()

# Bonds observed this turn, keyed by run, graded by `reinforce` -- the same
# shape and the same lifecycle as `_served` and `_planted` above.
_bonded: dict[int, set[tuple[str, int]]] = {}
_bonded_lock = threading.Lock()


def _corpus_changed() -> None:
    """A write has landed, so the vocabulary and neighbour maps are stale."""
    global _CORPUS_GEN, _CORPUS_CACHE
    with _CORPUS_LOCK:
        _CORPUS_GEN += 1
        _CORPUS_CACHE = None


def _corpus_maps(rows: list[tuple]) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    """(stem index, neighbour counts) over every stored fact.

    The stem index maps a prefix to the stored words that start with it, which
    is what makes the STEM pellet a dictionary lookup rather than a scan. The
    neighbour counts are plain co-occurrence: how many separate facts put two
    words in the same sentence. Counted once per fact and symmetrically, so a
    word repeated within one sentence cannot inflate its own bonds.
    """
    global _CORPUS_CACHE
    cached = _CORPUS_CACHE
    if cached is not None and cached[0] == _CORPUS_GEN:
        return cached[1], cached[2]
    stems: dict[str, set[str]] = {}
    near: dict[str, dict[str, int]] = {}
    for row in rows:
        toks = _tokens(row[2])
        for t in toks:
            if len(t) >= _SHOT_STEM_MIN:
                stems.setdefault(t[:_SHOT_STEM_MIN], set()).add(t)
        ordered = sorted(toks)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                pa = near.setdefault(a, {})
                pa[b] = pa.get(b, 0) + 1
                pb = near.setdefault(b, {})
                pb[a] = pb.get(a, 0) + 1
    with _CORPUS_LOCK:
        _CORPUS_CACHE = (_CORPUS_GEN, stems, near)
    return stems, near


def _digits(word: str) -> str:
    """The digits in a word, in order -- "rtx4090" -> "4090", "deploy" -> ""."""
    return "".join(c for c in word if c.isdigit())


def _same_stem(a: str, b: str) -> bool:
    """True when two words differ only by an ending -- allergy/allergic.

    A bare `a[:4] == b[:4]` is what this replaced, and it is far too coarse on
    long words: every token beginning "sprocket" collapses to "spro", so a
    query about one of them draws a pellet from all of them. The bar therefore
    scales with the word -- the shared prefix must reach within
    `_SHOT_STEM_SUFFIX` of the SHORTER word's full length -- so a short word
    still folds on a short prefix ("deploy"/"deployment") while a long one has
    to agree very nearly all the way ("sprocket01ed" does not reach
    "sprocket02ing"). `_SHOT_STEM_MIN` stays as the floor, because at three
    characters "car", "card" and "care" are not the same word by any rule.
    """
    if a == b:
        return True
    # Digits are never an ending. A personal store is full of words that differ
    # only in their numbers -- rtx4090 and rtx4080, gpt4 and gpt5, port8080 and
    # port8081, v1 and v2 -- and every one of them is a DIFFERENT thing wearing
    # the same prefix. The length rule alone folds them together (seven
    # characters tolerate a two-character ending, which is exactly the size of
    # a model number), so the two most confusable words in the store would be
    # treated as one. A suffix that changes a digit changes the referent.
    if _digits(a) != _digits(b):
        return False
    # A short query word is its own whole prefix, so the length rule below
    # degenerates into "is one a prefix of the other" and every compound
    # beginning with it comes back: `type` reached `typescript`, and a store
    # asked for a blood type answered with the language the frontend is
    # written in. An ENDING is a few characters, not a second word, so cap
    # what the longer one may add -- deploy/deployment (four) is a conjugation,
    # type/typescript (six) is a coincidence of spelling.
    if abs(len(a) - len(b)) > _SHOT_STEM_GROW:
        return False
    need = max(_SHOT_STEM_MIN, min(len(a), len(b)) - _SHOT_STEM_SUFFIX)
    if len(a) < need or len(b) < need:
        return False
    return a[:need] == b[:need]


def _pellets(query: str, rows: list[tuple], routed: list[str]) -> dict[str, float]:
    """The spread: terms worth searching for besides the ones actually asked."""
    q = _tokens(query)
    if not q:
        return {}
    stems, near = _corpus_maps(rows)
    shot: dict[str, float] = {}

    def _add(term: str, weight: float) -> None:
        # A term the query already contains is not a pellet -- the tight shot
        # has it covered, and crediting it here would let the spread re-score
        # the same evidence under a second name. A term reachable two ways
        # keeps the STRONGER pellet rather than the sum: adding would let a
        # word that is merely common in the store accumulate past a bond,
        # which is the one ordering these weights exist to prevent.
        if term in q:
            return
        if shot.get(term, 0.0) < weight:
            shot[term] = weight

    for t in q:
        if len(t) >= _SHOT_STEM_MIN:
            # The prefix bucket is candidate generation only -- cheap, and
            # deliberately loose. `_same_stem` is the actual test.
            for word in stems.get(t[:_SHOT_STEM_MIN], ()):
                if _same_stem(t, word):
                    _add(word, _SHOT_W["stem"])
        if _SHOT_NEIGHBOURS:
            ranked = sorted(near.get(t, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            for word, seen in ranked[:_SHOT_NEIGHBOURS]:
                if seen >= _SHOT_MIN_COOC:
                    _add(word, _SHOT_W["neighbour"])
    if routed:
        for name, keys in _THEME_ROUTING:
            if name in routed:
                for word in keys:
                    if len(word) > 2 and word not in _STOP:
                        _add(word, _SHOT_W["sector"])
    return shot


def _bond_hits(query: str) -> dict[int, float]:
    """Facts your own successful turns tie to the words of this question.

    The one pellet that is not a guess about language. Every other kind reasons
    from the shape of words; this one reads a tally of what actually worked,
    which is why it outranks them and why it is the only pellet that can bridge
    two strings with nothing in common. Written by `reinforce`, and only ever
    for a turn that ENDED WELL -- see `_bond_pairs`.
    """
    if not (_SHOT_BONDS and _SCATTER):
        return {}
    q = _tokens(query)
    if not q:
        return {}
    try:
        conn = _connect()
        marks = ",".join("?" * len(q))
        rows = conn.execute(
            f"SELECT mem_id, SUM(weight) FROM bonds WHERE term IN ({marks}) "
            f"GROUP BY mem_id HAVING SUM(weight) >= ?",
            (*sorted(q), _SHOT_BOND_MIN),
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return {}   # the spread degrades to its string-shaped pellets
    return {int(r[0]): float(r[1]) for r in rows}


def _scatter_rank(query: str, rows: list[tuple], routed: list[str],
                  exclude: set[int]) -> list[int]:
    """Ids the tight shot missed, best pellet-score first.

    `exclude` is what the fused result already served, so the spread can only
    ever propose something new -- it has no way to reorder what it is given.
    """
    if not _SCATTER:
        return []
    shot = _pellets(query, rows, routed)
    bonds = _bond_hits(query)
    if not shot and not bonds:
        return []
    scored: list[tuple[float, int]] = []
    for row in rows:
        rid = row[0]
        if rid in exclude:
            continue
        # Walk the fact's tokens against the spread, not the spread against
        # the facts: the spread can run to a hundred terms on a routed query
        # and a fact is a sentence.
        hit = sum(shot[t] for t in _tokens(row[2]) if t in shot)
        # A bond's weight counts the successful turns behind it, capped so a
        # well-worn bond sorts above a fresh one without becoming unbeatable.
        bond = bonds.get(rid, 0.0)
        if bond:
            hit += _SHOT_W["bond"] * min(bond, 5.0)
        if hit <= 0.0:
            continue   # gated on real pellet evidence, before any tiebreak
        # Tiebreaks only, and sized as such -- at or under the weakest pellet,
        # so they order facts the spread already reached and never admit one.
        if routed and sector_of(row[4]) in routed:
            hit += _SHOT_W["sector"]
        if _REINFORCE:
            hit += _familiarity(row[5], row[6], row[7]) * _SHOT_W["sector"]
        scored.append((hit + rid / 1e6, rid))
    scored.sort(key=lambda t: t[0], reverse=True)
    # An empty slot is not an obligation to fill it. Ranked pellet evidence
    # falls away fast -- the best candidate is usually reached by a bond or a
    # stem and the eighth by one incidental word of a subject vocabulary -- so
    # filling every slot spends real prompt on facts that are barely more than
    # noise, and teaches the model to distrust the ones above them. Two bounds,
    # both relative to what this query actually found: at most `_SHOT_FILL`
    # slots, and nothing scoring under `_SHOT_FLOOR` of the best pellet hit.
    if not scored:
        return []
    floor = scored[0][0] * _SHOT_FLOOR
    return [rid for score, rid in scored[:_SHOT_FILL] if score >= floor]


def _bond_pairs(query: str, served: list[int], by_id: dict[int, tuple]) -> None:
    """Note which words led to which facts, for `reinforce` to grade.

    Only words the fact does NOT already contain are worth a bond: a word the
    sentence carries is one the tight shot can already find it by, and bonding
    it would fill the table with rows that never change an outcome. What is
    left is exactly the interesting case -- the question's vocabulary that the
    fact itself has no way to match.

    Recorded per recall rather than per turn, so a turn that calls recall three
    times bonds each question to the facts THAT question served, instead of
    every word in the turn to every fact in it.

    Marked here and graded later, which is the same contract `_served` has and
    deliberately not the one `_plant` has. `_plant` gates on `_turn_is_live`
    because it grades a WRITE, and the app writes facts from worker threads
    that no turn is going to grade. Nothing calls `recall` but a turn, and the
    marks a turn never grades are drained by `discard_turn` exactly as the
    served ones are.
    """
    if not (_SHOT_BONDS and _SCATTER):
        return
    q = _tokens(query)
    if not q:
        return
    pairs = {(t, rid) for rid in served for t in q - _tokens(by_id[rid][2])}
    if not pairs:
        return
    with _bonded_lock:
        _bonded.setdefault(_run_key(), set()).update(pairs)


def _write_bonds(conn: sqlite3.Connection, pairs: list[tuple[str, int]],
                 stamp: str) -> None:
    """Credit this turn's bonds, then keep the table bounded."""
    for term, rid in pairs:
        conn.execute(
            "INSERT INTO bonds (term, mem_id, weight, last) VALUES (?,?,1,?) "
            "ON CONFLICT(term, mem_id) DO UPDATE SET weight = weight + 1, last = ?",
            (term, int(rid), stamp, stamp),
        )
    # Unbounded, this table grows with every successful turn forever. Pruning
    # the WEAKEST first is the right order: a bond with one turn behind it is a
    # coincidence until a second turn agrees, and a bond that has fired ten
    # times is the store's best evidence about its own vocabulary.
    over = conn.execute("SELECT COUNT(*) FROM bonds").fetchone()[0] - _BOND_CAP
    if over > 0:
        conn.execute(
            "DELETE FROM bonds WHERE rowid IN ("
            "  SELECT rowid FROM bonds ORDER BY weight ASC, last ASC LIMIT ?)",
            (int(over),),
        )


def _drop_bonds(mem_id: Any) -> None:
    """Forget what a fact was found by, when the fact itself is gone or reversed.

    Deliberately NOT gated on `_SHOT_BONDS`. The flag decides whether bonds are
    written and read; it must not decide whether they are cleaned up, or a store
    that runs a while with the knob off comes back to rows about facts it no
    longer holds. (`memories.id` is AUTOINCREMENT, so those rows can never be
    re-pointed at a new fact -- they would simply sit there until the cap
    pruned them. Which is survivable, and not a reason to leave them.)
    """
    try:
        conn = _connect()
        conn.execute("DELETE FROM bonds WHERE mem_id = ?", (int(mem_id),))
        conn.commit()
        conn.close()
    except (sqlite3.Error, TypeError, ValueError):
        # Swallowed because the fact itself is already gone and the caller has
        # nothing left to undo: a stranded bond costs one wrong pellet in a
        # slot the spread was going to leave empty anyway, and raising here
        # would turn a successful `forget` into a failed one.
        pass


def bond_count() -> int:
    """How many learned query-term bonds the store holds. For the CLI/UI."""
    try:
        conn = _connect()
        n = conn.execute("SELECT COUNT(*) FROM bonds").fetchone()[0]
        conn.close()
        return int(n)
    except sqlite3.Error:
        return 0


def _clean_folder(folder: Any) -> str:
    """Normalize a user- or model-supplied folder path (max 3 levels)."""
    parts = [p.strip()[:40] for p in str(folder or "").replace("\\", "/").split("/") if p.strip()]
    return "/".join(parts[:3])


@registry.tool(
    name="remember",
    description=(
        "Save a durable fact, preference or decision to long-term memory so it "
        "survives across sessions. Use when the user states a preference, a "
        "recurring detail about their setup, or a decision worth keeping. "
        "Write it so it can be found again: recall searches these sentences by "
        "the words in them, so name the subject outright and include the terms "
        "a future question would actually use."
    ),
    parameters={
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": (
                    "The fact, as one sentence that stands on its own. Name who "
                    "or what it is about -- 'Dave prefers the dark theme', never "
                    "'he prefers the dark theme'. Next session this sentence is "
                    "all there is; a word that only this conversation supplies "
                    "is a word recall cannot search for."
                ),
            },
            "category": {
                "type": "string",
                "description": "One of: preference, fact, decision, context.",
            },
        },
        "required": ["fact"],
    },
    read_only=False,
)
def remember(fact: str, category: str = "fact", context: str = "") -> str:
    fact = (fact or "").strip()
    if not fact:
        return "ERROR: fact cannot be empty."
    if category not in ("preference", "fact", "decision", "context"):
        category = "fact"

    conn = _connect()
    # Avoid piling up near-duplicates of the same fact.
    existing = conn.execute(
        "SELECT id, fact, created_at, lineage, uses, wins, unproven, grafted "
        "FROM memories").fetchall()
    new_tokens = _tokens(fact)
    stamp = datetime.now().isoformat(timespec="seconds")
    # Where this write is coming from, traced stem to branch to tree. Computed
    # once, before the loop, because it describes the WRITER and not any row it
    # is about to be compared against.
    #
    # Recorded whatever the knob says. `provenance = false` is a decision about
    # the RULES below, not a request to stop noticing where a fact came from --
    # and a store that stopped recording it would have nothing to work with on
    # the day the knob went back on, which is the one moment the history would
    # have been worth having.
    new_lineage = _lineage()
    standing = _standing([r[3:] for r in existing]) if _PROVENANCE else {}
    for (row_id, old, old_stamp, old_lineage,
         old_uses, old_wins, _old_held, old_grafted) in existing:
        old_tokens = _tokens(old)
        if not old_tokens or not new_tokens:
            continue
        overlap = len(new_tokens & old_tokens) / max(len(new_tokens | old_tokens), 1)
        # The second arm is the corrective rewrite: a cue-poor fact being said
        # properly clears containment but not the ratio. See `_elaborates`.
        elaborated = _elaborates(new_tokens, old, old_stamp)
        # ...and the third is the same agent restating its own claim in fewer
        # words. Gated on kinship, not on words alone: containment is only
        # evidence of one claim when one hand wrote both, and two hands writing
        # a broad fact and a narrow one are two witnesses, not a duplicate.
        kin = _kin(new_lineage, old_lineage) if _PROVENANCE else ""
        restated = kin in ("stem", "branch") and _restates(new_tokens, old_tokens)
        if not (overlap > 0.75 or elaborated or restated):
            continue
        if restated and not elaborated and overlap <= 0.75:
            # Nothing in the newer sentence that the store does not already
            # hold, so the FULLER text stays and no row is added. This is the
            # one merge path that writes no text at all: replacing the richer
            # sentence with the poorer one would be a duplicate resolved by
            # throwing away the half worth keeping.
            #
            # `created_at` is deliberately left alone too. Refreshing it would
            # re-open `_elaborates`' hour-long window on a row that has been
            # sitting there for weeks, and that window is the only thing
            # standing between a stale vague fact and being overwritten by a
            # sentence about somebody else.
            conn.commit()
            conn.close()
            # Safe to claim the incumbent for this turn's grade only because of
            # the kinship gate above: the row can only be reached here if THIS
            # run wrote it, so it was already planted at its own write and this
            # adds nothing. Without that gate the fold would let a failing turn
            # mark another tree's fact unproven for a sentence it never wrote.
            _plant(row_id)
            return (
                f"Already remembered as #{row_id}, in fuller words: {old}"
                f"\nNOTE: the sentence just saved says the same thing with "
                f"less in it, so the fuller one was kept rather than a second "
                f"row added. Call `remember` again only if there is something "
                f"here the stored sentence does not already say."
            )
        # A high overlap means the same WORDS, which is not the same as the
        # same CLAIM. Insert one "not" into a long sentence and it reverses
        # while scoring above 0.9 -- so this path, which exists to let a
        # re-wording keep the rank it earned, was also the path by which the
        # exact opposite of a fact inherited it. A statement proven across
        # twenty successful turns handed its whole tally to its own negation,
        # which then arrived at the top of the next recall carrying the
        # maximum boost the system can grant, backed by evidence gathered
        # about the thing it contradicts.
        #
        # The user's newer wording still wins the slot -- a correction is
        # meant to correct, and leaving both would only make recall serve the
        # model a contradiction to referee. What it does not inherit is the
        # proof. Reversed polarity resets the tally, so the new claim starts
        # where every new claim starts and earns its own rank back if it is
        # right. A re-wording that keeps its polarity keeps its tally, which
        # is what this branch was always for.
        reversed_claim = _polarity(fact) != _polarity(old)
        # And the second way a merge can hand over evidence it did not earn.
        # Reversal is about the CLAIM changing; this is about the TREE changing
        # under a claim that did not. When the incoming fact traces back to a
        # different tree than the row it is taking over, the words still win the
        # slot -- but they may only inherit the row's proof if the tree they
        # grew on is standing at least as well as the one that grew the fact
        # being replaced. A corrupt tree cannot bring forth good fruit, and it
        # certainly cannot inherit another tree's.
        #
        # Measured, never assumed: `_standing` counts what has come of each
        # tree's facts and returns nothing at all for a tree that has borne no
        # finished fruit, so an unproven tree is never demoted for being new.
        # Both sides have to be known before this can fire.
        new_tree, old_tree = _tree_of(new_lineage), _tree_of(old_lineage)
        demoted = False
        if _PROVENANCE and kin == "grove" and new_tree in standing and old_tree in standing:
            demoted = standing[new_tree] + _TREE_MARGIN < standing[old_tree]
        sets = "fact = ?, category = ?, created_at = ?"
        args: list[Any] = [fact, category, stamp]
        if context:
            sets += ", context = ?"
            args.append(context)
        if reversed_claim or demoted:
            sets += ", uses = 0, wins = 0, last_used = ''"
        # The row now bears the newer sentence, so it hangs on the tree that
        # grew it. Tracking the text is the whole point -- a lineage left
        # pointing at the replaced claim's origin would make the very next
        # collision be judged against a tree this fact never touched.
        sets += ", lineage = ?"
        args.append(new_lineage)
        if kin == "grove" and old_tree != _TREE_IMPORT:
            # ...and the tree it is being taken FROM leaves its record on the
            # row, because otherwise the takeover erases the evidence about the
            # tree that lost it -- and a worse-standing tree could dismantle
            # the whole rule simply by overwriting enough. Written on every
            # cross-tree takeover, not only a demotion: a tree does not get to
            # shed a poor record by having its fruit taken any more than a good
            # one deserves to lose a proven one. `_graft_record` writes nothing
            # when there was no finished fruit to remember.
            sets += ", grafted = ?"
            args.append(_graft_record(old_grafted, old_tree, old_uses, old_wins))
        args.append(row_id)
        conn.execute(f"UPDATE memories SET {sets} WHERE id = ?", tuple(args))
        conn.commit()
        conn.close()
        palace.mirror_fact(row_id, category, fact, stamp)
        _corpus_changed()
        if reversed_claim or demoted:
            # The tally resets above and the bonds have to go with it, for
            # the same reason: they are evidence gathered about the claim
            # this sentence now contradicts. Leaving them would hand the
            # reversal a set of question-words proven against its opposite.
            # A demotion owes the same debt -- the bonds were earned by the
            # replaced fact's turns, on a tree this one does not grow on.
            _drop_bonds(row_id)
        _plant(row_id)
        note = " (reversed, so its earned rank starts over)" if reversed_claim else ""
        if demoted and not reversed_claim:
            # Named on the result for the same reason the reversal is: the
            # model has just replaced a fact it did not write, and an
            # indistinguishable "Updated existing memory #7" would leave it
            # believing the rank came with it.
            note = (f" (it grew on a different tree than the {old_tree} fact it "
                    f"replaced, so its earned rank starts over)")
        if elaborated and not reversed_claim and not demoted:
            # Say that the fix landed. The model has just been told the
            # fuller sentence would replace the vague one, and an
            # indistinguishable "Updated existing memory #7" is no evidence
            # either way -- leaving it to guess whether the store now holds
            # one good fact or two, one of which it cannot see.
            note = " (replacing a wording recall could not have found)"
        return f"Updated existing memory #{row_id}{note}: {fact}{_cue_note(fact)}"

    cur = conn.execute(
        "INSERT INTO memories (category, fact, context, created_at, folder, lineage) "
        "VALUES (?,?,?,?,?,?)",
        (category, fact, context, stamp, auto_folder(fact, category), new_lineage),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    palace.mirror_fact(new_id, category, fact, stamp)
    _corpus_changed()
    _plant(new_id)
    return f"Remembered #{new_id} [{category}]: {fact}{_cue_note(fact)}"


def _plant(mem_id: Any) -> None:
    """Note that this turn wrote memory #id, for `reinforce` to grade at the end.

    Every write counts as a seed, an update as much as an insert: the text the
    model just put in the store is the claim the turn's outcome is evidence
    about, whichever row it landed in.
    """
    if not _HOLD_UNPROVEN or not _turn_is_live():
        return
    with _planted_lock:
        _planted.setdefault(_run_key(), set()).add(int(mem_id))


def _familiarity(uses: Any, wins: Any, last_used: Any) -> float:
    """The rank boost a fact has earned by serving well before, in [0, 0.5).

    A Laplace-shrunk win share (wins / (uses + 2)), weighted by
    _REINFORCE_WEIGHT and halved for every _REINFORCE_HALFLIFE_DAYS since the
    fact last fed a finished turn. wins <= uses < uses + 2 keeps the share
    strictly under 1, so the result stays strictly under the weight and the
    under-one-token invariant documented at the constants holds by
    construction rather than by tuning.
    """
    uses, wins = int(uses or 0), int(wins or 0)
    if wins <= 0:
        return 0.0
    try:
        age = datetime.now() - datetime.fromisoformat(str(last_used))
    except (TypeError, ValueError):
        return 0.0  # never reinforced (or an unreadable stamp): no history, no boost
    days = max(0.0, age.total_seconds() / 86400.0)
    share = wins / (uses + 2)
    return _REINFORCE_WEIGHT * share * (0.5 ** (days / _REINFORCE_HALFLIFE_DAYS))


@registry.tool(
    name="recall",
    description=(
        "Search long-term memory for facts about a topic. Use when the user "
        "refers to something from a previous session."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to search for."},
            "limit": {"type": "integer", "description": "Max results. Default 8."},
        },
        "required": ["query"],
    },
)
def recall(query: str, limit: int = 0) -> str:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, category, fact, created_at, folder, uses, wins, last_used "
        "FROM memories ORDER BY id DESC"
    ).fetchall()
    conn.close()
    if not rows:
        return "Memory is empty."

    limit = max(1, int(limit)) if limit else _RECALL_LIMIT
    fetch = limit * 3  # over-fetch candidates; fusion trims back down

    # Route to a sector, and let that ROUTING BOOST the facts filed under it --
    # it does not decide which facts are eligible.
    #
    # It used to. `scoped = [r for r in rows if sector_of(r[4]) in routed]` kept
    # only rows inside the (at most two) routed themes, with a whole-store retry
    # if that scored nothing. Two kinds of fact fell through that hole and the
    # retry could not catch either, because it fired only on a TOTAL miss: a
    # sector-less core fact (`sector_of` returns '' and '' is never in `routed`)
    # and anything the user hand-filed outside the theme list from the Memories
    # panel. One weak in-sector match was enough to suppress a far better fact
    # elsewhere -- ask "how much VRAM does my GPU have" with the GPU fact moved
    # to "Hardware/GPU" and recall answered with a monitor's refresh rate.
    #
    # A bonus gets the intended benefit with none of that. `rows` is already
    # fully materialised above, so scoring every row costs no extra database
    # work -- the old two-pass version was doing MORE work on the miss path, not
    # less. And the bonus is deliberately smaller than one token of overlap, so
    # it breaks ties toward the routed subject without ever outranking a fact
    # that simply matches the question better.
    routed = route_sectors(query) if _ROUTING else []

    def _keyword_rank(pool: list[tuple]) -> list[int]:
        # Empty query degrades to recency.
        q = _tokens(query)
        if not q:
            return [r[0] for r in pool[:fetch]]
        scored = []
        for row in pool:
            overlap = len(q & _tokens(row[2]))
            if overlap:
                bonus = _SECTOR_BONUS if (routed and sector_of(row[4]) in routed) else 0.0
                # Familiarity: a fact that has fed successful turns before
                # outranks an equal match that never has. Strictly under one
                # token of overlap even summed with the sector bonus -- see
                # the invariant at the constants.
                if _REINFORCE:
                    bonus += _familiarity(row[5], row[6], row[7])
                # Slight recency tiebreak so newer facts win equal matches.
                scored.append((overlap + bonus + row[0] / 1e6, row[0]))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [rid for _, rid in scored[:fetch]]

    kw_ranked = _keyword_rank(rows)

    # Semantic candidates, ranked. Keys are SQLite row ids where the hit maps
    # back to a stored fact (by mirror id or exact text), else the snippet
    # itself. Any failure degrades to keyword-only, as always.
    #
    # Deliberately NOT sector-scoped, unlike the keyword pass above. Semantic
    # search is already relevance-ranked, so it does not have the problem that
    # scoping solves -- and letting it cross sectors is what makes recall
    # associative: the routed sector answers the question asked, and the
    # semantic pass can still surface the related fact filed somewhere else.
    by_id = {r[0]: r for r in rows}
    by_fact = {r[2].strip().lower(): r[0] for r in rows}
    sem_ranked: list[Any] = []
    sem_text: dict[str, str] = {}
    if _FUSION != "keyword_only":
        try:
            for mem_id, snippet in palace.search(query, limit=fetch):
                snippet = snippet.strip()
                if len(snippet) <= 3:
                    continue
                key: Any = mem_id if mem_id in by_id else by_fact.get(snippet.lower())
                if key is None:
                    key = "txt:" + snippet.lower()
                    sem_text[key] = snippet
                if key not in sem_ranked:
                    sem_ranked.append(key)
        except Exception:  # noqa: BLE001 - recall must never fail on the optional path
            sem_ranked = []

    # Reciprocal rank fusion: both retrievers compete for the same `limit`
    # slots, so a strong semantic hit can displace a weak keyword hit.
    scores: dict[Any, float] = {}
    for ranked in (kw_ranked, sem_ranked):
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    lines = []
    served: list[int] = []
    for key, score in fused[:limit]:
        if score < _MIN_SCORE:
            break
        if key in by_id:
            r = by_id[key]
            served.append(r[0])
            lines.append(f"#{r[0]} [{r[1]}] {r[2]}  ({r[3][:10]})")
        else:
            lines.append(sem_text[key])

    def _serve(rid: int) -> None:
        r = by_id[rid]
        served.append(rid)
        lines.append(f"#{r[0]} [{r[1]}] {r[2]}  ({r[3][:10]})")

    # THE SPREAD, into whatever slots came back empty -- and only those. The
    # fused list above is finished and untouched at this point, so scattershot
    # cannot reorder it, cannot evict from it, and cannot push the answer past
    # the `limit` that already bound it. See THE CHOKE at `_SCATTER`.
    if _SCATTER and len(lines) < limit:
        for rid in _scatter_rank(query, rows, routed, set(served)):
            _serve(rid)
            if len(lines) >= limit:
                break

    # Last resort, and demoted to exactly that. The query named a subject and
    # NOTHING -- no shared word, no semantic hit, no pellet -- reached a fact,
    # which is the case the system prompt manufactures: it prints "more is
    # filed by subject: Personal, Development" and tells the model to recall
    # them, and "what do you know about my family" shares no token with "My
    # wife's name is Sarah". Answering the subject by recency beats making the
    # prompt's own instruction a dead end.
    #
    # It used to run as part of the RANKING, before fusion, on the weaker test
    # of an empty keyword list alone -- so a query with a routed subject and no
    # literal match was answered by whatever had been saved most recently in
    # that theme, in a real hit's format, even when the spread could have found
    # the fact actually asked for. Recency is the least evidence in the system
    # and it now sorts where that belongs: after everything that has any.
    if not lines and routed:
        for r in rows:
            if sector_of(r[4]) in routed:
                _serve(r[0])
                if len(lines) >= limit:
                    break

    if not lines:
        return f"No memories matching {query!r}. Stored facts: {len(rows)}."
    # What actually reached the model is what the turn's outcome is evidence
    # about. Marked here, credited by `reinforce` when the turn ends.
    if _REINFORCE and served:
        with _served_lock:
            _served.setdefault(_run_key(), set()).update(served)
    # ...and so is the question that reached them. A word that found this fact
    # in a turn that then works is the one pellet the spread cannot derive from
    # spelling -- see `_bond_pairs`.
    _bond_pairs(query, served, by_id)
    return "\n".join(lines)


def discard_turn(key: int | None = None) -> None:
    """Drop this run's marks without grading them.

    For a run with learning off -- a mission leg, a sub-agent -- which still
    calls recall and so still marks what it was served. Those marks have no
    turn-end hook to drain them, and keyed per run they would otherwise
    accumulate for the life of the process.
    """
    run = _run_key() if key is None else key
    with _served_lock:
        _served.pop(run, None)
    with _planted_lock:
        _planted.pop(run, None)
    with _bonded_lock:
        _bonded.pop(run, None)


def reinforce(outcome: str = "ok", key: int | None = None) -> list[int]:
    """Credit the facts recall served this turn with how the turn ended.

    The dream-journal observation, applied: a memory revisited right after the
    experience is the one that keeps its strength. Called from the same
    turn-end hook that reinforces learned shortcuts (`loop._remember_run`),
    this drains the served set and bumps `uses` -- plus `wins` when the turn
    succeeded -- on every fact recall put in front of the model this turn.
    `_familiarity` reads the tally back at the next recall.

    Deliberately narrow about what counts: only an explicit recall that fed a
    finished turn. `list_memories` is browsing, not remembering, and the
    prompt digest rides every turn -- crediting either would swamp the signal
    with uniform noise. And a stopped turn credits nothing, for the same
    reason brain.reinforce skips it: a run the user cut short is not evidence
    about the facts.

    It grades the turn's WRITES too -- see `_HOLD_UNPROVEN`. A fact the turn
    saved is held out of the prompt digest if the turn then failed, and
    released as soon as one succeeds. The return value stays what it always
    was, the served ids, because that is what the tests and the CLI read.
    """
    # The caller passes its own run identity, because the loop clears the
    # thread-local stop event before it gets here -- deliberately, so a later
    # dispatch cannot inherit a stale one. Deriving the key at this point would
    # therefore look at a different run than the one that did the recalling.
    run = _run_key() if key is None else key
    with _served_lock:
        ids = sorted(_served.pop(run, set()))
    # Drained unconditionally, and before any early return: an id left in the
    # set would otherwise be graded by whatever turn happens to end next.
    with _planted_lock:
        planted = sorted(_planted.pop(run, set()))
    with _bonded_lock:
        bonds = sorted(_bonded.pop(run, set()))
    # The run is over, so its branch id has nothing left to name. Dropped here
    # rather than in the loop for the same reason the sets above are drained
    # unconditionally: an entry left behind outlives the only thing it meant.
    _drop_branch(run)
    # Nothing recalled and nothing written is the common case for a turn that
    # never touched memory, and it must not cost a connection -- same reason
    # the schema work is latched.
    if not _REINFORCE or (not ids and not planted and not bonds) or outcome == "stopped":
        return []
    win = 1 if outcome == "ok" else 0
    stamp = datetime.now().isoformat(timespec="seconds")
    try:
        conn = _connect()
        for rid in ids:
            conn.execute(
                "UPDATE memories SET uses = uses + 1, wins = wins + ?, last_used = ? "
                "WHERE id = ?",
                (win, stamp, rid),
            )
        # A bond is written only by a turn that WORKED, and never unwritten by
        # one that did not. The asymmetry is the same one `_HOLD_UNPROVEN` uses
        # a few lines down for what recall merely served: a question can be in
        # front of the model through a turn that fails for reasons that have
        # nothing to do with the words in it. A bond that stops earning is
        # already handled -- it stops gaining weight, and the weakest are what
        # `_write_bonds` prunes first.
        if _SHOT_BONDS and win and bonds:
            _write_bonds(conn, bonds, stamp)
        if _HOLD_UNPROVEN:
            if win:
                # A fact that fed a turn that WORKED has done the only thing
                # that can prove it, so anything held back is released here --
                # whether it was served (recall put it to work) or written
                # (this turn's own claim, and the turn stands behind it).
                for rid in set(ids) | set(planted):
                    conn.execute(
                        "UPDATE memories SET unproven = 0 WHERE id = ?", (rid,))
            else:
                # The turn failed. What it WROTE is held back -- see the note
                # at _HOLD_UNPROVEN. What recall merely SERVED is not: a fact
                # can be in front of the model through a turn that fails for
                # reasons that have nothing to do with it, and the shrinking
                # win share above is the right, gentle answer to that. Only the
                # claims this turn actually made are the ones it is evidence
                # about.
                for rid in planted:
                    conn.execute(
                        "UPDATE memories SET unproven = 1 WHERE id = ?", (rid,))
        conn.commit()
        conn.close()
    except sqlite3.Error:
        return []
    return ids


@registry.tool(
    name="list_memories",
    description="List everything currently in long-term memory, newest first.",
    parameters={
        "type": "object",
        "properties": {"limit": {"type": "integer", "description": "Default 30."}},
    },
)
def list_memories(limit: int = 30) -> str:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, category, fact, created_at FROM memories ORDER BY id DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()
    if not rows:
        return "Memory is empty."
    body = "\n".join(f"#{r[0]} [{r[1]}] {r[2]}  ({r[3][:10]})" for r in rows)
    return f"{total} memories total:\n{body}"


@registry.tool(
    name="forget",
    description="Delete a memory by its ID number.",
    parameters={
        "type": "object",
        "properties": {"id": {"type": "integer", "description": "ID from recall or list_memories."}},
        "required": ["id"],
    },
    read_only=False,
    requires_confirm=True,
)
def forget(id: int) -> str:
    conn = _connect()
    row = conn.execute("SELECT fact FROM memories WHERE id = ?", (int(id),)).fetchone()
    if row is None:
        conn.close()
        return f"ERROR: no memory with id {id}."
    conn.execute("DELETE FROM memories WHERE id = ?", (int(id),))
    conn.commit()
    conn.close()
    palace.remove_fact(int(id))
    _drop_bonds(int(id))
    _corpus_changed()
    return f"Forgot #{id}: {row[0]}"


def preload(limit: int | None = None) -> str:
    """The CORE brain: sector-agnostic facts, injected into the system prompt.

    Core means a fact that matched no theme when it was filed, so it carries a
    bare top-level folder. That is not a leftover bucket -- a fact none of the
    sector vocabularies claimed is precisely the one that applies whatever the
    user is doing, which is what earns a permanent seat in the prompt.
    Everything filed under a sector is reached through `recall` instead.

    Two things get fixed by that. The digest used to be the 25 NEWEST facts,
    which is a recency sample, not a relevance one -- ask about a DAW plugin
    and the model was handed last week's git preferences.

    The second is about cache stability, and it is worth stating precisely
    because the obvious version of the claim is wrong. Saving a fact does NOT
    rewrite the prompt in place -- the `remember` tool does not trigger a
    rebuild, and `ui.refresh_prompt` deliberately leaves running chats alone.
    What it changes is what the NEXT rebuild produces, and rebuilds happen at
    every session start and on a settings, route or skills change.

    So under "newest 25", saving a fact in one session meant the next session
    opened with different prompt bytes, and a local runner's KV prefix -- which
    matches from position 0 -- missed on all of it. While the runner is still
    warm, that prefill is pure waste. Core facts change rarely and the
    sector line changes only when a subject gets its first fact, so the bytes
    now stay identical across ordinary use and the warm runner keeps its prefix.

    Worth being equally clear about what this does NOT do: it frees no VRAM.
    A runner that reserves the whole context window's KV cache up front -- as
    the common local ones do -- allocates it regardless of how much a
    conversation uses, so a smaller prompt is a prefill saving, never an
    allocation one. The lever that moves VRAM is the context size itself.
    """
    if limit is not None:
        n = max(0, int(limit))
    else:
        # Routing off is the pre-sector behaviour, digest size included.
        n = _PRELOAD_CORE if _ROUTING else 25
    if n <= 0:
        return ""
    try:
        conn = _connect()
        hold = "AND unproven = 0 " if _HOLD_UNPROVEN else ""
        hold_only = "WHERE unproven = 0 " if _HOLD_UNPROVEN else ""
        if _ROUTING:
            # Bare folder = no sector matched = core. A hand-moved fact with a
            # nested folder is the user filing it deliberately, so it counts as
            # sector material and stays out of the prompt.
            # Held-back facts are skipped, not counted: LIMIT still fills n
            # slots, so a fact on probation costs the digest a line of its own
            # rather than shrinking it.
            #
            # This is the one place the flag is read, and it keeps the byte
            # stability the rest of this docstring is about. `unproven` only
            # ever moves when a turn fails on a write or a held fact is
            # released -- rare, and rarer still in the same direction twice --
            # so the digest is unchanged across ordinary use, exactly as a
            # sector name is. Ordering by anything the tally moves (wins, say)
            # would churn the prompt every turn, which is the mistake this
            # comment exists to stop the next reader making.
            rows = conn.execute(
                "SELECT category, fact FROM memories WHERE folder NOT LIKE '%/%' "
                f"{hold}ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
        else:
            # The hold applies here too. Routing off restores the pre-sector
            # DIGEST -- the newest 25 of everything -- but it says nothing about
            # whether an unproven claim has earned a permanent seat in the
            # prompt, and those two knobs are independent. Leaving it out here
            # would mean turning routing off quietly re-opened the hole: a fact
            # from a failed turn back in front of the model on every turn.
            rows = conn.execute(
                f"SELECT category, fact FROM memories {hold_only}"
                "ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        conn.close()
    except sqlite3.Error:
        return ""
    if not rows:
        return ""
    return "\n".join(f"- [{c}] {f}" for c, f in rows)


def count() -> int:
    try:
        conn = _connect()
        n = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        return int(n)
    except sqlite3.Error:
        return 0


def all_memories(limit: int = 500) -> list[dict[str, Any]]:
    """Structured dump of stored memories for the UI, newest first.

    Unlike the `list_memories` tool (which returns text for the model), this
    returns rows the desktop app can render and manage individually.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, category, fact, context, created_at, folder, "
            "uses, wins, last_used, unproven, lineage "
            "FROM memories ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    return [
        {
            "id": r[0],
            "category": r[1],
            "fact": r[2],
            "context": r[3] or "",
            "created_at": r[4],
            "folder": r[5] or "",
            "uses": r[6] or 0,
            "wins": r[7] or 0,
            "last_used": r[8] or "",
            "unproven": bool(r[9]),
            # Where it came from, traced back. The raw path is carried too so
            # the panel can show the branch a fact grew on, but `tree` is what
            # a reader wants first and it is the one value that survives a
            # database written before lineage existed -- as "import", which is
            # the honest answer and not a guess at a writer.
            "lineage": r[10] or "",
            "tree": _tree_of(r[10]),
            # Empty for a fact that carries its own cues. Computed rather than
            # stored: it is a pure function of the text, so a row edited by
            # hand via `move_memory` is re-judged on the spot instead of
            # carrying a verdict about a sentence it no longer holds.
            "cue_gap": _cue_gap(r[2]),
        }
        for r in rows
    ]


def trust_memory(mem_id: Any) -> bool:
    """Release one held-back fact by hand. True if the row existed.

    The escape hatch the hold needs to be safe. A fact can be held back for a
    reason that has nothing to do with whether it is true -- the turn that
    wrote it went on to fail at something else entirely -- and waiting for a
    recall to happen to pick it up is a poor answer when the user can simply
    look at it and say it is right. A person vouching is better evidence than
    anything this module can gather on its own, so this needs no tally and no
    turn: it clears the flag outright.
    """
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT id FROM memories WHERE id = ?", (int(mem_id),)).fetchone()
        if row is None:
            conn.close()
            return False
        # Grafted onto the user's own tree at the same time, and this is the
        # one place a lineage is rewritten by something other than a write.
        # It is not an exception to "measured, never assumed" -- a person
        # vouching is not a claim ABOUT a tree, it is evidence about this
        # fruit, and the module already holds it to be the best evidence it can
        # get. What follows from the graft is that the row's own record then
        # counts toward what the user's tree is standing on, which is the only
        # way `_standing` ever learns anything about any tree.
        conn.execute(
            "UPDATE memories SET unproven = 0, lineage = ? WHERE id = ?",
            (_LINEAGE_USER, int(mem_id)))
        conn.commit()
        conn.close()
        return True
    except (sqlite3.Error, TypeError, ValueError):
        return False


def orchard() -> list[dict[str, Any]]:
    """Every tree in the store, with the fruit it has borne. Best-standing first.

    The reporting half of provenance, and the reason it is worth having on its
    own: the rules in `remember` act on a standing nobody could otherwise see,
    and a number that decides which of two facts keeps its rank has to be
    readable by the person whose store it is. `orchard()` returns
    this.

    `standing` is None for a tree that has borne no finished fruit yet -- the
    same distinction `_standing` draws, carried through to the renderer rather
    than flattened into a zero that would read as a bad tree.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT lineage, uses, wins, unproven, grafted FROM memories").fetchall()
        conn.close()
        graded = _standing(rows)
    except sqlite3.Error:
        return []
    def _blank(tree: str) -> dict[str, Any]:
        return {"tree": tree, "facts": 0, "uses": 0, "wins": 0, "held": 0,
                "taken": 0, "branches": set(), "standing": None}

    trees: dict[str, dict[str, Any]] = {}
    for lineage, uses, wins, held, grafted in rows:
        tree = _tree_of(lineage)
        acc = trees.setdefault(tree, _blank(tree))
        acc["facts"] += 1
        acc["uses"] += int(uses or 0)
        acc["wins"] += int(wins or 0)
        acc["held"] += 1 if held else 0
        parts = str(lineage or "").split("/")
        if len(parts) == 3 and all(parts):
            acc["branches"].add(parts[1])
        # Fruit this row bore for a tree that no longer holds it. Reported as
        # its own count rather than folded into `facts`, because a standing
        # resting on facts a tree has lost is a different claim from one
        # resting on facts it still bears, and the reader is owed the
        # difference -- Law 1, every number carries its provenance.
        for gtree, guses, gwins in _grafts(grafted):
            gacc = trees.setdefault(gtree, _blank(gtree))
            gacc["taken"] += 1
            gacc["uses"] += guses
            gacc["wins"] += gwins
    out = []
    for tree, acc in trees.items():
        acc["branches"] = len(acc["branches"])
        acc["standing"] = graded.get(tree)
        out.append(acc)
    # Ungraded trees sort last rather than as zeroes: "nothing has come of it
    # yet" is not "nothing good has come of it", and a list that ranks them
    # together is the false balance the whole module is written against.
    return sorted(out, key=lambda t: (t["standing"] is None,
                                      -(t["standing"] or 0.0), t["tree"]))


def unproven_count() -> int:
    """How many facts are currently held out of the prompt digest."""
    try:
        conn = _connect()
        n = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE unproven = 1").fetchone()[0]
        conn.close()
        return int(n)
    except sqlite3.Error:
        return 0


def move_memory(mem_id: Any, folder: Any) -> bool:
    """Reassign one memory's folder. True if the row existed."""
    clean = _clean_folder(folder)
    try:
        conn = _connect()
        row = conn.execute("SELECT id FROM memories WHERE id = ?", (int(mem_id),)).fetchone()
        if row is None:
            conn.close()
            return False
        conn.execute("UPDATE memories SET folder = ? WHERE id = ?", (clean, int(mem_id)))
        conn.commit()
        conn.close()
        return True
    except (sqlite3.Error, TypeError, ValueError):
        return False


def folder_list() -> list[str]:
    """Every folder path currently in use, sorted."""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT DISTINCT folder FROM memories WHERE folder <> ''"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    return sorted({r[0] for r in rows})


def backfill_folders() -> int:
    """Assign an auto folder to any memory that still has none.

    Runs once at startup so a database from before folders existed organizes
    itself the first time it is opened; later runs find nothing to do.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, category, fact FROM memories WHERE folder = ''"
        ).fetchall()
        for rid, cat, fact in rows:
            conn.execute(
                "UPDATE memories SET folder = ? WHERE id = ?",
                (auto_folder(fact, cat), rid),
            )
        conn.commit()
        conn.close()
        return len(rows)
    except sqlite3.Error:
        return 0


def delete_memory(mem_id: int) -> bool:
    """Delete one memory and its semantic mirror. True if the row existed.

    The UI's own confirm click is the gate here, so this is a plain function
    rather than the confirm-gated `forget` tool.
    """
    try:
        conn = _connect()
        row = conn.execute("SELECT id FROM memories WHERE id = ?", (int(mem_id),)).fetchone()
        if row is None:
            conn.close()
            return False
        conn.execute("DELETE FROM memories WHERE id = ?", (int(mem_id),))
        conn.commit()
        conn.close()
    except (sqlite3.Error, TypeError, ValueError):
        return False
    palace.remove_fact(int(mem_id))
    _drop_bonds(int(mem_id))
    _corpus_changed()
    return True


def backfill_mirror() -> int:
    """Mirror any SQLite facts that have no file yet.

    Covers the case where a semantic backend is registered after facts already
    exist.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, category, fact, created_at FROM memories"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return 0
    for row in rows:
        palace.mirror_fact(row[0], row[1], row[2], row[3])
    return len(rows)
