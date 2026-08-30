"""The self-learning layer -- the "second brain".

`memory.py` stores *facts* the user states. This module stores what the agent
*does*: the tasks it carries out, the specifics that made them work, and how
often they recur. Over time that turns a repeated request into a one-liner --
the point behind "if I say open the chat app on the desktop and message
so-and-so, next time I should only have to say open the chat app."

Three tables, all in the same `memory.db` as facts so the two brains can be
linked (an association can point a learned task at a related fact):

  * **patterns**  -- learned task shortcuts: a trigger phrase, the resolution
                     that worked (which app, which surface, which steps), and a
                     use/win tally that becomes a confidence score. Reinforced
                     automatically when a matching request succeeds, and pinned
                     when the user teaches it by hand.
  * **episodes**  -- a compact log of past turns (request, outcome, tools), so
                     the assistant can look back at how a similar task went.
  * **associations** -- weighted links between patterns that fire together, the
                     "synapses" that let recall pull in a related shortcut.

Design constraints match the rest of the kit: SQLite + stdlib only, no embedding
model competing for VRAM, every path degrades to a no-op if learning is off or
the database is unavailable. The learned shortcuts are folded into the system
prompt at session start, which is where the token saving comes from -- the model
is handed the answer instead of re-deriving it every time.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .memory import _tokens  # same stop-word-filtered tokenizer as facts use
from .memory import route_sectors
from . import dbsafe
from .registry import registry

_DB: Path | None = None
_ENABLED = True
_PRELOAD = 8  # learned shortcuts folded into the system prompt each session
_ROUTING = True  # bias pattern matching toward the request's sector

# What being in the request's sector adds to a pattern's score. Under the 2.0
# a user-taught shortcut carries and under a full point of confidence, so it
# orders comparable candidates without ever overriding either.
_SECTOR_BONUS = 0.25

# Schema setup runs once per process, not once per call. Every helper here
# opens its own connection, and re-issuing eight CREATE/ALTER statements on
# each of them was doing real work on a file the sidecar and the UI also hold
# open -- the intermittent lock stalls that read as "the brain being weird".
_SCHEMA_READY = False
# Guards the migration below. invalidate_schema() resets the latch while the
# desktop's worker threads are live (a backup restore does exactly that), so
# two first-connects can race: both read the same PRAGMA table_info, both
# decide a column is missing, and the loser's ALTER fails with 'duplicate
# column name' -- landing precisely in the just-restored window this
# mechanism exists to make safe.
_SCHEMA_LOCK = threading.Lock()

# Coverage a request must have of a pattern's trigger tokens to count as a hit.
# 0.6 lets "just open the chat app please" still match "open the chat app"
# while keeping unrelated requests from reinforcing the wrong shortcut.
_MATCH_COVERAGE = 0.6
_EPISODE_KEEP = 500  # cap the episode log so it never grows without bound

# Episode consolidation -- the sleep pass. Episodes are written at every turn
# end and, until this, only ever read back one at a time by `similar_episode`:
# a task the user repeats in different words piled up as near-identical rows
# and aged off the 500-row cap without ever becoming a shortcut, unless the
# model happened to call `learn_task` mid-task. `consolidate` closes that gap
# the way sleep does for the day's experience: every so many new episodes it
# clusters the successful ones by token overlap and promotes a cluster seen on
# enough distinct days into a pattern, seeded with the cluster's own tally so
# its confidence starts at the evidence that earned it. Deterministic and
# stdlib-only like everything else here -- no model call, so it runs at turn
# end unnoticed.
_CONSOLIDATE_EVERY = 20  # sweep cadence, in new episodes since the last sweep
_CONSOLIDATE_SIM = 0.5   # Jaccard overlap that clusters two requests
_CONSOLIDATE_MIN = 3     # successful episodes before a cluster can promote
_CONSOLIDATE_DAYS = 2    # distinct days the cluster must span -- one busy
                         # afternoon of retries is one task, not a habit
_CONSOLIDATE_CAP = 3     # promotions per sweep, so a backlog trickles into the
                         # preload digest instead of flooding it at once


def configure(state_dir: Path, cfg: dict[str, Any] | None = None) -> None:
    """Point the brain at memory.db and read learning settings.

    Shares the facts database on purpose, so associations can one day bridge a
    learned task and a stored fact without a cross-file join.
    """
    global _DB, _ENABLED, _PRELOAD, _ROUTING, _SCHEMA_READY
    cfg = cfg or {}
    _DB = Path(state_dir) / "memory.db"
    _ENABLED = bool(cfg.get("learning", True))
    _PRELOAD = max(0, int(cfg.get("preload_patterns", 8)))
    _ROUTING = bool(cfg.get("sector_routing", True))
    # A new state dir is a different database, so the once-per-process schema
    # setup has to run again against it.
    _SCHEMA_READY = False
    dbsafe.open_or_quarantine(_DB, lambda: _connect().close())


def enabled() -> bool:
    return _ENABLED and _DB is not None


def invalidate_schema() -> None:
    """Force the next `_connect` to re-run the migration.

    Same contract as `memory.invalidate_schema`, and the same trigger: a backup
    restore replaces memory.db in place, and one taken before the `sector`
    column existed left `learn` raising `no such column` until restart.
    """
    global _SCHEMA_READY
    _SCHEMA_READY = False


def _connect() -> sqlite3.Connection:
    global _SCHEMA_READY
    if _DB is None:
        raise RuntimeError("Brain not initialized; call configure() first.")
    conn = sqlite3.connect(_DB)
    # Once per process. Anything that swaps memory.db underneath a running
    # process must call `invalidate_schema` -- see that function.
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
            CREATE TABLE IF NOT EXISTS patterns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger     TEXT NOT NULL,
                trigger_key TEXT NOT NULL,
                resolution  TEXT NOT NULL,
                uses        INTEGER NOT NULL DEFAULT 1,
                wins        INTEGER NOT NULL DEFAULT 0,
                taught      INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL,
                last_used   TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      TEXT NOT NULL,
                request TEXT NOT NULL,
                summary TEXT,
                tools   TEXT,
                outcome TEXT NOT NULL DEFAULT 'ok'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS associations (
                a_id   INTEGER NOT NULL,
                b_id   INTEGER NOT NULL,
                weight INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (a_id, b_id)
            )
            """
        )
        # Which sector a shortcut belongs to, so matching can rank the brain the
        # request concerns above the rest. Migrated in place, the same way
        # memory.py added its folder column.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(patterns)")}
        if "sector" not in cols:
            conn.execute("ALTER TABLE patterns ADD COLUMN sector TEXT NOT NULL DEFAULT ''")
        # `learn` looks a pattern up by trigger_key on every call; without this it
        # was a full scan of the table. There is deliberately no index on `sector`:
        # matching reads the whole table and scores it in Python, so an index there
        # would be write cost paid for a query nothing issues.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_key ON patterns(trigger_key)")
        conn.execute("DROP INDEX IF EXISTS idx_patterns_sector")
        # Consolidation bookkeeping: the sweep's high-water mark, and a ledger of
        # trigger keys it already promoted. The ledger row outlives the pattern on
        # purpose -- deleting a mined shortcut has to mean "I don't want this",
        # not "mine it again next sweep".
        conn.execute(
            "CREATE TABLE IF NOT EXISTS brain_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS consolidated ("
            "trigger_key TEXT PRIMARY KEY, ts TEXT NOT NULL)"
        )
        # Readers no longer block on the writer. memory.db is opened concurrently
        # by the CLI, the sidecar and the desktop UI, and the default rollback
        # journal serialises all of them.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass  # a filesystem that cannot do WAL still works, just serialised
        conn.commit()
        _SCHEMA_READY = True
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_trigger(text: str) -> tuple[str, str]:
    """A trigger's display form and its order-independent match key.

    The key is the sorted set of significant tokens, so "open the chat app",
    "the chat app, open it" and "open up the chat app" all collapse to one
    shortcut.
    """
    display = " ".join((text or "").split()).strip(" .!?,;:")
    key = " ".join(sorted(_tokens(display)))
    return display[:120], key


def _confidence(uses: int, wins: int, taught: int) -> float:
    """Laplace-smoothed win rate, floored high for anything the user taught."""
    base = (wins + 1) / (uses + 2)
    return max(base, 0.9) if taught else base


# ------------------------------------------------------------------ learning

def learn(trigger: str, resolution: str, taught: bool = False) -> dict[str, Any]:
    """Record or refine a task shortcut.

    A new trigger inserts a pattern; a trigger already known updates its
    resolution (the model got a better recipe, or the user corrected it) and,
    when taught, pins it so automatic reinforcement can't erode it.
    """
    if not enabled():
        return {}
    display, key = normalize_trigger(trigger)
    resolution = (resolution or "").strip()
    if not key or not resolution:
        return {}
    stamp = _now()
    # Route on the trigger AND the resolution: a trigger is a handful of words
    # ("open the chat app") and often carries no sector vocabulary at all, while
    # the resolution names the app, the surface and the steps.
    sector = ""
    if _ROUTING:
        hits = route_sectors(f"{display} {resolution}", limit=1)
        sector = hits[0] if hits else ""
    conn = _connect()
    row = conn.execute(
        "SELECT id, taught FROM patterns WHERE trigger_key = ?", (key,)
    ).fetchone()
    if row is not None:
        pid, was_taught = int(row[0]), int(row[1])
        conn.execute(
            "UPDATE patterns SET trigger = ?, resolution = ?, taught = ?, "
            "sector = ?, last_used = ? WHERE id = ?",
            (display, resolution, 1 if (taught or was_taught) else 0, sector,
             stamp, pid),
        )
        action = "updated"
    else:
        cur = conn.execute(
            "INSERT INTO patterns (trigger, trigger_key, resolution, uses, wins, "
            "taught, created_at, last_used, sector) VALUES (?,?,?,?,?,?,?,?,?)",
            (display, key, resolution, 1, 1 if taught else 0, 1 if taught else 0,
             stamp, stamp, sector),
        )
        pid, action = int(cur.lastrowid), "learned"
    conn.commit()
    conn.close()
    return {"id": pid, "trigger": display, "resolution": resolution,
            "taught": bool(taught), "action": action}


def teach(trigger: str, resolution: str) -> dict[str, Any]:
    """Manual teaching: same as learn() but pinned as user-authoritative."""
    return learn(trigger, resolution, taught=True)


def _rows() -> list[tuple]:
    """Every stored shortcut, with the sector it was filed under.

    Deliberately unnarrowed. This used to take a `sectors` argument and push a
    `WHERE sector = '' OR sector IN (...)` filter into SQL, on the reasoning
    that an indexed lookup beats a full table read. The reasoning was sound and
    the arithmetic was backwards: `_match` retried unscoped whenever the scoped
    pass found nothing, so the miss path ran TWO queries plus two scoring passes
    where this runs one. The patterns table is small -- one row per shortcut a
    user has ever taught or the agent has ever learned -- and reading it whole
    is cheaper than the retry it replaces.
    """
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, trigger, trigger_key, resolution, uses, wins, taught, "
            "last_used, sector FROM patterns"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    return rows


def _match(request: str) -> list[tuple[dict[str, Any], float]]:
    """Patterns whose trigger the request substantially covers, best first.

    The request's sectors BOOST the shortcuts filed under them; they no longer
    decide which shortcuts are eligible. The old version scored only the routed
    sectors and retried unscoped `if that found nothing` -- but "nothing" meant
    zero hits in total, so one unrelated match was enough to suppress a
    shortcut the user had explicitly taught.

    That is not an edge case, because the two sides are computed from different
    text: a pattern's sector comes from its trigger AND resolution (see `learn`),
    while the route comes from the request alone. Teach "open the chat app" (filed
    under Apps & tools, from its resolution) and "check the gpu" (System & PC),
    then ask to do both: the request routes to System & PC only, and the
    shortcut vanishes -- from `predict`, and from the `reinforce` that
    would have kept its usage counters alive.
    """
    rq = _tokens(request)
    if not rq:
        return []
    routed = route_sectors(request) if _ROUTING else []
    return _score(rq, _rows(), routed)


def _score(rq: set[str], rows: list[tuple],
           routed: list[str] | None = None) -> list[tuple[dict[str, Any], float]]:
    routed = routed or []
    hits: list[tuple[dict[str, Any], float]] = []
    for rid, trig, key, res, uses, wins, taught, last, sector in rows:
        trig_tokens = set(key.split())
        if not trig_tokens:
            continue
        covered = len(trig_tokens & rq) / len(trig_tokens)
        if covered < _MATCH_COVERAGE:
            continue
        conf = _confidence(uses, wins, taught)
        pat = {"id": rid, "trigger": trig, "resolution": res, "uses": uses,
               "wins": wins, "taught": bool(taught), "confidence": round(conf, 2),
               "last_used": last}
        # Rank: user-taught first, then confident+well-covered, then well-worn.
        score = (2.0 if taught else 0.0) + conf * covered + min(uses, 20) / 100.0
        # Sized to sit under the taught bonus (2.0) and under a full point of
        # confidence, so the sector settles ties between comparable shortcuts
        # and never promotes a guess over one the user taught by hand.
        if routed and sector and sector in routed:
            score += _SECTOR_BONUS
        hits.append((pat, score))
    hits.sort(key=lambda t: t[1], reverse=True)
    return hits


def _associated(ids: list[int], limit: int = 3) -> list[dict[str, Any]]:
    """Patterns linked to the given ones -- the synapse hop for recall."""
    if not ids:
        return []
    try:
        conn = _connect()
        marks = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT p.id, p.trigger, p.resolution, SUM(a.weight) AS w
            FROM associations a
            JOIN patterns p
              ON p.id = CASE WHEN a.a_id IN ({marks}) THEN a.b_id ELSE a.a_id END
            WHERE (a.a_id IN ({marks}) OR a.b_id IN ({marks}))
              AND p.id NOT IN ({marks})
            GROUP BY p.id ORDER BY w DESC LIMIT ?
            """,
            (*ids, *ids, *ids, *ids, int(limit)),
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    return [{"id": r[0], "trigger": r[1], "resolution": r[2], "weight": r[3]}
            for r in rows]


def predict(request: str, limit: int = 3) -> list[dict[str, Any]]:
    """Best learned shortcuts for a request, plus any associated ones."""
    if not enabled():
        return []
    hits = [pat for pat, _ in _match(request)][:limit]
    seen = {p["id"] for p in hits}
    for rel in _associated(list(seen)):
        if rel["id"] not in seen:
            rel["related"] = True
            hits.append(rel)
            seen.add(rel["id"])
    return hits


def similar_episode(request: str) -> dict[str, Any] | None:
    """The most similar past turn that succeeded, for 'how did this go before'."""
    rq = _tokens(request)
    if not rq or not enabled():
        return None
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT request, summary, tools, ts FROM episodes "
            "WHERE outcome = 'ok' ORDER BY id DESC LIMIT 300"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return None
    best, best_score = None, 0
    for req, summary, tools, ts in rows:
        score = len(rq & _tokens(req))
        if score > best_score:
            best, best_score = {"request": req, "summary": summary or "",
                                "tools": tools or "", "ts": ts}, score
    return best if best_score >= 2 else None


# --------------------------------------------------------- reinforcement

def record_episode(request: str, summary: str, tools: list[str] | str,
                   outcome: str = "ok") -> None:
    """Append one turn to the episode log and trim it to the cap."""
    if not enabled():
        return
    tool_str = ", ".join(tools) if isinstance(tools, (list, tuple)) else str(tools or "")
    try:
        conn = _connect()
        conn.execute(
            "INSERT INTO episodes (ts, request, summary, tools, outcome) VALUES (?,?,?,?,?)",
            (_now(), (request or "")[:500], (summary or "")[:400], tool_str[:200],
             outcome or "ok"),
        )
        # Keep only the newest _EPISODE_KEEP rows.
        conn.execute(
            "DELETE FROM episodes WHERE id NOT IN "
            "(SELECT id FROM episodes ORDER BY id DESC LIMIT ?)",
            (_EPISODE_KEEP,),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        # Learning is best-effort; a locked database costs this pass, not the turn
        pass


def reinforce(request: str, outcome: str = "ok") -> list[int]:
    """Update the tally of every shortcut the request matched, and link them.

    A success bumps both uses and wins (confidence rises); a miss bumps only
    uses (confidence falls), so shortcuts that keep working float to the top and
    ones that stop working quietly sink. Patterns that fire together in the same
    request get an association weight -- the synapse that recall later follows.
    """
    if not enabled():
        return []
    # A run the user cut short is not evidence about the shortcut. Counting it
    # as a miss lets someone who changes their mind twice bury a pattern that
    # was working perfectly well.
    if outcome == "stopped":
        return []
    ids = [pat["id"] for pat, _ in _match(request)]
    if not ids:
        return []
    win = 1 if outcome == "ok" else 0
    stamp = _now()
    try:
        conn = _connect()
        for pid in ids:
            conn.execute(
                "UPDATE patterns SET uses = uses + 1, wins = wins + ?, last_used = ? "
                "WHERE id = ?",
                (win, stamp, pid),
            )
        # Strengthen the links between every pair that co-fired this turn.
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                lo, hi = sorted((a, b))
                conn.execute(
                    "INSERT INTO associations (a_id, b_id, weight) VALUES (?,?,1) "
                    "ON CONFLICT(a_id, b_id) DO UPDATE SET weight = weight + 1",
                    (lo, hi),
                )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        return []
    return ids


def consolidate(force: bool = False) -> list[dict[str, Any]]:
    """Mine the episode log for recurring tasks and promote them to shortcuts.

    The other half of the reinforcement story. `learn_task` depends on the
    model noticing, mid-task, that a task will recur -- which mostly does not
    happen, so the episode log fills with the same task in different words and
    the knowledge ages off the cap unused. This pass replays the log offline:

      * successful episodes are single-link clustered on token overlap
        (Jaccard >= _CONSOLIDATE_SIM);
      * a cluster of >= _CONSOLIDATE_MIN episodes spanning >= _CONSOLIDATE_DAYS
        distinct days becomes a pattern, unless one already covers its request
        shape -- mining fills holes, it never rewrites a resolution the model
        or the user wrote;
      * the trigger is the cluster's SHORTEST phrasing (people compress their
        own recurring requests, and that compact form is exactly what the
        shortcut should answer to), the resolution is built from the newest
        episode's summary and tools, and uses/wins are seeded with the cluster
        size, so the pattern's confidence is the evidence that earned it.

    Runs from the turn-end hook, gated to every _CONSOLIDATE_EVERY new
    episodes so it stays a rounding error; `force` (the CLI, tests) sweeps
    now. Returns what it promoted.
    """
    if not enabled():
        return []
    try:
        conn = _connect()
        newest = int(conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM episodes").fetchone()[0])
        row = conn.execute(
            "SELECT value FROM brain_meta WHERE key = 'consolidate_hwm'"
        ).fetchone()
        hwm = int(row[0]) if row and str(row[0]).isdigit() else 0
        if not force and newest - hwm < _CONSOLIDATE_EVERY:
            conn.close()
            return []
        rows = conn.execute(
            "SELECT id, ts, request, summary, tools FROM episodes "
            "WHERE outcome = 'ok' ORDER BY id"
        ).fetchall()
        mined = {r[0] for r in conn.execute("SELECT trigger_key FROM consolidated")}
        conn.close()
    except sqlite3.Error:
        return []

    # Single-link clustering via union-find. The log is capped at
    # _EPISODE_KEEP rows, so the pairwise pass is bounded and fast; episodes
    # whose requests share no vocabulary never join.
    toks = [(r, _tokens(r[2])) for r in rows]
    toks = [(r, t) for r, t in toks if len(t) >= 2]
    parent = list(range(len(toks)))

    def _root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            a, b = toks[i][1], toks[j][1]
            if len(a & b) / len(a | b) >= _CONSOLIDATE_SIM:
                ri, rj = _root(i), _root(j)
                if ri != rj:
                    parent[rj] = ri

    clusters: dict[int, list[tuple]] = {}
    for i, (r, _t) in enumerate(toks):
        clusters.setdefault(_root(i), []).append(r)

    promoted: list[dict[str, Any]] = []
    stamp = _now()
    for members in clusters.values():
        if len(promoted) >= _CONSOLIDATE_CAP:
            break
        if len(members) < _CONSOLIDATE_MIN:
            continue
        if len({m[1][:10] for m in members}) < _CONSOLIDATE_DAYS:
            continue
        shortest = min(members, key=lambda m: (len(_tokens(m[2])), len(m[2])))
        display, key = normalize_trigger(shortest[2])
        if len(key.split()) < 2:
            continue  # a one-token trigger would match far too much
        if key in mined:
            continue
        if _match(display):
            # An existing pattern already answers this request shape. Its
            # resolution -- written by the model in the moment, or taught by
            # the user -- beats anything reconstructed from a summary.
            continue
        newest_ep = max(members, key=lambda m: m[0])
        res = f"Recurring task, done {len(members)} times."
        tools = (newest_ep[4] or "").strip()
        if tools:
            res += f" Tools that worked: {tools}."
        summary = (newest_ep[3] or "").strip()
        if summary:
            res += f" Last time: {summary}"
        made = learn(display, res[:300])
        if not made or made.get("action") != "learned":
            continue
        try:
            conn = _connect()
            # Seed the tally with the cluster's own evidence, so the shortcut
            # starts at the confidence those episodes earned rather than at
            # one untested use -- and record the key so a later delete sticks.
            conn.execute(
                "UPDATE patterns SET uses = ?, wins = ? WHERE id = ?",
                (len(members), len(members), made["id"]),
            )
            conn.execute(
                "INSERT OR REPLACE INTO consolidated (trigger_key, ts) VALUES (?, ?)",
                (key, stamp),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error:
            continue
        mined.add(key)
        promoted.append({"id": made["id"], "trigger": display,
                         "resolution": res[:300], "uses": len(members)})

    try:
        conn = _connect()
        conn.execute(
            "INSERT INTO brain_meta (key, value) VALUES ('consolidate_hwm', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(newest),),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        # The sleep pass is best-effort; a locked database costs this pass only
        pass
    return promoted


# -------------------------------------------------------------- surfacing

def preload(limit: int | None = None) -> str:
    """Compact digest of top shortcuts, injected into the system prompt.

    This is where the learning pays for itself: the model reads the resolved
    shortcut instead of re-discovering it, so a recurring task costs a handful
    of prompt tokens rather than a fresh round of exploration.
    """
    if not enabled():
        return ""
    n = _PRELOAD if limit is None else int(limit)
    if n <= 0:
        return ""
    rows = _rows()
    if not rows:
        return ""
    scored = []
    for rid, trig, key, res, uses, wins, taught, last, _sector in rows:
        conf = _confidence(uses, wins, taught)
        scored.append(((2.0 if taught else 0.0) + conf, last, trig, res, taught))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    lines = []
    for _score, _last, trig, res, taught in scored[:n]:
        mark = " (you taught this)" if taught else ""
        lines.append(f'- when the request is like "{trig}": {res}{mark}')
    return "\n".join(lines)


def all_patterns(limit: int = 200) -> list[dict[str, Any]]:
    """Structured shortcuts for the desktop Learned panel, strongest first."""
    out = []
    for rid, trig, key, res, uses, wins, taught, last, _sector in _rows():
        out.append({
            "id": rid, "trigger": trig, "resolution": res, "uses": uses,
            "wins": wins, "taught": bool(taught),
            "confidence": round(_confidence(uses, wins, taught), 2),
            "last_used": last,
        })
    out.sort(key=lambda p: (p["taught"], p["confidence"], p["last_used"]), reverse=True)
    return out[: int(limit)]


def all_episodes(limit: int = 50) -> list[dict[str, Any]]:
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, ts, request, summary, tools, outcome FROM episodes "
            "ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    return [{"id": r[0], "ts": r[1], "request": r[2], "summary": r[3] or "",
             "tools": r[4] or "", "outcome": r[5]} for r in rows]


def forget_pattern(pattern_id: Any) -> bool:
    """Delete one shortcut and any synapses touching it. True if it existed."""
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT id FROM patterns WHERE id = ?", (int(pattern_id),)
        ).fetchone()
        if row is None:
            conn.close()
            return False
        pid = int(pattern_id)
        conn.execute("DELETE FROM patterns WHERE id = ?", (pid,))
        conn.execute("DELETE FROM associations WHERE a_id = ? OR b_id = ?", (pid, pid))
        conn.commit()
        conn.close()
        return True
    except (sqlite3.Error, TypeError, ValueError):
        return False


def pattern_count() -> int:
    try:
        conn = _connect()
        n = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        conn.close()
        return int(n)
    except sqlite3.Error:
        return 0


def episode_count() -> int:
    try:
        conn = _connect()
        n = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        conn.close()
        return int(n)
    except sqlite3.Error:
        return 0


def status() -> str:
    if not enabled():
        return "Learning: off"
    return (f"Learning: on | {pattern_count()} shortcut(s), "
            f"{episode_count()} episode(s)")


# ------------------------------------------------------------------- tools

@registry.tool(
    name="learn_task",
    description=(
        "Record how to do a recurring task so a shorter request works next time. "
        "Call this after you complete a task that involved a specific choice the "
        "user is likely to want again -- which app, which surface (desktop vs "
        "web), which steps. Example: after 'open the chat app on the desktop and "
        "message Sam', save trigger='open the chat app' with resolution='use the "
        "desktop app via the desktop tools, not the web version'. Set taught=true "
        "only when the user is explicitly teaching or correcting you."
    ),
    parameters={
        "type": "object",
        "properties": {
            "trigger": {
                "type": "string",
                "description": "The short request phrase this applies to, e.g. 'open the chat app'.",
            },
            "resolution": {
                "type": "string",
                "description": "What to do when you see it: the app, surface, and key steps.",
            },
            "taught": {
                "type": "boolean",
                "description": "True if the user is explicitly teaching/correcting you.",
            },
        },
        "required": ["trigger", "resolution"],
    },
    read_only=False,
)
def learn_task(trigger: str, resolution: str, taught: bool = False) -> str:
    if not enabled():
        return "Learning is turned off, so I can't save that shortcut."
    result = learn(trigger, resolution, taught=bool(taught))
    if not result:
        return "ERROR: give both a trigger phrase and what to do for it."
    verb = "Learned" if result["action"] == "learned" else "Refined"
    pin = " (pinned as taught)" if result["taught"] else ""
    return f"{verb} the shortcut for \"{result['trigger']}\"{pin}: {result['resolution']}"


@registry.tool(
    name="recall_task",
    description=(
        "Before doing a task that might be routine, check what you have already "
        "learned about it. Returns the saved resolution (which app, which "
        "surface, which steps) for the closest matching shortcut, plus any "
        "related ones, so you repeat a known-good approach instead of guessing."
    ),
    parameters={
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The user's request, or a short phrase describing the task.",
            },
        },
        "required": ["request"],
    },
)
def recall_task(request: str) -> str:
    if not enabled():
        return "Learning is turned off."
    hits = predict(request, limit=3)
    lines: list[str] = []
    for p in hits:
        if p.get("related"):
            lines.append(f'- (related) "{p["trigger"]}": {p["resolution"]}')
        else:
            tag = "taught" if p.get("taught") else f"{int(p.get('confidence', 0) * 100)}% conf"
            lines.append(f'- "{p["trigger"]}" [{tag}]: {p["resolution"]}')
    ep = similar_episode(request)
    if ep and ep.get("summary"):
        lines.append(f"- (last time, a similar task) {ep['summary'][:200]}")
    if not lines:
        return f"No learned shortcut matches {request!r} yet. Do the task, then save it with learn_task."
    return "What you've learned that fits:\n" + "\n".join(lines)
