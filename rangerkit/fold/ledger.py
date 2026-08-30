"""Stage 2 -- the local handle store. Nothing crosses by value that can cross
by handle.

A tool result that stays on this machine costs nothing, forever. The same
result pasted into the conversation costs its full token price on the turn it
arrives and again on every single step after it, because the whole history is
re-sent each time. On a twenty-step tool chain a 4,000-token directory listing
is not 4,000 tokens; it is closer to 60,000.

So: the body goes in SQLite, a handle and a twenty-token shape go to the model,
and the model reads back whatever slice it actually needs with `fold_peek`.

Two tables, and the distinction matters:

  * `blob`   -- a folded tool result. Transient working state, tied to a run.
  * `entity` -- a durable thing the user has (a file, an app, a project) with
                a stable handle and a hot score. Survives sessions.

Handles are `<letter><int>`, monotonic, and NEVER reused. Reusing a handle is
the single most confusing bug this architecture can produce: the model has an
old meaning in its context and the machine has a new one, and nothing errors.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

# F=file D=doc C=category R=result A=app T=task U=url E=redacted entity
KINDS = "FDCRATUE"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entity (
  handle       TEXT PRIMARY KEY,
  kind         TEXT NOT NULL,
  display      TEXT NOT NULL,
  uri          TEXT,
  summary      TEXT,
  fingerprint  TEXT,
  hot          REAL NOT NULL DEFAULT 0,
  epoch        INTEGER NOT NULL DEFAULT 1,
  created_at   INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS blob (
  handle      TEXT PRIMARY KEY,
  tool        TEXT NOT NULL,
  body        TEXT NOT NULL,
  shape       TEXT,
  chars       INTEGER NOT NULL DEFAULT 0,
  fingerprint TEXT,
  created_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS tomb (
  handle      TEXT PRIMARY KEY,
  tool        TEXT NOT NULL,
  shape       TEXT,
  chars       INTEGER NOT NULL DEFAULT 0,
  fingerprint TEXT,
  created_at  INTEGER NOT NULL,
  swept_at    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS counter (
  kind TEXT PRIMARY KEY,
  n    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS swept (
  kind TEXT PRIMARY KEY,
  n    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_hot ON entity(hot DESC);
CREATE INDEX IF NOT EXISTS idx_blob_created ON blob(created_at);
CREATE INDEX IF NOT EXISTS idx_tomb_print ON tomb(fingerprint);
CREATE INDEX IF NOT EXISTS idx_tomb_swept ON tomb(swept_at);
"""

# How many swept bodies keep their survey. A tomb row is the handle, the tool
# that made it, the shape line and the fingerprint -- about a hundred bytes
# against a body that is routinely six figures, which is why this can be
# generous. Bounded all the same: a resident app folding all day would
# otherwise carry every result it ever held, forever.
_TOMB_CAP = 2000

_HANDLE_RE = re.compile(r"^[A-Z][0-9]+$")

_db_path: Path | None = None
# Reentrant on purpose. `upsert_entity` has to hold the lock across its SELECT
# and its INSERT so the two cannot interleave with another thread's -- and
# `next_handle`, called in between, takes the same lock. A plain Lock would
# deadlock the thread against itself there; an RLock lets the owner re-enter
# while still excluding every other thread, which is exactly what the
# read-modify-write needs.
_lock = threading.RLock()
# Blobs older than this are pruned on open. A folded result is working state
# for one run; keeping a week of them would grow the file without bound and
# nothing would ever read them.
_BLOB_TTL = 3 * 24 * 3600

# The high-water handle the TTL has ever swept. Anything at or below it that is
# now missing EXPIRED; anything above it was never issued and the model invented
# it. Those are different failures and only the second one is the canary's
# business -- a session reopened after the weekend brings its old digests back
# into context complete with their read-back trailers, and charging those to
# `handle_misses` makes the one number that qualifies every other number on the
# panel read as a hallucinating model.
#
# In memory this is only a read cache of the `swept` table, and the distinction
# is the whole point. It used to be the entire record, rebuilt from the rows one
# sweep happened to delete -- which meant the memory lasted exactly as long as
# the process that did the deleting, and no longer. A host process gets closed.
# The weekend session above is restored on MONDAY, by which time
# the blobs were swept on Saturday by a process that has since exited; the sweep
# that runs at Monday's startup finds nothing left to delete, rebuilds the map
# from an empty list, and reports every real handle in that transcript as
# invented -- the exact reading this comment exists to prevent, arriving one
# launch after the check that proved it could not.
#
# One thing the table cannot do is recover history nobody wrote down: a handle
# swept by a build that predates it left no record, and reads as invented. That
# is bounded and it is once -- such a handle is three days old before it is
# eligible to be swept at all -- and the alternative, seeding the table from
# `counter`, would quietly widen the rule to call every unissued handle below
# the high-water mark expired. Losing a little of the past beats changing what
# the word means.
_swept_to: dict[str, int] = {}

# When `sweep` last ran, so a new chat does not re-run it every time.
_last_sweep = 0.0


def configure(state_dir: Path) -> None:
    global _db_path
    _db_path = Path(state_dir) / "vector.db"
    with _connect() as db:
        _set_pragmas(db)
        db.executescript(_SCHEMA)
        # A database written before blobs were content-addressed has no
        # fingerprint column, and CREATE TABLE IF NOT EXISTS will not add one.
        # Adding it here rather than rebuilding the table keeps every handle
        # already in a user's transcripts valid; the rows simply dedupe from
        # the next fold onwards.
        try:
            db.execute("ALTER TABLE blob ADD COLUMN fingerprint TEXT")
        except sqlite3.OperationalError:
            pass    # a database created since the column existed
        # After the column is guaranteed, never inside _SCHEMA: on a legacy
        # database the script runs before the ALTER and would index a column
        # that is not there yet.
        db.execute("CREATE INDEX IF NOT EXISTS idx_blob_print ON blob(fingerprint)")
        # upsert_entity looks entities up by (kind, uri) or (kind, display) on
        # every fold of a durable thing, and entities are never deleted -- so
        # without these the scan grows for the life of the install, while
        # holding the one lock every folding thread shares.
        db.execute("CREATE INDEX IF NOT EXISTS idx_entity_kind_uri ON entity(kind, uri)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_entity_kind_display "
                   "ON entity(kind, display)")
        # Drag each counter up to the highest handle that actually exists.
        # `next_handle` trusts the counter alone, so any state where the two
        # disagree -- a restored backup, a half-copied state directory, a
        # counter table lost while the blobs survived -- hands out a handle
        # that is already taken. The INSERT then fails the primary key, `fold`
        # catches it and returns the body verbatim, and folding is quietly off
        # for the life of that database with nothing said. Reconciling here
        # costs one query at startup and makes that state self-healing.
        # Both tables, not just blob. Walking blobs alone left entity handles
        # unreconciled, so the restored-backup case this paragraph describes as
        # self-healing stayed permanently broken for them: upsert_entity's
        # INSERT hits the primary key, its blanket except returns "", and
        # nothing durable can ever be remembered again on that database.
        # Tomb included: a buried handle is still SPENT. Leaving it out of the
        # reconciliation lets a restored backup hand its number to different
        # content, and then two results in one transcript answer to one name.
        rows = (db.execute("SELECT handle FROM blob").fetchall()
                + db.execute("SELECT handle FROM entity").fetchall()
                + db.execute("SELECT handle FROM tomb").fetchall())
        for row in rows:
            handle = str(row["handle"])
            if not _HANDLE_RE.match(handle):
                continue
            letter, number = handle[0], int(handle[1:])
            seen = db.execute("SELECT n FROM counter WHERE kind = ?", (letter,)).fetchone()
            if number > (seen["n"] if seen else 0):
                db.execute(
                    "INSERT INTO counter(kind, n) VALUES(?, ?) "
                    "ON CONFLICT(kind) DO UPDATE SET n = excluded.n",
                    (letter, number),
                )
    # Unconditional at startup: the rate limit exists for the new-chat path,
    # and configure() is a once-per-process event that must not be skipped.
    sweep(min_interval=0)


def sweep(min_interval: float = 3600.0) -> int:
    """Delete blobs past their TTL and record how far the sweep reached.

    Called at startup and again from `new_session`, because startup alone was
    not enough: this is a resident desktop app with scheduled automations, left
    running for days or weeks. The TTL documented as "blobs live three days"
    silently meant "blobs live until the next restart", and every folded result
    -- by definition the large ones -- accumulated verbatim in SQLite the whole
    time. put_blob bumps created_at on a re-read, so bodies still in use never
    age out; only genuinely idle ones do.

    `min_interval` keeps it cheap: a new chat is a common event and the work is
    one DELETE over an indexed column, so once an hour is plenty.
    """
    global _last_sweep
    if _db_path is None:
        return 0
    now = time.time()
    with _lock:
        if min_interval and (now - _last_sweep) < min_interval:
            return 0
        _last_sweep = now
    removed = 0
    with _connect() as db:
        cutoff = int(now) - _BLOB_TTL
        gone = db.execute(
            "SELECT handle FROM blob WHERE created_at < ?", (cutoff,)).fetchall()
        # Demolish the body, keep the survey. The blob is what costs disk and
        # the TTL is right to take it -- but everything AROUND it was earned
        # work and costs almost nothing to keep: which tool produced it, the
        # shape line the digest already computed, and the fingerprint that
        # made it content-addressed in the first place.
        #
        # Deleting those threw away two things the design elsewhere insists
        # on. `_no_handle` tells the model "re-run the tool that produced it"
        # while no longer knowing WHICH tool, which is not a recovery, it is a
        # guess. And `put_blob`'s headline promise -- the same body gets the
        # same handle -- silently stopped holding across a sweep: re-reading
        # an unchanged file three days later minted a rival address for
        # identical bytes, which is the churn that choosing `R7` over eight
        # hex characters exists to avoid.
        db.execute(
            "INSERT OR REPLACE INTO tomb"
            "  (handle, tool, shape, chars, fingerprint, created_at, swept_at) "
            "SELECT handle, tool, shape, chars, fingerprint, created_at, ? "
            "FROM blob WHERE created_at < ?",
            (int(now), cutoff),
        )
        db.execute("DELETE FROM blob WHERE created_at < ?", (cutoff,))
        db.execute(
            "DELETE FROM tomb WHERE handle NOT IN "
            "  (SELECT handle FROM tomb ORDER BY swept_at DESC, rowid DESC LIMIT ?)",
            (_TOMB_CAP,),
        )
        removed = len(gone)
        # The mark goes to disk in the same transaction as the deletion that
        # earned it, and `max` makes it monotonic: a later sweep with nothing to
        # delete cannot lower it, and neither can a restart. That the rule
        # itself is unchanged matters -- only handles at or below a number some
        # sweep actually deleted are expired, so a handle invented above the
        # line is still called invented.
        for row in gone:
            handle = str(row["handle"])
            if not _HANDLE_RE.match(handle):
                continue
            db.execute(
                "INSERT INTO swept(kind, n) VALUES(?, ?) "
                "ON CONFLICT(kind) DO UPDATE SET n = max(n, excluded.n)",
                (handle[0], int(handle[1:])),
            )
        db.commit()
        _swept_to.clear()
        for row in db.execute("SELECT kind, n FROM swept").fetchall():
            _swept_to[str(row["kind"])] = int(row["n"])
    return removed


def expired(handle: str) -> bool:
    """Whether this handle was issued once and has since been swept.

    Deliberately conservative: only handles at or below a number this process
    actually deleted count as expired. A handle the model made up is almost
    always numbered above anything ever issued, and one invented below the
    line is indistinguishable from a real expiry -- so the rule errs towards
    calling a miss a miss, which is the direction that keeps the canary useful.
    """
    text = handle.strip()
    if not is_handle(text):
        return False
    return int(text[1:]) <= _swept_to.get(text[0], 0)


def _connect() -> sqlite3.Connection:
    if _db_path is None:
        raise RuntimeError("vector.ledger.configure() was never called")
    db = sqlite3.connect(_db_path, timeout=10)
    db.row_factory = sqlite3.Row
    return db


def _set_pragmas(db: sqlite3.Connection) -> None:
    """WAL, once per database file, like memory.py and brain.py already do.

    This was the one store still on the default rollback journal with
    synchronous=FULL, and it sits on the agent's hot path: a fold happens per
    large tool result, and each one paid several fsync'd commits.
    """
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        pass  # a filesystem that cannot do WAL still works, just serialised


def ready() -> bool:
    return _db_path is not None


def is_handle(text: str) -> bool:
    return bool(_HANDLE_RE.match(text.strip()))


# ------------------------------------------------------------------- handles

def next_handle(kind: str) -> str:
    """Allocate the next handle of a kind. Monotonic; never reused.

    Serialised through a process lock as well as the SQLite transaction,
    because sub-agents run their own loops on their own threads against the
    same file and two of them folding a result at the same instant must not
    both be handed `R7`.
    """
    letter = (kind or "R")[0].upper()
    if letter not in KINDS:
        letter = "R"
    with _lock, _connect() as db:
        row = db.execute("SELECT n FROM counter WHERE kind = ?", (letter,)).fetchone()
        n = (row["n"] if row else 0) + 1
        db.execute(
            "INSERT INTO counter(kind, n) VALUES(?, ?) "
            "ON CONFLICT(kind) DO UPDATE SET n = excluded.n",
            (letter, n),
        )
        db.commit()
    return f"{letter}{n}"


# --------------------------------------------------------------------- blobs

def put_blob(tool: str, body: str, shape: str = "") -> str:
    """Hold a body and return its handle. The SAME body gets the SAME handle.

    Content-addressed, which is the one idea worth taking wholesale from ARC
    (arXiv:2607.25066): identity is a hash of what produced the result and the
    result itself, so re-reading a file the agent already read hands back the
    address it was already given instead of minting a rival for the identical
    bytes. A long task re-reads constantly -- to check an edit landed, to look
    at a file it half remembers -- and every one of those used to cost another
    row and another number for the model to keep straight.

    What is deliberately NOT taken is their handle FORMAT. ARC shows the first
    eight characters of the SHA1 (`a91f3c20`); this keeps `R7`. The measured
    problem here is a model inventing addresses -- two of ten read-backs on
    qwen3:14b -- and the fix for that is an address a 14B model can carry
    across a few hundred tokens and copy back exactly. Eight hex characters is
    the wrong direction on the one axis this system is actually failing.
    """
    now = int(time.time())
    print_ = hashlib.sha1(
        f"{tool}\x1f".encode("utf-8") + body.encode("utf-8", "replace")
    ).hexdigest()
    # One critical section for the whole read-modify-write. Split across two
    # lock acquisitions, two threads folding identical bytes could both miss the
    # fingerprint and both insert -- defeating the dedup that content-addressing
    # exists to provide and handing the model two names for one result. The same
    # shape was already fixed in upsert_entity, whose comment calls it a real
    # bug; the RLock is re-entrant, so next_handle nests without deadlock.
    with _lock:
        with _connect() as db:
            row = db.execute(
                "SELECT handle FROM blob WHERE fingerprint = ?", (print_,)).fetchone()
            if row:
                # Still in use, so still worth keeping: push its sweep date out
                # rather than letting a body the agent is actively re-reading age
                # out from under the handle it was just handed.
                db.execute("UPDATE blob SET created_at = ? WHERE handle = ?",
                           (now, row["handle"]))
                db.commit()
                return str(row["handle"])
            # Not held -- but these exact bytes may have been held before and
            # swept. Content-addressing says identity is the hash, and the
            # hash has not changed just because the body aged out, so the
            # address should not either. Re-reading an unchanged file after a
            # weekend hands back the handle the model already has in its
            # transcript instead of a rival for identical bytes.
            buried = db.execute(
                "SELECT handle FROM tomb WHERE fingerprint = ?", (print_,)).fetchone()
            if buried:
                revived = str(buried["handle"])
                db.execute(
                    "INSERT OR REPLACE INTO blob"
                    "  (handle, tool, body, shape, chars, fingerprint, created_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (revived, tool, body, shape, len(body), print_, now),
                )
                db.execute("DELETE FROM tomb WHERE handle = ?", (revived,))
                db.commit()
                return revived
        handle = next_handle("R")
        with _connect() as db:
            db.execute(
                "INSERT INTO blob(handle, tool, body, shape, chars, fingerprint, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (handle, tool, body, shape, len(body), print_, now),
            )
            db.commit()
    return handle


def drop_blob(handle: str) -> None:
    """Forget a blob. Only for a fold that decided not to fold after all.

    The handle number is NOT recycled -- `next_handle` keeps counting, because
    a reused handle means the model's context and this machine disagree about
    what it points at and nothing errors.
    """
    if not is_handle(handle):
        return
    with _lock, _connect() as db:
        db.execute("DELETE FROM blob WHERE handle = ?", (handle.strip(),))
        db.commit()


def tomb_meta(handle: str) -> dict[str, Any] | None:
    """What a swept handle used to hold -- tool, shape and size, never a body.

    The point of keeping it: "re-run the tool that produced it" is only advice
    if the model is told which tool. Callers must still decide whether this
    handle belongs to the conversation asking (see `gate._no_handle`); this
    answers what it WAS, not who may hear about it.
    """
    if not is_handle(handle):
        return None
    try:
        with _connect() as db:
            row = db.execute(
                "SELECT handle, tool, shape, chars, created_at, swept_at "
                "FROM tomb WHERE handle = ?", (handle.strip(),)).fetchone()
    except sqlite3.Error:
        return None     # a ledger that cannot be read simply has no survey
    return dict(row) if row else None


def tomb_count() -> int:
    """How many swept handles still carry their survey. For the Vector screen."""
    try:
        with _connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM tomb").fetchone()[0])
    except sqlite3.Error:
        return 0


def blob_meta(handle: str) -> dict[str, Any] | None:
    """Everything about a held result EXCEPT its body.

    describe_handle renders a one-line digest and trim() only asks whether a
    handle still exists, but both went through get_blob -- a SELECT * that
    materialises a body deliberately held because it was too big to send. One
    compaction does up to 40 of those lookups, so writing a twelve-line note
    could read hundreds of megabytes out of SQLite.
    """
    if not is_handle(handle):
        return None
    with _connect() as db:
        row = db.execute(
            "SELECT handle, tool, shape, chars, created_at FROM blob WHERE handle = ?",
            (handle.strip(),)).fetchone()
    return dict(row) if row else None


def get_blob(handle: str) -> dict[str, Any] | None:
    if not is_handle(handle):
        return None
    with _connect() as db:
        row = db.execute("SELECT * FROM blob WHERE handle = ?", (handle.strip(),)).fetchone()
    return dict(row) if row else None


def live_handles(limit: int = 12) -> list[dict[str, Any]]:
    """The handles that actually exist, newest first, for a miss to name.

    A model that invents a handle is guessing from the digest, and the reply it
    used to get -- "no such handle, never invent one" -- is a scolding with no
    way forward: it names the mistake and not one thing the model could do
    instead. Handing back the handles that DO exist turns the dead end into a
    correction, which is the difference between a wasted turn and a recovered
    one. Newest first because a fold the model has just been shown is the one
    it was almost certainly reaching for.
    """
    try:
        with _connect() as db:
            rows = db.execute(
                "SELECT handle, tool, shape, chars FROM blob "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001 - a miss must never become a second failure
        return []


def blob_count() -> int:
    with _connect() as db:
        row = db.execute("SELECT COUNT(*) AS n, COALESCE(SUM(chars), 0) AS c FROM blob").fetchone()
    return int(row["n"] or 0)


def held_chars() -> int:
    """Total characters currently held locally instead of sent."""
    with _connect() as db:
        row = db.execute("SELECT COALESCE(SUM(chars), 0) AS c FROM blob").fetchone()
    return int(row["c"] or 0)


# ------------------------------------------------------------------ entities

def upsert_entity(kind: str, display: str, uri: str = "", summary: str = "",
                  fingerprint: str = "") -> str:
    """Record a durable thing and return its stable handle.

    Keyed on (kind, uri) when there is a uri, else (kind, display), so the same
    file seen in three different runs keeps one handle -- which is the entire
    reason the model can carry a reference across turns without being re-told
    what it points at.
    """
    now = int(time.time())
    key_col, key_val = ("uri", uri) if uri else ("display", display)
    # ONE critical section spanning the SELECT and the INSERT. Splitting them --
    # a SELECT under the lock, the lock released, then an INSERT under the lock
    # again -- was a real bug: two sub-agent threads folding the same file at the
    # same instant both miss the SELECT, both fall through, and both INSERT, so
    # one durable thing ends up with two handles (F1 and F2 for one path). That
    # is precisely the never-reused-handle invariant this module exists to hold,
    # and nothing errors when it breaks -- the model just carries two names for
    # one thing. Holding the lock across the whole read-modify-write closes it;
    # `next_handle` re-enters the RLock without deadlocking.
    with _lock:
        with _connect() as db:
            row = db.execute(
                f"SELECT handle FROM entity WHERE kind = ? AND {key_col} = ?", (kind, key_val)
            ).fetchone()
            if row:
                handle = row["handle"]
                db.execute(
                    "UPDATE entity SET last_seen_at = ?, hot = hot + 1, "
                    "summary = COALESCE(NULLIF(?, ''), summary), "
                    "fingerprint = COALESCE(NULLIF(?, ''), fingerprint) WHERE handle = ?",
                    (now, summary, fingerprint, handle),
                )
                db.commit()
                return handle
        handle = next_handle(kind)
        with _connect() as db:
            db.execute(
                "INSERT INTO entity(handle, kind, display, uri, summary, fingerprint, hot, "
                "epoch, created_at, last_seen_at) VALUES(?, ?, ?, ?, ?, ?, 1, 1, ?, ?)",
                (handle, kind, display, uri, summary, fingerprint, now, now),
            )
            db.commit()
    return handle


def hot_entities(limit: int = 40) -> list[dict[str, Any]]:
    """The catalogue that goes in the cached prefix.

    Sorted by handle, NOT by relevance, even though `hot` decides membership.
    Relevance order changes every turn and would rewrite the cached prefix
    every turn -- turning the cheapest region of the prompt into the most
    expensive one. `hot` selects; the handle sorts.
    """
    with _connect() as db:
        rows = db.execute(
            "SELECT handle, kind, display, summary FROM entity "
            "ORDER BY hot DESC, last_seen_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return sorted((dict(r) for r in rows), key=lambda r: (r["handle"][0], int(r["handle"][1:])))


def decay(factor: float = 0.9) -> None:
    """Age the hot scores so last month's favourite is not permanent."""
    with _lock, _connect() as db:
        db.execute("UPDATE entity SET hot = hot * ?", (factor,))
        db.commit()


def stats(since: float = 0.0) -> dict[str, int]:
    """Counts for the panel. `since` bounds the BLOBS only, and deliberately.

    Blobs live three days so a handle stays readable across a restart, but the
    headline they feed -- logical context held locally, against what actually
    sat in the model's window -- divides them by tokens measured since the
    meters were last reset. Counting every blob in the file against that put two
    different windows on the two sides of one ratio: yesterday's leftovers
    inflated today's claim, and immediately after "Reset counters" the panel
    credited the mode with everything it had ever held and nothing it had spent.

    Entities are not bounded, because they are the opposite kind of thing: a
    durable catalogue of what is on this machine, whose whole value is that it
    survives the session.
    """
    with _connect() as db:
        ent = db.execute("SELECT COUNT(*) AS n FROM entity").fetchone()
        blb = db.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(chars), 0) AS c FROM blob "
            "WHERE created_at >= ?", (int(since or 0),)
        ).fetchone()
    return {
        "entities": int(ent["n"] or 0),
        "blobs": int(blb["n"] or 0),
        "held_chars": int(blb["c"] or 0),
    }


def clear() -> None:
    """Drop every handle. Counters keep going -- handles are never reused."""
    with _lock, _connect() as db:
        db.execute("DELETE FROM blob")
        db.execute("DELETE FROM entity")
        db.commit()
