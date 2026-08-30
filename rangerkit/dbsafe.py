"""Corruption handling for the SQLite stores.

Every JSON store in Ranger already survives a damaged file: mobile.py moves a
bad mobile.json aside to .corrupt.json and carries on with fresh pairings, the
scheduler and the session store read unreadable data as empty, config.load falls
back to built-in defaults with a printed warning. The SQLite stores had none of
it. One damaged memory.db raised straight out of configure(), and since every
entry point calls it -- desktop, chat, run, mobile, daemon -- the whole product
became unlaunchable, with a traceback the packaged exe may never even show.

`doctor` was the worst of it: it configures memory before printing its first
line, so the one command whose job is to explain what is wrong failed in exactly
the same way as everything else, with no output at all.

Corruption is a when-not-if event for databases that live through power loss,
antivirus, and an in-place backup restore. Losing one store is recoverable --
apps.db rebuilds itself by scanning, memory.db restores from a backup. Refusing
to start is not.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# What this process moved aside, so doctor and the UI can say so rather than
# leaving the user to wonder where their memories went.
quarantined: list[str] = []


def quarantine(db: Path) -> Path | None:
    """Move a corrupt database aside, taking its WAL sidecars with it.

    The sidecars matter: a -wal left next to a fresh database is replayed into
    it on the next open, which is how a "clean" reset resurrects the very
    corruption it was supposed to clear.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = db.with_name(f"{db.name}.corrupt-{stamp}")
    try:
        db.replace(target)
    except OSError:
        return None
    for suffix in ("-wal", "-shm"):
        try:
            db.with_name(db.name + suffix).unlink(missing_ok=True)
        except OSError:
            pass  # a locked sidecar is not worth failing the recovery over
    quarantined.append(f"{db.name} -> {target.name}")
    return target


def open_or_quarantine(db: Path, prepare: Callable[[], Any]) -> Any:
    """Run `prepare`; if the database is corrupt, move it aside and retry once.

    One retry, deliberately: if a freshly created database also fails, the
    problem is the disk or the directory, and looping would only bury that.
    """
    try:
        return prepare()
    except sqlite3.DatabaseError:
        if quarantine(db) is None:
            raise
        return prepare()
