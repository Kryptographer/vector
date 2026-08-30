"""The fold -- a local context-virtualization layer.

The claim is deliberately narrow, because the tempting version of it is false.
This does NOT turn a 32K model into a 320K model. What it does is separate two
things that are usually the same thing:

    LOGICAL context   everything the agent can reach -- files, command output,
                      folded results, the handle ledger. Unbounded. Free.
    ACTIVE context    what is actually inside the model's attention window on
                      this request. Bounded. Expensive.

Off, those are identical: a 4,000-line log read on step 3 sits in the window
for steps 3 through 40, re-sent on every one of them. On, the body goes into
SQLite under a handle, the model gets a shape digest, and it reads back only
the slice it turns out to need. The window becomes a working set over a much
larger local state -- disk to RAM to cache to registers, applied to attention.

Four modules, and the split between them is the design:

    ledger    the store. Handles (`R7`) are monotonic and NEVER reused, blobs
              carry a TTL, and what a sweep deleted is remembered so an expired
              handle can be answered with "held, then cleared" rather than the
              silence that reads as a hallucinating model.
    codec     what shape is this text. Detects delimited data, strips a
              reader's line gutter, names columns -- in ONE place, so the
              digest's column names and `gate.stats`'s arithmetic can never be
              reading two different tables.
    gate      the fold itself, the read-back verbs, the egress ceiling, and the
              redaction and sanitisation that guard the surface folding creates.
    tools     those verbs as tool schemas a model can call, registered disabled
              until `set_active(True)`.

Three verbs, and the split between them is the part most designs get wrong:

    peek    a slice of the body        bytes back
    grep    the lines that match       bytes back
    stats   a total over ALL of it     a number back

The first two are the same bet -- that the model wants a few lines out of many.
Where it holds, folding wins enormously. Where it fails it fails *unboundedly*:
a sum, a row count, the commonest value is a property of every row, so the only
way to answer with the first two is to page the whole body back a screen at a
time with the conversation re-sent at every hop. `stats` is the case that bet
cannot cover -- the body is already on this machine, where arithmetic is free,
so the arithmetic happens here and only the scalar crosses.

Usage::

    from pathlib import Path
    from rangerkit.fold import gate, ledger

    ledger.configure(Path.home() / ".rangerkit")
    digest, handle = gate.fold("read_file", body, threshold=2000)
    print(digest)                          # what the model sees
    print(gate.peek(handle, start=1, count=20))
"""

from __future__ import annotations

from . import codec, gate, ledger

__all__ = ["codec", "gate", "ledger", "alias", "tools", "configure"]


def configure(state_dir) -> None:
    """Point the ledger at a state directory. The only setup the fold needs."""
    ledger.configure(state_dir)


def __getattr__(name: str):
    """`alias` and `tools` load on demand.

    `tools` registers three tool specs as a side effect of import, and a caller
    embedding only the fold engine should not acquire them by touching the
    package. `alias` is the one stage that ships off by default.
    """
    if name in ("alias", "tools"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
