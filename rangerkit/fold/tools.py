"""The three read-back tools, and the reason folding is safe.

Elimination is only defensible if nothing is actually lost. These tools are the
guarantee: the body of every folded result is still on this machine,
addressable, and one call away. The model trades "I have all 4,213 lines in my
context" for "I can read any of the 4,213 lines on demand", which is the same
information at a fraction of the price -- and closer to how a person would work.

Two of them answer by returning bytes and one does not, and that split is the
point:

    fold_peek   a slice of the body        bytes back
    fold_grep   the lines that match       bytes back
    fold_stats  a total over ALL of it     a number back

The first two are bets that the model needs a few lines out of many. Where the
bet is right they win enormously; where it is wrong -- a sum, a row count, the
commonest value -- they lose worse than never folding at all, because answering
means paging the whole body back a screen at a time with the conversation
re-sent at every hop. `fold_stats` is the case that bet cannot cover: the
body is already on the user's machine, where arithmetic is free, so the
arithmetic happens there and only the scalar crosses.

They register disabled and only exist to the model once folding is switched
on. The model must never be shown a tool it cannot
call: a handle-based protocol with no way to read a handle is worse than no
protocol at all.
"""

from __future__ import annotations

from typing import Any

from ..registry import registry
from . import gate

_ON = False
_hooks: dict[str, Any] = {}


def set_active(on: bool) -> None:
    """Flip the tools with the mode. Both layers move together, deliberately.

    The registry flag decides whether the model can SEE the tool; the gate
    below re-checks on every call. Two independent checks, because a stale
    schema in a cached prefix could otherwise let a call through after the mode
    was switched off.
    """
    global _ON
    _ON = bool(on)
    for name in gate._READBACK_TOOLS:  # noqa: SLF001 - one list, one owner (gate.py)
        spec = registry.spec_for(name)
        if spec is not None:
            spec.enabled = _ON


def _count(result: str) -> str:
    """Record the read, and separately record a read that named nothing.

    A miss means the model produced a handle that never existed -- it is
    working from the summary and guessing rather than reading the held body.
    Nothing errors upstream, the answer just quietly degrades, so this counter
    is the only warning available and it belongs next to the savings it
    qualifies.

    A handle that EXPIRED is not that, and is deliberately not counted here:
    blobs live three days and sessions are restored from disk, so an old
    conversation reopened on Monday can carry perfectly real handles the sweep
    has since cleared. Charging those to the canary would make the machine's
    own housekeeping look like a model inventing things -- and the canary is
    the number that qualifies every saving above it, so a false reading there
    devalues the whole panel.
    """
    missed = result.startswith("ERROR: no such handle")
    meters = _hooks.get("meters")
    if meters is not None:
        meters.add("peeks")
        if missed:
            meters.add("handle_misses")
    # The same observation, kept per driving model rather than as a lifetime
    # total, because that is the form a DECISION can be made from: the meters
    # answer "has this ever happened", and withdrawing the fold needs "is this
    # model doing it now". See `vector.folding_trusted`.
    on_read = _hooks.get("on_read")
    if on_read is not None:
        try:
            on_read(missed)
        except Exception:  # noqa: BLE001 - bookkeeping must not break a read
            pass
    return result


def attach(meters: Any, on_read: Any = None) -> None:
    _hooks["meters"] = meters
    if on_read is not None:
        _hooks["on_read"] = on_read


_OFF = ("ERROR: folding is switched off, so there are no held results to read. "
        "Use the ordinary tools instead.")


@registry.tool(
    name="fold_peek",
    description=(
        "Read a slice of a result that was held on this machine instead of being sent to "
        "you. Folded results appear in the conversation as a handle like R7 with a short "
        "shape description. Use this to read any part of one. Never invent a handle -- if "
        "you have not seen it in this conversation, it does not exist."
    ),
    parameters={
        "type": "object",
        "properties": {
            "handle": {"type": "string", "description": "The handle, e.g. R7."},
            "start": {"type": "integer", "description": "First line to read (1-based). Default 1."},
            "count": {"type": "integer", "description": "How many lines. Default 60, max 500."},
            "part": {"type": "integer", "description": "Only for a single line too long to "
                                                       "return at once: which piece of it to "
                                                       "read (1, 2, 3...). The reply says how "
                                                       "many pieces there are."},
        },
        "required": ["handle"],
    },
    read_only=True,
    # Switched with the mode, never by the user -- see registry.ToolSpec.internal.
    internal=True,
)
def fold_peek(handle: str, start: int = 1, count: int = 60, part: int = 0) -> str:
    if not _ON:
        return _OFF
    return _count(gate.peek(str(handle), int(start or 1), int(count or 60),
                            part=int(part or 0)))


@registry.tool(
    name="fold_grep",
    description=(
        "Search inside a result held on this machine, by regular expression, and get back "
        "only the matching lines with their line numbers. Far cheaper than reading the "
        "whole thing: prefer this over fold_peek when you know what you are looking for."
    ),
    parameters={
        "type": "object",
        "properties": {
            "handle": {"type": "string", "description": "The handle, e.g. R7."},
            "pattern": {"type": "string", "description": "Regular expression, case-insensitive."},
            "max_hits": {"type": "integer", "description": "Cap on matching lines. Default 40."},
        },
        "required": ["handle", "pattern"],
    },
    read_only=True,
    internal=True,
)
def fold_grep(handle: str, pattern: str, max_hits: int = 40) -> str:
    if not _ON:
        return _OFF
    return _count(gate.grep(str(handle), str(pattern), int(max_hits or 40)))


@registry.tool(
    name="fold_stats",
    description=(
        "Answer a question about a WHOLE held result — a total, a count, the commonest "
        "value — without reading it into your context. The body is on the user's machine, "
        "so the arithmetic happens there and you get back only the number. Use this "
        "instead of paging through a large result with fold_peek: 'what is the sum of "
        "the amount column', 'how many rows are there', 'which merchant appears most "
        "often', 'how many lines say FAILED', 'the median amount', 'the p95 latency'. ops: "
        "count, sum, mean, median, stdev, p90, p95, p99, min, max, distinct, top. Give "
        "`column` for delimited data (the shape line names the columns); leave it out to "
        "count lines. `where` first restricts to matching rows."
    ),
    parameters={
        "type": "object",
        "properties": {
            "handle": {"type": "string", "description": "The handle, e.g. R7."},
            "op": {
                "type": "string",
                "enum": ["count", "sum", "mean", "median", "stdev", "p90", "p95", "p99",
                         "min", "max", "distinct", "top"],
                "description": "What to compute. Default count.",
            },
            "column": {
                "type": "string",
                "description": "Column name from the shape line, or its 1-based position. "
                               "Required for sum/mean/min/max.",
            },
            "where": {
                "type": "string",
                "description": "Optional regular expression; only rows matching it count.",
            },
        },
        "required": ["handle"],
    },
    read_only=True,
    internal=True,
)
def fold_stats(handle: str, op: str = "count", column: str = "", where: str = "") -> str:
    if not _ON:
        return _OFF
    return _count(gate.stats(str(handle), column=str(column or ""),
                             op=str(op or "count"), where=str(where or "")))


# Registered disabled. Nothing sees them until set_active(True).
set_active(False)
