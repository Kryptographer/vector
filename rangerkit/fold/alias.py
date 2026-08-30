"""Stage 5 -- alias. The only open bet in the pipeline, and it ships off.

Everything else here is well supported. This is not, so it is built last, kept
isolated, defaults to `false`, and can be deleted without touching another
file. Killing it is a success outcome, not a failure: it would mean the 90%
that works got built and the 10% that does not got declined.

What it does: replaces strings the tokenizer has never seen -- absolute
Windows paths, UUIDs, long URLs, generated identifiers -- with two-token
handles, and puts the legend in the cached prefix where it bills at a tenth.
Measured, those strings really are expensive:

    C:\\Users\\me\\Documents\\Finance\\budget_2026.xlsx   15 tok -> F1 (2)   7.5x
    550e8400-e29b-41d4-a716-446655440000                  21 tok -> R3 (2)  10x
    https://api.example.com/v2/transactions?filter=cat    16 tok -> U2 (2)   8x

What it must NEVER do is alias ordinary English. This is the trap that eats
projects like this one:

    "spreadsheet"  = 1 token        "A1"    = 2 tokens
    "information"  = 1 token        "AB127" = 2 tokens
    "calculate"    = 1 token        "BZ4821" = 3 tokens

Substituting there DOUBLES the cost. The tokenizer is already a codebook --
one trained on trillions of words, which handed single-token slots to whatever
recurs most. A hand-built grid competing with it is a second, worse codebook.
So the pool is not invented, it is SEARCHED: generate candidates, measure each
one, keep only those that actually cost <= 2 tokens.

Every count in here is measured, never reasoned about. An unmeasured count is
a bug.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from . import ledger

# Long strings only. Below this length the substitution cannot win, whatever
# the string is, and the check is cheaper than the measurement.
MIN_LENGTH = 24

_CANDIDATE = re.compile(
    r"(?:[A-Za-z]:\\[^\s\"'<>|]{12,})"                       # C:\Users\...\file.xlsx
    r"|(?:/(?:home|Users|var|opt|mnt)/[^\s\"'<>|]{10,})"      # /home/user/...
    r"|(?:https?://[^\s\"'<>)\]]{16,})"                       # URLs
    r"|(?:\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b)"                      # UUIDs
    r"|(?:\b[a-z][a-z0-9]*(?:_[a-z0-9]+){3,}\b)"              # snake_case_identifiers
)


# Letters an alias handle may use. Every one of these must stay OUTSIDE
# ledger.KINDS: both schemes hand the model `<letter><number>` names in the
# same prompt, so a letter in both means one prompt can define U3 twice, and
# Codebook.decode would then rewrite what the model said about the ledger's U3.
# The default used to be "PUQZ", whose U is ledger's `url` kind -- harmless
# only because the entity half has no production producer yet. Derived rather
# than hand-picked, so adding a kind to the ledger cannot reintroduce it.
_ALIAS_LETTERS = "".join(c for c in "PQZWYXKMJVNH" if c not in ledger.KINDS)[:4]


def build_pool(counter: Callable[[str], int], kinds: str = "", n: int = 400) -> list[str]:
    """Search for cheap handles instead of assuming any short string is cheap.

    Generates `<letter><int>` candidates, measures every one with the same
    counter that prices the request, and keeps only those costing at most two
    tokens -- cheapest first, so the hottest strings get the cheapest handles.

    `kinds` defaults to _ALIAS_LETTERS, which excludes every letter the ledger
    hands out, so the two naming schemes cannot collide in one prompt.
    """
    kinds = kinds or _ALIAS_LETTERS
    scored: list[tuple[int, str]] = []
    for letter in kinds:
        for i in range(1, n + 1):
            candidate = f"{letter}{i}"
            scored.append((counter(candidate), candidate))
    scored.sort()
    return [name for cost, name in scored if cost <= 2]


class Codebook:
    """Alias assignments for one epoch.

    Epoch discipline, because the alternative is the most confusing bug this
    architecture can produce:

      * append-only within an epoch -- an alias NEVER changes meaning while the
        model might still have the old meaning in its context;
      * rotation invalidates the cached prefix, so it happens on a schedule or
        a model change, never per request;
      * a model change forces rotation, because the tokenizer moved and every
        measurement in here is now stale.
    """

    def __init__(self, counter: Callable[[str], int], epoch: int = 1,
                 min_saving: int = 4, max_entries: int = 60):
        self.counter = counter
        self.epoch = epoch
        self.min_saving = min_saving
        self.max_entries = max_entries
        self.pool = build_pool(counter)
        self.forward: dict[str, str] = {}
        self.reverse: dict[str, str] = {}
        self.saved_tokens = 0
        self.seen: dict[str, int] = {}

    # ------------------------------------------------------------ assignment

    def _assign(self, raw: str) -> str | None:
        if len(self.forward) >= self.max_entries or not self.pool:
            return None
        expanded = self.counter(raw)
        handle = self.pool[len(self.forward)]
        alias_cost = self.counter(handle)
        # The legend entry itself costs something. It lives in the cached
        # prefix and so bills at roughly a fifth of face value across a
        # session -- which is the entire reason this layer is viable at all --
        # but it is still a cost, and it is charged against the first use.
        legend_cost = int((expanded + alias_cost + 2) * 0.215)
        if expanded - alias_cost <= self.min_saving:
            return None
        self.forward[raw] = handle
        self.reverse[handle] = raw
        self.saved_tokens += max(0, expanded - alias_cost - legend_cost)
        return handle

    def encode(self, text: str) -> str:
        """Substitute known-expensive strings; assign handles for repeat offenders.

        A string is aliased on its SECOND sighting, not its first. One mention
        of a path is cheaper left alone than paid for twice -- once inline and
        once in the legend.
        """
        if not text:
            return text

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            if len(raw) < MIN_LENGTH:
                return raw
            existing = self.forward.get(raw)
            if existing:
                return existing
            self.seen[raw] = self.seen.get(raw, 0) + 1
            if self.seen[raw] < 2:
                return raw
            return self._assign(raw) or raw

        return _CANDIDATE.sub(replace, text)

    def decode(self, text: str) -> str:
        """Expand handles the model emitted back into real strings.

        Word-boundary anchored: a handle must not match inside a longer token,
        or `P1` would rewrite part of `P12` and hand a tool the wrong path.
        """
        if not text or not self.reverse:
            return text
        # ONE left-to-right pass, so raw text inserted for one handle can never be
        # re-scanned and rewritten by another. The old per-handle loop expanded
        # P10 -> "...P1..." and then a later iteration for P1 rewrote the P1
        # sitting inside that expansion -- turning a correct path into a wrong one,
        # silently, which for a handle protocol is a wrong action on a real
        # machine. Longest handle first so P12 wins over P1 at a shared position;
        # a replacement FUNCTION (not a template) means backslashes in the raw
        # value go in verbatim rather than being read as group references.
        handles = sorted(self.reverse, key=len, reverse=True)
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(h) for h in handles) + r")\b")
        return pattern.sub(lambda m: self.reverse[m.group(0)], text)

    def decode_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Expand aliases everywhere in the arguments, nested ones included.

        Same hole redaction had: rewriting only the top level left an alias
        un-expanded anywhere the model nested one -- and an un-expanded alias is
        a made-up string typed into a real application.
        """
        if not self.reverse:
            return args
        from .gate import walk

        return {key: walk(value, self.decode) for key, value in args.items()}

    # ---------------------------------------------------------------- legend

    def legend(self) -> str:
        """The block that goes in the cached prefix. Empty until something is aliased."""
        if not self.reverse:
            return ""
        from .gate import sanitize  # a filename in the legend is attacker-influenced

        lines = [f"Alias legend (epoch {self.epoch}). These handles stand for exact strings; "
                 f"use the handle, never retype the string:"]
        for handle, raw in sorted(self.reverse.items()):
            lines.append(f"  {handle} = {sanitize(raw)[:160]}")
        return "\n".join(lines)

    def rotate(self) -> None:
        """New epoch: drop every binding and start again. Invalidates the prefix."""
        self.epoch += 1
        self.forward.clear()
        self.reverse.clear()
        self.seen.clear()

    def validate(self, text: str) -> list[str]:
        """Handles the model used that this epoch never defined.

        A hallucinated handle is the most dangerous failure this layer can
        produce -- it looks exactly like a real reference. Caught here it costs
        one clarification; uncaught it costs a wrong action on a real machine.
        """
        used = set(re.findall(r"\b[A-Z][0-9]{1,3}\b", text or ""))
        return sorted(h for h in used if h not in self.reverse)

    def stats(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "entries": len(self.reverse),
            "pool": len(self.pool),
            "saved_tokens": self.saved_tokens,
        }
