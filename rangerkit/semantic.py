"""The optional semantic side of recall -- a seam, not an engine.

`memory.py` ranks facts by weighted keyword overlap, and for a store of a few
hundred personal facts that is both adequate and instant. It is also, on its
own, blind in one specific way: a question worded differently from the fact
that answers it scores zero, however well the fact answers it. Scattershot
(see `memory._pellets`) closes most of that gap from inside the store's own
vocabulary. A semantic retriever closes the rest -- at the cost of a model.

That cost is why this is a seam rather than a dependency. An embedding model
competes with the main model for VRAM, and the whole point of the keyword
default is that it asks for none. So nothing is bundled, nothing is imported,
and nothing here reaches the network: register a backend and recall fuses its
hits in, register nothing and recall is keyword-only. Both are supported
configurations, and the second is the default.

**The fusion is the part worth having.** Register a backend and its hits do not
get appended to the keyword hits -- the two retrievers compete for the same
`recall_limit` slots by reciprocal rank fusion, which is what lets semantic
recall SHRINK the tokens sent to a model (by displacing a weak keyword hit)
rather than only ever adding to them. RRF works on ranks, so an overlap count
and a cosine score never need a shared scale, and a backend that returns
garbage costs at most its share of the slots.

Writing a backend is four methods, all optional except `search`:

    class MyBackend:
        def configure(self, state_dir, cfg): ...      # called by memory.configure
        def mirror_fact(self, mem_id, category, fact, created_at): ...
        def remove_fact(self, mem_id): ...
        def search(self, query, limit=5): ...         # -> [(mem_id | None, text)]

    from rangerkit import memory, semantic
    semantic.register(MyBackend())
    memory.configure(state_dir)

`search` returns `(mem_id, snippet)` pairs, best first. Return the SQLite row id
when the backend knows it and `None` when it does not -- a snippet whose text
matches a stored fact is mapped back to its row anyway, and one that matches
nothing is carried into the results as loose text. That is deliberate: a
retriever holding knowledge the fact table does not is still allowed to answer.

SQLite remains the source of truth in every configuration. Ids, `forget`, and
exact-delete semantics stay where they are, so a backend that is uninstalled,
broken, or changed loses nothing. Every call below is wrapped by its caller and
every failure degrades silently to keyword-only, because a store that stops
answering when its optional half breaks is worse than one that never had it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SemanticBackend(Protocol):
    """What `register` accepts. Only `search` is required in practice."""

    def search(self, query: str, limit: int = 5) -> list[tuple[int | None, str]]:
        """Return `(mem_id | None, snippet)` pairs, best first."""
        ...


_BACKEND: Any = None


def register(backend: Any) -> None:
    """Install a semantic backend. Pass `None` to remove one.

    Safe to call before or after `memory.configure`; if a store is already
    configured the backend is handed the same paths on the next `configure`.
    """
    global _BACKEND
    _BACKEND = backend


def active() -> bool:
    """True when a backend is installed. `memory.recall` reads this for its stats."""
    return _BACKEND is not None


def backend() -> Any:
    """The installed backend, or None. For callers that want to introspect it."""
    return _BACKEND


def _call(name: str, *args: Any, **kw: Any) -> Any:
    """Invoke an optional method. A backend missing it is not an error."""
    fn = getattr(_BACKEND, name, None) if _BACKEND is not None else None
    if fn is None:
        return None
    return fn(*args, **kw)


# --------------------------------------------------------------- the seam
# Each of these mirrors one call site in memory.py. They are total functions:
# with no backend registered every one is a no-op returning an empty result,
# which is what makes keyword-only the default rather than a special case.


def configure(state_dir: Path, cfg: dict[str, Any] | None = None) -> None:
    _call("configure", Path(state_dir), dict(cfg or {}))


def mirror_fact(mem_id: int, category: str, fact: str, created_at: str) -> None:
    _call("mirror_fact", mem_id, category, fact, created_at)


def remove_fact(mem_id: int) -> None:
    _call("remove_fact", mem_id)


def search(query: str, limit: int = 5) -> list[tuple[int | None, str]]:
    hits = _call("search", query, limit=limit)
    return list(hits) if hits else []
