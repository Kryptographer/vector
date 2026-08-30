"""Ranger Kit -- the memory layer and the fold, extracted to stand alone.

Two mechanisms, one idea between them: an agent's useful context is far larger
than the window it is allowed to think in, and both halves of that gap can be
closed on the machine rather than in the prompt.

    rangerkit.memory   what survives a session. Facts written by the agent or
                       the user, ranked by keyword overlap plus a spread of
                       related terms, graded by what they were worth to a turn
                       that finished, and filed by subject so recall can be
                       narrowed to the one the request is about.

    rangerkit.brain    what the agent LEARNED to do, as opposed to what it was
                       told. Trigger phrases, the resolution that worked, a
                       use/win tally that turns a repeated request into a
                       one-liner. Same database, linkable to facts.

    rangerkit.fold     what never has to enter the window at all. A large tool
                       result goes into a local ledger under a handle, the
                       model receives a shape digest, and three read-back verbs
                       reach the parts it turns out to need.

Everything is standard library and SQLite. Nothing here reaches the network,
loads a model, or requires one to be running -- which is the property that
makes the benchmark harnesses in `bench/` deterministic.

Quickstart::

    from pathlib import Path
    from rangerkit import memory

    memory.configure(Path.home() / ".rangerkit")
    memory.remember("Dave runs the training scripts on the RTX 4090")
    print(memory.recall("what GPU do I have"))

See `docs/MEMORY.md` and `docs/FOLDING.md` for the mechanisms, and
`docs/BENCHMARKS.md` for how to measure them on your own store.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["memory", "brain", "fold", "semantic", "registry", "dbsafe", "__version__"]


def __getattr__(name: str):
    """Import submodules lazily.

    `import rangerkit` should cost nothing but the name. Importing `memory`
    eagerly here would open no database and start no thread, but it would drag
    `brain`, `semantic` and the tokenizer tables in behind it for a caller that
    only wanted `fold` -- and the fold is the half most likely to be embedded
    in something that already has a memory layer of its own.
    """
    if name in ("memory", "brain", "fold", "semantic", "registry", "dbsafe"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
