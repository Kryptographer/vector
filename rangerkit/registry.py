"""Tool registry.

Tools declare their own JSON schema. The registry renders the schema list the
model sees and dispatches calls back to Python, with uniform error handling so
a failing tool returns a useful message instead of killing the run.
"""

from __future__ import annotations

import inspect
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]
    read_only: bool = True
    # Destructive tools can require an interactive yes before running.
    requires_confirm: bool = False
    enabled: bool = True
    # Plumbing rather than a capability: `enabled` here is owned by a mechanism
    # (the fold's disclosure, the held-result readers) and not by anything
    # the user can switch. It matters because `enabled = False` is otherwise
    # read as "the user turned this off", and the prompt then tells the model to
    # ask for a Permissions switch that does not exist for these.
    internal: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.confirm_hook: Callable[[ToolSpec, dict[str, Any]], bool] | None = None
        # Called with (spec, cleaned arguments) before a resolved call runs.
        # Returns "" to let it run, or replacement text -- conventionally a
        # "BLOCKED: ..." line -- that becomes the tool result WITHOUT the tool
        # executing. This is the conductor's conscience seam, and it exists
        # apart from confirm_hook for two reasons: confirm_hook fires only
        # when `requires_confirm` is set (which shell.confirm_writes can
        # switch off), and its contract is a human's yes/no, not a model's
        # judgement. Ordered BEFORE confirmation on the safety ladder --
        # denylist refuses, the guard refuses, confirmation asks -- so the
        # user is never asked to approve a call that was about to be bounced
        # anyway. Exceptions are swallowed: a conscience must never break a
        # tool call.
        self.guard_hook: Callable[[ToolSpec, dict[str, Any]], str] | None = None
        # Called with a tool name before the name is resolved. The fold's
        # progressive disclosure uses it to open a group the model has just
        # reached into, which is what stops a deferred tool from ever being a
        # dead end. Runs on the caller's thread and does no I/O.
        self.on_dispatch: Callable[[str], None] | None = None
        # Guards the `enabled` flags AS A SET. Settings application passes
        # through states no request must ever see -- disclosure un-hides every
        # tool before scopes and re-hiding narrow it again -- while the agent
        # loop re-reads schemas() between steps on its own worker thread. A
        # read landing inside that window serialized the full 40-tool list for
        # exactly one step, which is two cached-prefix breaks per settings
        # save, even a no-op one. Mutators hold this across the whole
        # transition; schemas()/names() read under it and therefore see the
        # registry before or after, never mid-flight. An RLock because the
        # mutation paths nest (apply_settings -> set_enabled -> apply_registry
        # -> disclosure) on one thread.
        self.lock = threading.RLock()
        # The stop event of the run currently dispatching on THIS thread. Kept
        # thread-local because several chats dispatch through one shared registry
        # on their own worker threads, and a nested run (run_subagent) needs its
        # parent's event -- not another chat's -- to honour a Stop mid-delegation.
        self._local = threading.local()

    def set_current_stop(self, event: Any) -> None:
        """Publish the dispatching run's stop event for this thread."""
        self._local.stop = event

    def current_stop(self) -> Any:
        """The stop event of the run dispatching on this thread, or None."""
        return getattr(self._local, "stop", None)

    def set_current_prompt(self, text: str) -> None:
        """Publish the dispatching run's system prompt for this thread.

        Thread-local for the same reason the stop event is: several chats
        dispatch through one shared registry on their own worker threads, and a
        tool that starts a nested run needs the prompt of the run that called
        IT. Rebuilding one instead would be a quietly different agent -- the
        skills index, the memories and the learned shortcuts all live in the
        prompt the surrounding app assembled, and none of them survive a
        `prompt.build` from inside a tool.
        """
        self._local.prompt = text

    def current_prompt(self) -> str:
        """The system prompt of the run dispatching on this thread, or ""."""
        return getattr(self._local, "prompt", "")

    def set_current_notify(self, fn: Any) -> None:
        """Publish the dispatching run's event sink for this thread.

        Thread-local rather than a module-level `set_...` hook like
        `ask.set_asker`, because unlike the asker there is not one of these per
        app: every chat has its own, and a module global would narrate a phone
        session's nested run into whatever transcript was wired up last.
        """
        self._local.notify = fn

    def current_notify(self) -> Any:
        """The event sink of the run dispatching on this thread, or None."""
        return getattr(self._local, "notify", None)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        read_only: bool = True,
        requires_confirm: bool = False,
        internal: bool = False,
    ) -> Callable[[Callable[..., str]], Callable[..., str]]:
        def deco(fn: Callable[..., str]) -> Callable[..., str]:
            self.register(
                ToolSpec(
                    name=name,
                    description=description,
                    parameters=parameters,
                    fn=fn,
                    read_only=read_only,
                    requires_confirm=requires_confirm,
                    internal=internal,
                )
            )
            return fn

        return deco

    # `enabled` conflates two different things -- "the user switched this off"
    # and "disclosure has deferred it" -- so anything that needs the tools a run
    # is PERMITTED, rather than the ones it can currently see, cannot use get()
    # or names(). Ten call sites were reading self._tools directly, each with a
    # lint suppression copied from the last one; that made the private dict the
    # real API while the public one described a narrower thing. These name that
    # need properly, and take the lock.
    #
    # (enable_only/disable used to live here. Both were dead -- no caller
    # anywhere in the app -- and neither took the lock the class contract says
    # every mutator holds, so they also documented the wrong rule.)

    def all_specs(self) -> list[ToolSpec]:
        """Every registered tool, including disabled and deferred ones."""
        with self.lock:
            return list(self._tools.values())

    def all_named(self) -> dict[str, ToolSpec]:
        """Every registered tool by name, including disabled and deferred."""
        with self.lock:
            return dict(self._tools)

    def spec_for(self, name: str) -> ToolSpec | None:
        """One tool by name whatever its enabled state -- unlike `get`."""
        with self.lock:
            return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        with self.lock:
            return [t.schema() for t in self._tools.values() if t.enabled]

    def names(self) -> list[str]:
        with self.lock:
            return [n for n, t in self._tools.items() if t.enabled]

    def get(self, name: str) -> ToolSpec | None:
        spec = self._tools.get(name)
        return spec if spec and spec.enabled else None

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool call. Never raises: errors come back as text the model can read."""
        if self.on_dispatch is not None:
            try:
                self.on_dispatch(name)
            except Exception:  # noqa: BLE001 - a hook must never break a tool call
                pass

        spec = self.get(name)
        if spec is None:
            # Registered but off is a different situation from never existing,
            # and they need opposite advice. Telling a model that a tool it can
            # see in the Permissions screen simply is not available teaches it
            # to substitute or give up; naming the switch turns a dead end into
            # one click for the user.
            if name in self._tools:
                if self._tools[name].internal:
                    # Plumbing, not a permission. There is no switch for these,
                    # so the Permissions advice below would send the user
                    # hunting for one that does not exist -- and teach the model
                    # to stop and ask instead of getting on with the task.
                    return (
                        f"ERROR: {name} is not part of this configuration. Nothing is "
                        f"switched off and there is nothing for the user to enable -- "
                        f"carry on with the ordinary tools."
                    )
                return (
                    f"ERROR: {name} exists but is switched off right now. Tell the user "
                    f"exactly which switch to flip on the Permissions screen (or the "
                    f"globe button, for web tools) -- do not claim it is impossible and "
                    f"do not ask them to do the task by hand."
                )
            available = ", ".join(self.names())
            return (
                f"ERROR: no such tool {name!r}. Available tools: {available}. "
                f"Call one of these exactly."
            )

        # Strip the salvage markers the parser adds for malformed model output.
        # Done before the guard so it judges the arguments the tool will
        # actually receive, not the parser's scaffolding around them.
        clean = {k: v for k, v in arguments.items() if not k.startswith("_")}

        if self.guard_hook is not None:
            try:
                veto = self.guard_hook(spec, clean)
            except Exception:  # noqa: BLE001 - a hook must never break a tool call
                veto = ""
            if veto:
                return str(veto)

        if spec.requires_confirm and self.confirm_hook is not None:
            if not self.confirm_hook(spec, arguments):
                return "DENIED: the user declined this action. Do not retry it; choose another approach."

        # Bind first, call second. Catching TypeError around the CALL meant any
        # TypeError raised inside the tool body or a library it uses -- a None
        # reaching a format, a Playwright or comtypes internal -- was reported
        # to the model as "bad arguments", with the real traceback discarded.
        # The model then retries with permuted arguments that were never the
        # problem, a wasted-steps loop that looks reasonable in the log.
        try:
            inspect.signature(spec.fn).bind(**clean)
        except TypeError as exc:
            expected = list((spec.parameters.get("properties") or {}).keys())
            return (
                f"ERROR: bad arguments for {name}: {exc}. "
                f"Expected parameters: {expected}. Got: {list(clean.keys())}."
            )
        except (ValueError, AttributeError):
            pass  # unintrospectable callable: fall through and just call it
        try:
            return spec.fn(**clean)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model deliberately
            return f"ERROR running {name}: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"


registry = ToolRegistry()
