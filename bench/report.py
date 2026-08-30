"""One command, one file: run every harness and bundle the evidence.

Each harness in this directory answers its own question well, but handing the
results to somebody else means copying console text and then remembering what
machine it ran on. This wraps the loop: run them all, record what they ran on,
and fold results, platform, package version and git revision into ONE file that
travels.

    python bench/report.py                 # self-test + all three harnesses
    python bench/report.py --scale medium  # the fold bench at 8x
    python bench/report.py --quick         # self-test and the fold bench only

Output lands in bench-results/:

    rangerkit-bench.json     the machine-readable bundle -- send this one
    rangerkit-bench.md       the same story for human eyes

Stdlib only, like everything else here. The report contains no file contents
from your machine -- only benchmark numbers, a platform string, and the two
version strings. Nothing is uploaded; both files are written locally and that
is the end of it.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"


_ANSI = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """The harnesses colour their output; a report parsing it must not care."""
    return _ANSI.sub("", text)


def _git_rev() -> str:
    """The revision this ran at, or a clear absence of one.

    A benchmark whose provenance is unknown is worth less than one that says so,
    which is why this returns a word rather than an empty string.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return "not-a-git-checkout"
        rev = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10)
        return rev + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _environment() -> dict[str, Any]:
    try:
        from rangerkit import __version__ as ver
    except Exception:  # noqa: BLE001 - a report must not fail on its own header
        ver = "unknown"
    return {
        "rangerkit_version": ver,
        "git_revision": _git_rev(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def _run_selftest() -> dict[str, Any]:
    out = subprocess.run([sys.executable, str(ROOT / "tests" / "selftest.py")],
                         capture_output=True, text=True)
    # Match the tally line specifically. A substring search for "passed"/"failed"
    # also matches a CHECK NAME containing either word, which put a test title
    # into the summary field of every report this wrote.
    tally = re.compile(r"^\s*(\d+)/(\d+) passed\s*$")
    found = [m for m in (tally.match(_strip_ansi(l)) for l in out.stdout.splitlines()) if m]
    summary = f"{found[-1].group(1)}/{found[-1].group(2)} passed" if found else "no tally line"
    return {"exit_code": out.returncode, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run every harness and bundle the results.")
    ap.add_argument("--scale", choices=["small", "medium", "large"], default="small",
                    help="corpus scale for the fold bench (default: small)")
    ap.add_argument("--quick", action="store_true",
                    help="skip the two memory harnesses (they are the slow ones)")
    ap.add_argument("--out", default=str(ROOT / "bench-results"))
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}  rangerkit benchmark report{RESET}")
    env = _environment()
    print(f"{DIM}  {env['platform']} · python {env['python']} · "
          f"rangerkit {env['rangerkit_version']} · {env['git_revision']}{RESET}\n")

    results: dict[str, Any] = {"environment": env, "harnesses": {}}
    failures: list[str] = []

    print("  running self-test ...", end=" ", flush=True)
    st = _run_selftest()
    results["harnesses"]["selftest"] = st
    ok = st["exit_code"] == 0
    print(f"{GREEN}{st['summary']}{RESET}" if ok else f"{RED}FAILED{RESET}")
    if not ok:
        failures.append("selftest")

    print(f"  running fold bench ({args.scale}) ...", end=" ", flush=True)
    from foldsim import run_sim as fold_sim  # noqa: E402 - after sys.path is set
    fold = fold_sim(args.scale, quiet=True)
    results["harnesses"]["fold"] = fold
    print(f"{GREEN}passed{RESET}" if fold["passed"] else f"{RED}FAILED{RESET}")
    if not fold["passed"]:
        failures.append("fold")

    if not args.quick:
        print("  running memory bench ...", end=" ", flush=True)
        import memsim  # noqa: E402
        mem = memsim.run_sim(quiet=True)
        results["harnesses"]["memory"] = mem
        mem_ok = bool(mem.get("passed", True))
        print(f"{GREEN}passed{RESET}" if mem_ok else f"{RED}FAILED{RESET}")
        if not mem_ok:
            failures.append("memory")

        print("  running recall-rate bench ...", end=" ", flush=True)
        rr = subprocess.run([sys.executable, str(ROOT / "bench" / "recallrate.py")],
                            capture_output=True, text=True)
        results["harnesses"]["recall_rate"] = {
            "exit_code": rr.returncode,
            "report": rr.stdout.strip().splitlines(),
        }
        print(f"{GREEN}done{RESET}" if rr.returncode == 0 else f"{RED}FAILED{RESET}")

    results["passed"] = not failures
    results["failures"] = failures

    jpath = outdir / "rangerkit-bench.json"
    jpath.write_text(json.dumps(results, indent=2), encoding="utf-8")
    mpath = outdir / "rangerkit-bench.md"
    mpath.write_text(_markdown(results), encoding="utf-8")

    print(f"\n  wrote {jpath}")
    print(f"  wrote {mpath}\n")
    if failures:
        print(f"  {RED}failed: {', '.join(failures)}{RESET}\n")
    return 0 if not failures else 1


def _markdown(r: dict[str, Any]) -> str:
    env = r["environment"]
    L = [
        "# rangerkit benchmark report",
        "",
        f"- **rangerkit** {env['rangerkit_version']} (`{env['git_revision']}`)",
        f"- **Python** {env['python']} on {env['platform']}",
        f"- **Run at** {env['run_at']}",
        "",
        "Every harness below is deterministic and runs with no model and no network.",
        "",
        "## Self-test",
        "",
        f"`{r['harnesses']['selftest']['summary'] or 'no summary'}` "
        f"(exit {r['harnesses']['selftest']['exit_code']})",
        "",
    ]

    fold = r["harnesses"].get("fold")
    if fold:
        off, on, rb = fold["arms"]["off"], fold["arms"]["on"], fold["arms"]["readback"]
        cut = 1.0 - on["cumulative_chars"] / max(1, off["cumulative_chars"])
        L += [
            f"## Fold ({fold['scale']} corpus)",
            "",
            "| | fold off | fold on | read-back only |",
            "|---|---:|---:|---:|",
            f"| peak active context (chars) | {off['peak_chars']:,} | {on['peak_chars']:,} "
            f"| {rb['peak_chars']:,} |",
            f"| cumulative prefill (chars) | {off['cumulative_chars']:,} "
            f"| {on['cumulative_chars']:,} | {rb['cumulative_chars']:,} |",
            f"| round trips | {off['round_trips']} | {on['round_trips']} | {rb['round_trips']} |",
            f"| retrieval answers | {off['retrieval']}/{off['retrieval_of']} "
            f"| {on['retrieval']}/{on['retrieval_of']} | {rb['retrieval']}/{rb['retrieval_of']} |",
            f"| aggregate answers | {off['aggregate']}/{off['aggregate_of']} "
            f"| {on['aggregate']}/{on['aggregate_of']} | {rb['aggregate']}/{rb['aggregate_of']} |",
            "",
            f"Cumulative prefill cut by **{cut:.1%}**. The read-back column is the fold's",
            "losing case priced rather than hidden: paging a whole body back to answer a",
            "question about every row costs more than never folding it, which is the case",
            "`stats` exists to answer instead.",
            "",
            "**Gates**",
            "",
        ]
        L += [f"- {'PASS' if g['ok'] else 'FAIL'} — {g['claim']}" for g in fold["gates"]]
        L += [""]

    if "recall_rate" in r["harnesses"]:
        L += ["## Recall rate", "", "```"]
        L += r["harnesses"]["recall_rate"]["report"]
        L += ["```", ""]

    L += [
        "## What this does and does not show",
        "",
        "These are **mechanical** results: with a fixed workload, the mechanisms do what",
        "they claim. They say nothing about whether a real model asks recall better",
        "questions, writes better facts, or picks the right read-back verb — those are",
        "behavioural questions, they depend on the model and on your own data, and no",
        "arrangement of these harnesses can settle them.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
