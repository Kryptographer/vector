"""An UNRIGGED recall bench.

Method, stated so the number can be discounted properly:
  1. Facts written first, in the voice a real store accumulates them in.
  2. Questions written afterwards as a person would type them -- NOT reverse
     engineered from the pellet kinds. Some share words with their fact, some
     do not; that mix is the thing being measured.
  3. Each question is tagged with the id of the fact that answers it, or None
     when the store genuinely cannot answer -- those are the false-positive
     controls.
  4. Same store, same questions, one variable: `scattershot`.

The bias that remains: one author wrote both halves. Treat this as an
indication, not a measurement of anyone's real store.

The persona below is invented. It is written in the register a real store
accumulates facts in, because that is what the bench is measuring, but no
name, code, date or number in it belongs to anybody. Nothing here was
transcribed from a real store, and nothing here should be read as if it was.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rangerkit import memory as m  # noqa: E402 - runnable without installation

FACTS = [
    "Dave keeps all his active projects in D:\\work",
    "Dave runs the training scripts on the RTX 4090 in the workstation",
    "The workstation has 64 GB of memory and a 2 TB NVMe drive",
    "Dave prefers dark mode in every editor and terminal",
    "Dave uses Neovim for quick edits and VS Code for longer sessions",
    "The deployment pipeline for the billing API is managed with Terraform",
    "Staging deploys run automatically on every merge to main",
    "Production deploys need a manual approval from Priya",
    "The billing API is written in Go and the frontend is TypeScript",
    "Dave's team uses Linear for tickets and Slack for everything else",
    "Standup is at 9:15 every weekday morning",
    "Priya is the tech lead and reviews all schema changes",
    "Sarah is allergic to shellfish and avoids seafood restaurants",
    "Sarah's birthday is the 14th of March",
    "Dave and Sarah were married in 2019",
    "Their dog Biscuit is a border collie who needs two walks a day",
    "The vet appointment for Biscuit is the first Tuesday of each month",
    "Dave's mother lives in Leeds and visits at Christmas",
    "The house alarm code is 1234",
    "The garage door code is 5678",
    "Bin collection is Thursday mornings",
    "The boiler was serviced in October and is due again next October",
    "Dave's car is a 2015 Skoda Octavia with 62,000 miles on it",
    "The Octavia needs its timing belt done before 100,000 miles",
    "Dave's mortgage is with Ashcombe on a five year fix ending in 2027",
    "Dave plays bass and records in Reaper with a Scarlett interface",
    "The bass is a Fender Precision from 1996",
    "Dave mixes with Ozone and prints stems at 48 kHz",
    "The studio monitors are Yamaha HS8s on isolation pads",
    "Dave's coffee order is a flat white with oat milk",
    "Dave is trying to cut down to two coffees a day",
    "The gym membership is at PureGym and renews in January",
    "Dave swims on Mondays and lifts on Wednesdays and Fridays",
    "Dave's passport expires in March 2031",
    "The flight to Lisbon is on the 3rd of June from Gatwick",
    "Dave gets motion sick on boats and takes tablets beforehand",
    "The spare house key is with the neighbour across the road",
    "Dave's accountant is Marcus at Hedley and Cole",
    "The tax return is due at the end of January every year",
    "Dave invoices clients on the last working day of the month",
]

# (question, id of the fact that answers it or None)
QUESTIONS = [
    ("where do I keep my projects", 1),
    ("what GPU do I have", 2),
    ("how much RAM is in the workstation", 3),
    ("what theme do I use", 4),
    ("which editor do I use", 5),
    ("how do we ship the billing service", 6),
    ("when does staging deploy", 7),
    ("who approves production releases", 8),
    ("what language is the backend written in", 9),
    ("where do we track tickets", 10),
    ("what time is standup", 11),
    ("who reviews database migrations", 12),
    ("what food is Sarah allergic to", 13),
    ("when is Sarah's birthday", 14),
    ("when did we get married", 15),
    ("what breed is the dog", 16),
    ("when is Biscuit's next vet visit", 17),
    ("where does my mum live", 18),
    ("what is the alarm code", 19),
    ("what is the code for the garage", 20),
    ("what day are the bins", 21),
    ("when was the boiler last serviced", 22),
    ("what car do I drive", 23),
    ("when does the timing belt need doing", 24),
    ("who is my mortgage with", 25),
    ("what DAW do I record in", 26),
    ("what bass do I play", 27),
    ("what sample rate do I print at", 28),
    ("what speakers are in the studio", 29),
    ("what coffee do I order", 30),
    ("how many coffees am I allowed", 31),
    ("where is my gym", 32),
    ("what days do I train", 33),
    ("when does my passport run out", 34),
    ("when is the Lisbon trip", 35),
    ("do I get seasick", 36),
    ("who has the spare key", 37),
    ("who does my accounts", 38),
    ("when is the tax return due", 39),
    ("when do I send invoices", 40),
    # Controls: the store cannot answer these.
    ("what is my blood type", None),
    ("what is the wifi password", None),
    ("when is my dentist appointment", None),
    ("what size shoes do I wear", None),
    ("which airline did I book", None),
]


# Written as task requests, not requests to recall a past conversation. These
# and the hints are hand-authored fixtures, not held-out model evaluation.
IMPLICIT_QUESTIONS = [
    ("Open the active projects folder", 1),
    ("Estimate whether the next training run fits on my GPU", 2),
    ("Check the workstation RAM before suggesting an upgrade", 3),
    ("Set up an editor theme that suits me", 4),
    ("Pick an editor for a quick patch", 5),
    ("Prepare the billing service deployment checklist", 6),
    ("Tell the team what happens after a merge to main", 7),
    ("Prepare the production release approval request", 8),
    ("Choose a backend library for the billing API", 9),
    ("Create a ticket in the tracker we use", 10),
    ("Plan a meeting that avoids standup", 11),
    ("Find a reviewer for the schema migration", 12),
    ("Suggest a dinner restaurant for Sarah", 13),
    ("Schedule a birthday reminder for Sarah", 14),
    ("Draft an anniversary card for Sarah", 15),
    ("Make a daily care plan for Biscuit", 16),
    ("Plan next month's vet visit", 17),
    ("Plan a Christmas visit to my mother", 18),
    ("Help the house sitter disarm the alarm", 19),
    ("Help the visitor open the garage door", 20),
    ("Add a bin collection reminder", 21),
    ("Schedule the next boiler service", 22),
    ("Find compatible parts for my car", 23),
    ("Schedule the Octavia timing belt work", 24),
    ("Prepare for the mortgage fix ending", 25),
    ("Set up a recording session in my DAW", 26),
    ("Suggest strings for my bass guitar", 27),
    ("Export stems at my usual sample rate", 28),
    ("Suggest desk stands for the studio speakers", 29),
    ("Add my usual coffee to the order", 30),
    ("Help me stay within my daily coffee target", 31),
    ("Remind me before the gym membership renews", 32),
    ("Plan a meeting around my training days", 33),
    ("Check my passport validity for the trip", 34),
    ("Plan the airport transfer for Lisbon", 35),
    ("Help me prepare for a boat journey", 36),
    ("Help a guest get the spare house key", 37),
    ("Draft an email to my accountant", 38),
    ("Schedule preparation for my tax deadline", 39),
    ("Plan the monthly client invoicing", 40),
    ("Fill in my blood type on this form", None),
    ("Connect a guest to my wifi network", None),
    ("Reschedule my dentist appointment", None),
    ("Order shoes in my size", None),
    ("Check in with the airline I booked", None),
]

CUES = {
    2: "gpu graphics training capacity", 3: "ram hardware capacity",
    4: "theme appearance editor", 5: "editor quick patch",
    8: "release approval production", 10: "tracker tickets",
    13: "dinner restaurant allergy", 26: "daw recording audio",
    28: "export sample rate", 34: "passport validity travel",
    36: "boat journey seasick", 39: "tax deadline preparation",
}


def _seed(d: Path, on: bool, size: int, seed: int, cues: bool) -> None:
    m.configure(d, {"scattershot": on, "fusion": "keyword_only"})
    for rid, fact in enumerate(FACTS, 1):
        m.remember(fact, cues=CUES.get(rid, "") if cues else "")
    assert m.count() == len(FACTS), "fixture IDs changed through deduplication"
    # This is a retrieval benchmark, so load unrelated people's records in one
    # transaction rather than timing ingestion. Maintain the derived token index.
    rng = random.Random(seed)
    conn = m._connect()
    try:
        for index in range(size - len(FACTS)):
            template = rng.choice(FACTS)
            template = re.sub(r"\b(?:Dave|Sarah|Priya|Biscuit|Marcus)\b", "Colleague", template)
            fact = f"Archived profile person{index:05d}: {template} (sample {rng.randrange(1000000)})"
            cur = conn.execute(
                "INSERT INTO memories (category, fact, created_at, folder) VALUES (?,?,?,?)",
                ("fact", fact, "2020-01-01", m.auto_folder(fact, "fact")),
            )
            m._index_tokens(conn, cur.lastrowid, fact)
        conn.commit()
    finally:
        conn.close()
    m._corpus_changed()
    assert m.count() == size


def evaluate(questions: list[tuple[str, int | None]], limit: int = 8,
             proactive: bool = False) -> dict:
    hit = top = answered = false_pos = lines_total = chars = 0
    probes = []
    for query, want in questions:
        out = m.prime(query, limit=limit) if proactive else m.recall(query, limit=limit)
        m.discard_turn()  # a benchmark label must never become learned evidence
        ids = [int(match) for match in re.findall(r"^#(\d+) ", out, re.MULTILINE)]
        chars += len(out)  # includes abstention messages, not just successful hits
        lines_total += len(ids)
        assert len(ids) <= limit, "result capacity exceeded"
        if proactive:
            assert len(out) <= 2000, "prime character budget exceeded"
        if want is None:
            false_pos += bool(ids)
        else:
            answered += bool(ids)
            hit += want in ids
            top += bool(ids) and ids[0] == want
        probes.append({"query": query, "expected": want, "returned": ids, "chars": len(out)})
    return {"n": sum(w is not None for _, w in questions), "answered": answered,
            "hit": hit, "top": top, "false_pos": false_pos,
            "controls": sum(w is None for _, w in questions),
            "lines": lines_total, "chars": chars, "probes": probes}


def _arm(d: Path, on: bool, warm: bool) -> dict:
    _seed(d, on, len(FACTS), 0, False)
    if warm:
        # A modest, realistic amount of prior successful use: for a THIRD of
        # the answerable questions, the turn found the fact some other way
        # (the model rephrased, or it was in the prompt digest) and worked.
        for n, (q, want) in enumerate(QUESTIONS):
            if want is None or n % 3:
                continue
            m.recall(f"{q} {FACTS[want - 1]}", limit=1)
            m.reinforce("ok")
    return evaluate(QUESTIONS)


def arm(on: bool, warm: bool = False) -> dict:
    with tempfile.TemporaryDirectory(prefix="vector-recall-") as directory:
        return _arm(Path(directory), on, warm)


def matrix(sizes: list[int], seeds: list[int]) -> dict:
    results = []
    # Match the eight-result capacity before attributing gains to a mechanism.
    # The larger baseline measures whether simply returning more does as well.
    variants = [("plain", False, False, 4), ("plain", False, False, 8),
                ("plain", False, False, 12), ("scatter", True, False, 8),
                ("cues", False, True, 8), ("prime", False, True, 3)]
    for size in sizes:
        for seed in seeds:
            for name, scatter, cues, limit in variants:
                with tempfile.TemporaryDirectory(prefix="vector-matrix-") as directory:
                    _seed(Path(directory), scatter, size, seed, cues)
                    for style, questions in (("explicit", QUESTIONS), ("implicit", IMPLICIT_QUESTIONS)):
                        result = evaluate(questions, limit, proactive=name == "prime")
                        result.update(size=size, seed=seed, arm=name, limit=limit, style=style)
                        results.append(result)
                        print(f"{size:5} seed={seed} {name:7} k={limit:2} {style:8} "
                              f"hit={result['hit']:2}/{result['n']} top={result['top']:2} "
                              f"control_returns={result['false_pos']}/{result['controls']} "
                              f"chars={result['chars']}", flush=True)
    return {"schema_version": 1, "kind": "synthetic retrieval capacity matrix",
            "caveat": "Hand-authored facts, queries and hints; no model, latency or answer-quality claim.",
            "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--sizes", nargs="+", type=int, default=[40, 400, 4000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--json", type=Path, help="save per-query results")
    args = parser.parse_args(argv)
    if any(size < len(FACTS) for size in args.sizes):
        parser.error(f"sizes must be at least {len(FACTS)}")
    if args.matrix:
        result = matrix(args.sizes, args.seeds)
    else:
        result = {name: arm(on, warm) for name, on, warm in
                  (("OFF", False, False), ("ON", True, False), ("ON+warm", True, True))}
        print(f"{len(FACTS)} invented facts; 40 answerable questions; 5 controls")
        print(f"{'arm':10} {'hit':>6} {'top-1':>6} {'controls':>9} {'lines':>7} {'chars':>9}")
        for name, values in result.items():
            print(f"{name:10} {values['hit']:>3}/40 {values['top']:>3}/40 "
                  f"{values['false_pos']:>7}/5 {values['lines']:>7} {values['chars']:>9}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
