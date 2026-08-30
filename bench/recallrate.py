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
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rangerkit import memory as m

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
    "The house alarm code is 8823",
    "The garage door code is 4417",
    "Bin collection is Thursday mornings",
    "The boiler was serviced in October and is due again next October",
    "Dave's car is a 2018 Volvo V60 with 74,000 miles on it",
    "The Volvo needs its timing belt done before 100,000 miles",
    "Dave's mortgage is with Nationwide on a five year fix ending in 2027",
    "Dave plays bass and records in Reaper with a Scarlett interface",
    "The bass is a Fender Precision from 1996",
    "Dave mixes with Ozone and prints stems at 48 kHz",
    "The studio monitors are Yamaha HS8s on isolation pads",
    "Dave's coffee order is a flat white with oat milk",
    "Dave is trying to cut down to two coffees a day",
    "The gym membership is at PureGym and renews in January",
    "Dave swims on Mondays and lifts on Wednesdays and Fridays",
    "Dave's passport expires in August 2028",
    "The flight to Lisbon is on the 3rd of June from Gatwick",
    "Dave gets motion sick on boats and takes tablets beforehand",
    "The spare house key is with the neighbour at number 14",
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


def arm(on, warm=False):
    d = Path(tempfile.mkdtemp())
    m.configure(d, {"backend": "sqlite", "scattershot": on})
    for f in FACTS:
        m.remember(f, "fact")
    if warm:
        # A modest, realistic amount of prior successful use: for a THIRD of
        # the answerable questions, the turn found the fact some other way
        # (the model rephrased, or it was in the prompt digest) and worked.
        for n, (q, want) in enumerate(QUESTIONS):
            if want is None or n % 3:
                continue
            m.recall(f"{q} {FACTS[want - 1]}", limit=1)
            m.reinforce("ok")
    hit = top = answered = false_pos = 0
    lines_total = 0
    ans = [(q, w) for q, w in QUESTIONS if w]
    for q, want in QUESTIONS:
        out = m.recall(q, limit=8)
        m.reinforce("stopped")
        got = "No memories matching" not in out
        rows = out.splitlines() if got else []
        lines_total += len(rows)
        if want is None:
            false_pos += 1 if got else 0
            continue
        answered += 1 if got else 0
        if any(line.startswith(f"#{want} ") for line in rows):
            hit += 1
        if rows and rows[0].startswith(f"#{want} "):
            top += 1
    n = len(ans)
    return {"n": n, "answered": answered, "hit": hit, "top": top,
            "false_pos": false_pos, "lines": lines_total}


off = arm(False)
on = arm(True)
warm = arm(True, warm=True)
n, ctrl = off["n"], len(QUESTIONS) - off["n"]
print(f"{len(FACTS)} facts · {n} answerable questions · {ctrl} unanswerable controls\n")
hdr = f"{'':34}{'OFF':>8}{'ON':>8}{'ON+warm':>10}"
print(hdr); print("-" * len(hdr))
def row(label, k, d=n):
    print(f"{label:34}{off[k]:>4}/{d:<3}{on[k]:>4}/{d:<3}{warm[k]:>6}/{d:<3}")
row("returned anything at all", "answered")
row("correct fact somewhere in results", "hit")
row("correct fact ranked FIRST", "top")
row("answered an unanswerable question", "false_pos", ctrl)
print(f"{'total result lines (prompt cost)':34}{off['lines']:>8}{on['lines']:>8}{warm['lines']:>10}")
for lab, a in (("ON", on), ("ON+warm", warm)):
    dh = (a["hit"] - off["hit"]) / n * 100
    dt = (a["top"] - off["top"]) / n * 100
    print(f"\n{lab}: hit-rate {off['hit']/n*100:.0f}% -> {a['hit']/n*100:.0f}% "
          f"({dh:+.0f} pts) · top-1 {off['top']/n*100:.0f}% -> {a['top']/n*100:.0f}% ({dt:+.0f} pts)")
