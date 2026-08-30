"""Stage 4 -- structure. Never pick a serialization; measure it.

The rule this module exists to enforce: encode the payload in every candidate
format, count each one with the same budgeter that prices the request, and ship
the winner. Choosing a format because it "looks compact" is how projects spend
a month on a 20% win they never verified and sometimes on a loss.

The win, where there is one, comes from declaring field names ONCE and letting
position carry the rest -- so it is confined to UNIFORM data. On uniform rows,
tabular beats compact JSON by roughly a third once there are enough rows to
amortise the header. On nested or irregular structures, compact JSON usually
wins outright, and this picks it without being told to.

There is also a prompt tax: any format the model has to be told about costs
the explanation. The fold pays that once, in the cached prefix,
where it bills at a tenth of face value -- which is what makes a custom
encoding viable at all at this scale.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Callable

# A format is only worth switching to if it beats compact JSON by more than
# noise. Below this the encoding difference is not worth the model having to
# think about a second shape.
MIN_GAIN = 0.08


def _compact_json(rows: list[dict[str, Any]], cols: list[str]) -> str:
    return json.dumps(rows, separators=(",", ":"), default=str)


def _tabular(rows: list[dict[str, Any]], cols: list[str]) -> str:
    """Header once, then positional values. The classic uniform-data win."""
    out = [" | ".join(cols)]
    for row in rows:
        out.append(" | ".join(_carry(row.get(c), "|", "\n", "\r") for c in cols))
    return "\n".join(out)


def _tsv(rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = ["\t".join(cols)]
    for row in rows:
        out.append("\t".join(_carry(row.get(c), "\t", "\n", "\r") for c in cols))
    return "\n".join(out)


def _csv(rows: list[dict[str, Any]], cols: list[str]) -> str:
    # The writer quotes commas and quotation marks properly, so only a newline
    # is a problem -- it would be quoted into a field spanning two lines, and a
    # digest labelled "first 8 rows" that occupies eleven of them misleads the
    # reader about the shape as surely as a mangled value misleads about the
    # contents.
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for row in rows:
        writer.writerow([_carry(row.get(c), "\n", "\r") for c in cols])
    return buf.getvalue().rstrip("\n")


def _python_literal(rows: list[dict[str, Any]], cols: list[str]) -> str:
    """A language the model has seen trillions of tokens of.

    Included because in-distribution notation preserves reasoning; an invented
    symbol scheme costs the same tokens AND needs a legend AND degrades the
    model's handling of it. If Python literals win on size, they win outright.
    """
    # The trailing comma on a one-column table is load-bearing, not style.
    # `('x')` is not a tuple in Python -- the parentheses are grouping, so the
    # row IS the string. A header declaring one column above rows that are bare
    # strings reads back as one column per CHARACTER, and on a numeric column it
    # does not read back at all: `(8080)[0]` raises. This is the exact failure
    # `_carry` exists to prevent one line below -- a digest quietly disagreeing
    # with the body, in the region of the prompt the model is told to trust
    # most -- and it fires precisely where this format WINS, because a single
    # narrow column is where declaring the name once pays best.
    tail = "," if len(cols) == 1 else ""
    body = ",".join(
        "(" + ",".join(repr(row.get(c)) for c in cols) + tail + ")" for row in rows
    )
    return f"cols={cols!r}\nrows=[{body}]"


FORMATS: dict[str, Callable[[list[dict[str, Any]], list[str]], str]] = {
    "compact_json": _compact_json,
    "tabular": _tabular,
    "tsv": _tsv,
    "csv": _csv,
    "python_literal": _python_literal,
}


class Unrepresentable(ValueError):
    """This format cannot carry this value without changing it.

    A ValueError so `choose` skips the format through the path it already has
    for one that fails to encode. Declining is the correct answer: this module
    measures formats and ships the winner, so a format that cannot represent
    the data is simply not a candidate for it.
    """


def _cell(value: Any) -> str:
    """The value as text, unaltered."""
    return "" if value is None else str(value)


def _carry(value: Any, *cannot: str) -> str:
    """The value, or a refusal if this format would have to change it.

    `_cell` used to substitute -- newline to space, `|` to `/`, tab to space --
    for every format at once, including the two that never needed it (csv
    quotes, python_literal reprs). So a table holding `p|q` was shown to the
    model as `p/q`, silently, in the region of the prompt it trusts most, with
    nothing anywhere reporting a change. A digest may summarise the body; it may
    not quietly disagree with it.
    """
    text = _cell(value)
    if any(ch in text for ch in cannot):
        raise Unrepresentable(f"value contains {cannot[0]!r}")
    return text


def uniform(rows: list[Any]) -> list[str] | None:
    """Column names if these rows are a uniform table, else None.

    "Uniform" is deliberately strict: every element a flat dict, sharing a key
    set, with no nested values. Anything looser and the header stops paying
    for itself and compact JSON is the right answer anyway.
    """
    if not isinstance(rows, list) or len(rows) < 2:
        return None
    first = rows[0]
    if not isinstance(first, dict) or not first:
        return None
    cols = list(first.keys())
    keyset = set(cols)
    for row in rows:
        if not isinstance(row, dict) or set(row.keys()) != keyset:
            return None
        if any(isinstance(v, (dict, list)) for v in row.values()):
            return None
    return cols


def choose(
    rows: list[Any],
    counter: Callable[[str], int],
    allowed: list[str] | None = None,
) -> tuple[str, str, int, int]:
    """Encode every candidate, count each, return the cheapest.

    Returns (format_name, encoded_text, tokens, tokens_saved_vs_json). The
    baseline is always compact JSON, because that is what an unoptimised
    implementation would have sent -- measuring against a deliberately verbose
    baseline is how this class of project reports wins it did not earn.
    """
    cols = uniform(rows)
    baseline = json.dumps(rows, separators=(",", ":"), default=str)
    base_tokens = counter(baseline)
    if cols is None:
        return "compact_json", baseline, base_tokens, 0

    names = [n for n in (allowed or list(FORMATS)) if n in FORMATS]
    if "compact_json" not in names:
        names.append("compact_json")

    best_name, best_text, best_tokens = "compact_json", baseline, base_tokens
    for name in names:
        try:
            text = FORMATS[name](rows, cols)
        except (TypeError, ValueError):
            continue
        tokens = counter(text)
        if tokens < best_tokens:
            best_name, best_text, best_tokens = name, text, tokens

    # Gate on a real margin, not on being one token shorter.
    if best_name != "compact_json" and base_tokens:
        if (base_tokens - best_tokens) / base_tokens < MIN_GAIN:
            return "compact_json", baseline, base_tokens, 0
    return best_name, best_text, best_tokens, max(0, base_tokens - best_tokens)


def parse_rows(text: str) -> list[dict[str, Any]] | None:
    """Recover uniform rows from a tool result that happens to be JSON.

    Best-effort and quiet: most agent tools return prose or line-oriented
    text, and this only fires on the ones that genuinely return a table.
    """
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    return parsed if uniform(parsed) else None


# ------------------------------------------------------------------ delimited

# Tried in this order, and the order is the tie-break: tab and pipe are
# unambiguous in ordinary text, a semicolon is nearly so, and a comma is both
# the common case and the one most likely to appear in prose by accident.
_DELIMITERS = ("\t", "|", ";", ",")

# How many lines decide the question. Enough that a short run of
# coincidentally-similar prose cannot carry the vote, bounded so the check
# stays cheap on a 60,000-line body that is not a table at all.
_SNIFF_LINES = 200

# Below three lines there is no shape to infer -- a header and one row is
# indistinguishable from two lines that happen to share a comma.
_MIN_LINES = 3


def _numeric(cell: str) -> bool:
    try:
        float(cell.strip())
    except (TypeError, ValueError):
        return False
    return True


# `read_file` renders a file as `{i:>5}  {line}` and `fold_grep` as `{n}: {line}`.
# Neither prefix is data, and both are the shape a CSV arrives in through the
# tool a model actually uses to read one.
_GUTTER = re.compile(r"^\s{0,8}(\d+)(?:  |: )")


def ungutter(lines: list[str]) -> list[str] | None:
    """The lines without a machine-added line-number column, or None.

    Without this the whole delimited path is theatre: nobody hands the agent a
    raw CSV, they hand it `read_file`'s view of one, and every line of that
    begins with a right-aligned line number and two spaces. The sniff below
    would see one ragged column and give up.

    The discriminator is that the numbers COUNT: a gutter is consecutive by
    construction, and a data column of consecutive integers two spaces from
    the next field is not a format anyone writes. Requiring the run rather
    than just the shape is what keeps this from eating the first column of a
    space-aligned table.

    Lines without a gutter are dropped rather than kept, because the one that
    occurs in practice is `read_file`'s banner -- a path and a range, which is
    not a row of anything.
    """
    seen: list[tuple[int, int]] = []      # (line number, prefix length)
    for line in lines:
        m = _GUTTER.match(line)
        if m is not None:
            seen.append((int(m.group(1)), m.end()))
    if len(seen) < _MIN_LINES or len(seen) < len(lines) - 1:
        return None
    if any(b[0] != a[0] + 1 for a, b in zip(seen, seen[1:])):
        return None
    return [line[_GUTTER.match(line).end():] for line in lines if _GUTTER.match(line)]


def view(text: str) -> list[str]:
    """The rows of a body as data, with a line-number gutter removed if present.

    The one place that decision is made, so the digest's column names and
    `gate.stats`'s arithmetic can never be reading two different tables.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    return ungutter(lines) or lines


def delimited(text: str) -> dict[str, Any] | None:
    """Column names and separator if this body is delimited text, else None.

    `parse_rows` above only fires on a JSON array -- the shape a tool returns
    when it was designed to return a table. The shape the USER's data actually
    arrives in is a CSV, and for one of those the digest previously said
    nothing beyond a line count. That left a question about a column with no
    cheaper answer than paging the whole body back through `fold_peek`, which
    costs strictly more than never folding it: 18 round trips and 141,000
    characters for a 138,000-character file, and the conversation re-sent at
    every one of them.

    Deliberately strict, for the same reason `uniform` is. This result goes
    into the cached prefix, the region the model trusts most, so a body that is
    only PROBABLY a table is worse than one that is not claimed to be.

    Returns {"delimiter", "columns", "has_header"} or None. Note what it does
    NOT return: a row count. The consistency check reads a sample, and a count
    inferred from a sample and stated as fact is the kind of number this
    package exists not to produce. The digest already carries an exact line
    count, and `fold_stats(op="count")` reads every row for the exact one.
    """
    lines = view(text)
    if len(lines) < _MIN_LINES:
        return None
    sample = lines[:_SNIFF_LINES]

    best: dict[str, Any] | None = None
    for delim in _DELIMITERS:
        if delim not in sample[0]:
            continue
        try:
            rows = list(csv.reader(sample, delimiter=delim))
        except (csv.Error, ValueError):
            continue
        if not rows or len(rows[0]) < 2:
            continue
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            continue
        # Widest consistent parse wins: a tab-separated body whose cells
        # contain commas is a 5-column TSV, not a ragged CSV, and only the
        # column count can tell the two apart. Measured, not assumed -- the
        # same rule as `choose` above.
        if best is None or width > best["width"]:
            best = {"delimiter": delim, "width": width, "rows": rows}
    if best is None:
        return None

    head = [cell.strip() for cell in best["rows"][0]]
    # A header is every cell non-empty, none of them a number, and no name
    # repeated. A first row of numbers is data, and naming a column `41.10`
    # would invite the model to ask for a column that means nothing.
    has_header = (
        all(head)
        and not any(_numeric(cell) for cell in head)
        and len(set(head)) == len(head)
    )
    columns = head if has_header else [f"c{i + 1}" for i in range(best["width"])]
    return {"delimiter": best["delimiter"], "columns": columns, "has_header": has_header}


DELIMITER_NAMES = {"\t": "tab", "|": "pipe", ";": "semicolon", ",": "comma"}


# ------------------------------------------------------------------ outliers

# What varies between two otherwise identical log lines: a counter, a
# timestamp, a duration, a request id. Collapsing those is what turns two
# thousand distinct strings into one shape and five exceptions.
_VARIABLE = re.compile(r"\d+|\b[0-9a-fA-F]{8,}\b")

# Padding is not shape. `read_file` renders a body as `{i:>5}  {line}`, so the
# width of the gutter changes at line 10, at 100 and at 1000 -- which split one
# build log into four "shapes" of 997, 899, 90 and 9 lines and left the largest
# three short of a majority of a 2,000-line file. Every aligned listing on a
# machine does the same thing: `ls -l`, `ps`, `df`, any column padded to fit its
# widest value. Collapsing runs of whitespace costs one more pass and removes
# the whole class.
_PADDING = re.compile(r"\s+")

# Signatures compare PREFIXES. Two lines whose first 160 characters match are
# the same shape for this purpose, and the clip keeps the pass proportional to
# the line count rather than to the width of the widest line.
_SIG_CHARS = 160

# Under this many lines a head-and-tail view already shows most of the body,
# so there is nothing a claim about its shape could add.
_PATTERN_MIN_LINES = 24

# How deep to go before giving up on a body that is not patterned. Prose and
# source code produce a new signature on nearly every line and show it
# immediately, so the scan stops rather than paying for the whole body to
# learn what the first two thousand lines already said.
_PROBE_LINES = 2000
_PROBE_VARIETY = 0.25

# Past this the pass costs more than a fold should -- roughly a third of a
# second at 200,000 lines, and linear after that. Above the ceiling this
# returns None rather than analysing a prefix: a claim that names "the lines
# that differ" after reading part of the body is the same silent-partial the
# rest of this package refuses to commit, and a wrong list of exceptions is
# worse than no list, because the model would stop looking.
_PATTERN_MAX_LINES = 250_000


def signature(line: str) -> str:
    """The shape of a line: its variable parts collapsed, its padding normalised.

    Public because the digest groups the outliers it lists by shape, and a
    caller that has to reach into a private helper to do that is one rename
    away from silently listing the same WARN six times.
    """
    return _PADDING.sub(" ", _VARIABLE.sub("#", line[:_SIG_CHARS])).strip()


def outliers(lines: list[str], rare_at_most: int = 40) -> dict[str, Any] | None:
    """The lines that a head-and-tail sample of a patterned body hides.

    A build log, a test run, an access log, a trace: thousands of lines of one
    routine shape with a handful of exceptions, and the exceptions are the only
    reason anyone opened the file. For a body like that, showing the first
    eight lines and the last four is not merely incomplete -- it is
    MISLEADING, because every line it shows is routine and the model correctly
    concludes from a representative-looking sample that the whole body is.
    Measured on Ranger's own model-driven bench at small scale: three of the
    four hard-band
    failures in the folded arm answered "all steps passed, no failures
    recorded" against a log whose line 1,237 reads FATAL, and did so without
    reading anything back, because nothing they were shown suggested there was
    anything to read.

    The body is already in hand and on the user's machine, so the cost of
    knowing better is one pass over the lines and none of the model's context.

    Returns None -- say nothing rather than something shaky -- unless the body
    is mostly repetition with few enough exceptions to list. Prose and source
    bail out early, on variety; a uniform table has no exceptions to name; a
    body whose lines merely vary has too many. Where it does fire, `rare` is
    every line whose shape is rare, in line order, as (line number, how many
    lines share its shape, the line).
    """
    total = len(lines)
    if total < _PATTERN_MIN_LINES or total > _PATTERN_MAX_LINES:
        return None

    # Signatures are interned to small ints on the way past, so the second pass
    # below can ask "how common is this line's shape" without running the
    # regex again -- the alternative being either double the CPU or a list of
    # 250,000 clipped strings held in memory to avoid it.
    ids: list[int] = []
    index: dict[str, int] = {}
    counts: list[int] = []
    for seen, line in enumerate(lines, 1):
        sig = signature(line)
        at = index.get(sig)
        if at is None:
            at = index[sig] = len(counts)
            counts.append(0)
        counts[at] += 1
        ids.append(at)
        if seen == _PROBE_LINES and len(counts) > _PROBE_LINES * _PROBE_VARIETY:
            return None

    # Rare against the body's own scale: one line in a hundred, at least one.
    # A fixed threshold would call every shape in a hundred-line file rare and
    # no shape in a million-line file rare.
    rare_at = max(1, total // 100)
    rare = [(n, counts[at], lines[n - 1])
            for n, at in enumerate(ids, 1) if counts[at] <= rare_at]
    routine = total - len(rare)
    # Exceptions have to be exceptional: at most one line in fifty, and never
    # more than the caller will show. Past that this is not a body with a few
    # anomalies, it is a body whose lines simply vary -- and the difference
    # matters because the failure is not symmetric. A 400-row ledger whose
    # merchant column holds "cafe nord" beside "greengrocer" has a different
    # signature on those rows and nothing whatever is wrong with them; naming
    # six ordinary transactions as unlike the rest is worse than saying
    # nothing at all, because the model has no way to tell that it is noise
    # and every reason to think the digest knows something.
    if not rare or len(rare) > min(rare_at_most, max(2, total // 50)) \
            or routine * 2 < total:
        # Nothing stands out, too much does, or there is no routine mass for a
        # line to stand out FROM. In all three the ends mislead no one, and the
        # middle case is a body to search rather than to characterise.
        #
        # Note what is deliberately not required: that ONE shape dominate. A
        # log of interleaved GETs and POSTs has two routine shapes and neither
        # is a majority, yet its ten stack traces are exactly as findable and
        # exactly as hidden by a head-and-tail view.
        return None
    return {"total": total, "shapes": len(counts), "routine": routine, "rare": rare}
