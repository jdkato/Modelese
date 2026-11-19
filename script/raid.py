#!/usr/bin/env python3
"""Samples RAID into corpus/raid/: human and model text across eight domains.

    python3 script/raid.py

RAID (Dugan et al., ACL 2024; MIT) is the benchmark the published detector
numbers are on: eleven generators, eight domains, and a leaderboard scored as
the share of model text caught with the false-positive rate held at 5%.

Its training file is 11.8 GB, so this reads pieces of it. The file is in
order by domain, and each domain opens with its human documents, so the
boundaries can be found by bisection and each domain sampled on its own.
The generated rows after them come in runs by generator and attack, in no
order worth relying on, so those are sampled at even intervals and only the
attack-free ones kept:

    corpus/raid/human/<domain>-<id>.md
    corpus/raid/model/<domain>-<generator>-<id>.md

Only attack-free rows are taken, which is what the leaderboard's headline
column measures.
"""
import csv
import io
import pathlib
import re
import shutil
import urllib.request

URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv"
COLUMNS = ["id", "adv_source_id", "source_id", "model", "decoding", "repetition_penalty",
           "attack", "domain", "title", "prompt", "generation"]
# A row starts on a line that opens with its UUID.
ROW = re.compile(r"\n(?=[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12},)")

HUMAN_BYTES = 3_000_000   # read from the start of each domain
MODEL_RANGES = 48         # pieces spread over the rest of it; one row in twelve is attack-free
MODEL_BYTES = 700_000

csv.field_size_limit(10**9)
root = pathlib.Path(__file__).resolve().parent.parent / "corpus" / "raid"


def fetch(start, length):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{start + length - 1}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as res:
                return res.read().decode("utf-8", errors="replace")
        except OSError:
            if attempt == 3:
                raise
    return ""


def rows(start, length):
    """The complete rows inside a byte range."""
    pieces = ROW.split(fetch(start, length))
    # The first piece began mid-row unless the range began the file, and the
    # last was cut off.
    pieces = pieces[(0 if start == 0 else 1):-1]
    out = []
    for piece in pieces:
        try:
            row = next(csv.reader(io.StringIO(piece)))
        except (csv.Error, StopIteration):
            continue
        if len(row) == len(COLUMNS) and row[0] != "id":
            out.append(dict(zip(COLUMNS, row)))
    return out


def first_row(offset):
    for length in (400_000, 3_000_000):
        found = rows(offset, length)
        if found:
            return found[0]
    return None


def bisect(lo, hi, changed):
    """The first offset in [lo, hi) whose row satisfies changed, to 200 KB."""
    while hi - lo > 200_000:
        mid = (lo + hi) // 2
        row = first_row(mid)
        if row is None or changed(row):
            hi = mid
        else:
            lo = mid
    return hi


with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD")) as res:
    size = int(res.headers["Content-Length"])

shutil.rmtree(root, ignore_errors=True)
for label in ("human", "model"):
    (root / label).mkdir(parents=True)

start = 0
while start < size - 400_000:
    domain = first_row(start)["domain"]
    end = bisect(start, size, lambda r, d=domain: r["domain"] != d)

    seen = set()
    counts = {"human": 0, "model": 0}

    def keep(row):
        text = row["generation"].strip()
        if row["attack"] != "none" or row["domain"] != domain or row["id"] in seen or len(text.split()) < 50:
            return
        seen.add(row["id"])
        label = "human" if row["model"] == "human" else "model"
        name = f"{domain}-{row['id'][:12]}.md" if label == "human" else f"{domain}-{row['model']}-{row['id'][:12]}.md"
        text = re.sub(r"\s*\n\s*", "\n\n", text) + "\n"
        (root / label / name).write_text(text, encoding="utf-8")
        counts[label] += 1

    for row in rows(start, HUMAN_BYTES):
        if row["model"] == "human":
            keep(row)
    # The boundary is good to 200 KB; a domain's human rows start at it.
    span = end - start - HUMAN_BYTES
    for i in range(MODEL_RANGES):
        for row in rows(start + HUMAN_BYTES + span * i // MODEL_RANGES, MODEL_BYTES):
            if row["model"] != "human":
                keep(row)

    print(f"{domain:10} {(end - start) / 1e6:6.0f} MB  human {counts['human']:4}  model {counts['model']:4}")
    start = end
