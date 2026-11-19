#!/usr/bin/env python3
"""Ranks phrases by how much more often model answers use them than human ones.

    python3 script/mine.py [--kind phrase|opener|closer] [--min 0.01] [--top 80]

Counts are of documents, not occurrences, on corpus/train only: a phrase one
answer repeats ten times is one answer's habit. A candidate has to appear in
at least --min of model documents, and is ranked by the ratio of the two
rates (with one document added to each side, so a phrase no human used does
not divide by zero).

  phrase   any run of one to four words
  opener   the first one to four words of a sentence
  closer   the first one to four words of a document's last sentence

The output is a list to read, not a rule: what goes in the style is chosen by
hand from it and then measured on corpus/test with bench.py.
"""
import collections
import pathlib
import re
import sys

args = sys.argv[1:]
opt = {"--kind": "phrase", "--min": "0.01", "--top": "80"}
while args:
    opt[args[0]] = args[1]
    args = args[2:]
kind, floor, top = opt["--kind"], float(opt["--min"]), int(opt["--top"])

root = pathlib.Path(__file__).resolve().parent.parent / "corpus" / "train"
WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
SENT = re.compile(r'(?<=[.!?])["\')\]]*\s+')


def grams(words, limit=None):
    out = set()
    for n in range(1, 5):
        stop = len(words) - n + 1 if limit is None else min(1, len(words) - n + 1)
        for i in range(max(0, stop)):
            out.add(" ".join(words[i:i + n]))
    return out


def features(text):
    if kind == "phrase":
        return grams(WORD.findall(text.lower()))
    sentences = [s for s in SENT.split(text.replace("\n", " ")) if WORD.search(s.lower())]
    if kind == "closer":
        sentences = sentences[-1:]
    out = set()
    for s in sentences:
        out |= grams(WORD.findall(s.lower()), limit=1)
    return out


df = {}
n = {}
for label in ("human", "model"):
    counter = collections.Counter()
    files = list((root / label).glob("*.md"))
    n[label] = len(files)
    for path in files:
        counter.update(features(path.read_text(encoding="utf-8")))
    df[label] = counter

rows = []
for gram, m in df["model"].items():
    if m / n["model"] < floor:
        continue
    h = df["human"].get(gram, 0)
    ratio = ((m + 1) / n["model"]) / ((h + 1) / n["human"])
    rows.append((ratio, gram, m / n["model"], h / n["human"]))

rows.sort(reverse=True)
print(f"{kind}s in at least {floor:.1%} of {n['model']} model documents ({n['human']} human)")
print(f"{'ratio':>7} {'model':>7} {'human':>7}  {kind}")
for ratio, gram, m, h in rows[:top]:
    print(f"{ratio:7.1f} {m:7.1%} {h:7.2%}  {gram}")
