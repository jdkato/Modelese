#!/usr/bin/env python3
"""Writes the score as data: dist/score.json, for a page to compute.

    MODELESE_WORDS=20000 MODELESE_PAIRS=50000 python3 script/export.py

Trains on the training halves of everything there is -- RAID, HC3, and the
GitHub registers -- without the tagger's features, since a page has no
tagger, and writes out what a scorer needs to reach the same number:

  features    each measured feature's mean, spread, and weight
  phrases     each phrase's weight
  words       log-probability of each word in the table
  pairs       log-probability of the second word of each pair, given the first
  thresholds  the score that 5% and 1% of held-out human text exceeds,
              overall and for each register

It also writes dist/cases.json: a few texts and the score each must get, so
a scorer in another language can be checked against this one.
"""
import json
import math
import pathlib
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model  # noqa: E402

GROUPS = ["rhythm", "words", "function", "marks", "surprise", "phrases"]
REGISTERS = ("issues", "prs", "commits")
root = pathlib.Path(__file__).resolve().parent.parent

train = model.load("raid:0") + model.load("train")
held = {"general": model.load("raid:1") + model.load("test")}
for name in REGISTERS:
    train += model.load(name + ":0")
    held[name] = model.load(name + ":1")

score, _ = model.fit(train, GROUPS)
params = score.params


def cut(docs, fpr):
    human = sorted(score(d, g) for label, _, d, g in docs if label == "human")
    return human[min(len(human) - 1, int((1 - fpr) * len(human)))]


thresholds = {}
for name, docs in held.items():
    t5, t1 = cut(docs, 0.05), cut(docs, 0.01)
    generated = [score(d, g) for label, _, d, g in docs if label == "model"]
    thresholds[name] = {"fp5": round(t5, 4), "fp1": round(t1, 4)}
    if generated:
        thresholds[name]["caught5"] = round(sum(1 for s in generated if s > t5) / len(generated), 3)
        thresholds[name]["caught1"] = round(sum(1 for s in generated if s > t1) / len(generated), 3)
    print(name, thresholds[name])

out = {
    "bias": round(params["bias"], 5),
    "features": [{"name": f["name"], "mean": round(f["mean"], 6), "sd": round(f["sd"], 6), "weight": round(f["weight"], 5)} for f in params["features"]],
    "phrases": {p: round(w, 4) for p, w in params["phrases"].items()},
    "floor": round(model.FLOOR, 4),
    "words": {w: round(math.log(c / model.total), 3) for w, c in model.uni.items()},
    # null marks a pair that is known but whose first word is not in the table.
    "pairs": {k: (round(math.log(c / model.uni[k.split(" ")[0]]), 3) if k.split(" ")[0] in model.uni else None) for k, c in model.pair.items()},
    "thresholds": thresholds,
}
dist = root / "dist"
dist.mkdir(exist_ok=True)
(dist / "score.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))

# The repository's own prose and a few lines written for the purpose: text
# with nothing to license, covering plain ASCII, typographic marks, a short
# text, and one under the fifty-word window.
samples = [
    (root / "README.md").read_text(encoding="utf-8"),
    (root / "LICENSE").read_text(encoding="utf-8"),
    "It’s important to note that the cache — once warmed — isn’t cold. Overall, it's fast… mostly.\n\nHere are a few things to check: the TTL, the key, and the size.",
    "Short text. Two sentences.",
    "the cache is cold. nobody warmed it after the deploy, so the first request for each key goes to the database. warm it. a script in the repo does that in under a minute, and the deploy job can call it. after that, requests are fast. that is the whole fix, and it costs 3 lines of YAML (plus a 2nd job).",
]
cases = [{"text": text, "score": round(score(model.dense(text, []), model.grams(text)), 4)} for text in samples]
(dist / "cases.json").write_text(json.dumps(cases, ensure_ascii=False))
print(f"{len(out['features'])} features, {len(out['phrases'])} phrases, {len(out['words'])} words, {len(out['pairs'])} pairs; {(dist / 'score.json').stat().st_size / 1e6:.1f} MB")
