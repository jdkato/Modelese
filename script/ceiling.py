#!/usr/bin/env python3
"""How far weighted phrases can go: a ceiling for the style, before writing rules.

    python3 script/ceiling.py [--train train] [--test test]

A split is a directory under corpus/, or several joined with a comma. `raid:0`
and `raid:1` are the two halves of one, by a hash of each file's name, for a
corpus that has no split of its own.

Every word run of one to three words that enough training documents use gets
a weight: the log of how much more often model documents use it than human
ones. A document's score is the sum of the weights of the phrases in it.
That is what a style with a rule per phrase and a weight per rule computes,
so its result on held-out text bounds what such a style can reach.

It is run with the k strongest phrases, to see how many rules the result
needs, and with and without phrases that point the other way: the ones
people use and models avoid.
"""
import collections
import hashlib
import math
import pathlib
import re
import sys

args = sys.argv[1:]
opt = {"--train": "train", "--test": "test"}
while args:
    opt[args[0]] = args[1]
    args = args[2:]
test = opt["--test"]
root = pathlib.Path(__file__).resolve().parent.parent / "corpus"
WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?|[!?;:()\"]|\.\.\.")


def grams(text):
    words = WORD.findall(text.lower())
    out = set()
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            out.add(" ".join(words[i:i + n]))
    return out


def load(splits, label):
    docs = []
    for spec in splits.split(","):
        name, _, half = spec.partition(":")
        for p in sorted((root / name / label).glob("*.md")):
            if half and int(hashlib.sha1(p.name.encode()).hexdigest()[:2], 16) % 2 != int(half):
                continue
            docs.append(grams(p.read_text(encoding="utf-8")))
    return docs


train = {label: load(opt["--train"], label) for label in ("human", "model")}
held = {label: load(test, label) for label in ("human", "model")}
n = {label: len(docs) for label, docs in train.items()}

df = {label: collections.Counter(g for doc in docs for g in doc) for label, docs in train.items()}
weights = {}
for gram in set(df["human"]) | set(df["model"]):
    h, m = df["human"].get(gram, 0), df["model"].get(gram, 0)
    if h + m < 25:
        continue
    weights[gram] = math.log((m + 1) / (n["model"] + 2)) - math.log((h + 1) / (n["human"] + 2))


def evaluate(chosen):
    score = lambda doc: sum(chosen[g] for g in doc if g in chosen)
    h = sorted(score(d) for d in held["human"])
    m = [score(d) for d in held["model"]]
    out = []
    for fpr in (0.01, 0.05):
        threshold = h[min(len(h) - 1, int((1 - fpr) * len(h)))]
        out.append(sum(1 for s in m if s > threshold) / len(m))
    pairs = sorted([(s, 0) for s in h] + [(s, 1) for s in m])
    total, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        total += (i + j + 1) / 2 * sum(1 for k in range(i, j) if pairs[k][1])
        i = j
    return (total - len(m) * (len(m) + 1) / 2) / (len(h) * len(m)), out[0], out[1]


# Strength is weight times reach: a phrase has to be both lopsided and common.
strength = lambda g: abs(weights[g]) * math.log(df["human"].get(g, 0) + df["model"].get(g, 0))
model_side = sorted((g for g in weights if weights[g] > 0), key=strength, reverse=True)
human_side = sorted((g for g in weights if weights[g] < 0), key=strength, reverse=True)

print(f"trained on {opt['--train']}: {n['human']} human, {n['model']} model; held out ({test}): {len(held['human'])} human, {len(held['model'])} model")
print(f"{len(model_side)} phrases lean model, {len(human_side)} lean human")
print(f"{'phrases':28} {'AUC':>5} {'caught @1% FP':>14} {'caught @5% FP':>14}")
for k in (25, 100, 300, 1000, 5000):
    chosen = {g: weights[g] for g in model_side[:k]}
    print(f"{'model-side, top ' + str(k):28} {'%.3f' % evaluate(chosen)[0]:>5} {evaluate(chosen)[1]:14.1%} {evaluate(chosen)[2]:14.1%}")
for k in (100, 300, 1000, 5000):
    chosen = {g: weights[g] for g in model_side[:k] + human_side[:k]}
    r = evaluate(chosen)
    print(f"{'both sides, top ' + str(k) + ' each':28} {'%.3f' % r[0]:>5} {r[1]:14.1%} {r[2]:14.1%}")
print("\nstrongest human-side:", ", ".join(human_side[:40]))
