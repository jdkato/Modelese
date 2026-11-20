#!/usr/bin/env python3
"""Tests whether measured style, not just phrases, can tell model text apart.

    python3 script/model.py

Every feature here is something a Vale rule can compute: a rate, a length, a
ratio, a count of tag sequences from the same tagger `sequence` rules use, or
a word's frequency looked up in a table. A regularized linear model weighs
them, so a document's score is still a sum of named parts.

Groups of features, so each can be left out:

  rhythm    sentence lengths, and how much they vary
  words     word length, vocabulary spread, repetition
  function  how often each common function word is used
  marks     punctuation, capitals, digits, contractions
  surprise  how expected each word is, from web-scale word and word-pair counts
  tags      part-of-speech tags, tag pairs, and how sentences open
  phrases   the phrases most lopsided between the two sides in training

Needs corpus/ (corpus.py, raid.py), tags.jsonl in each directory
(script/tags), and corpus/ngrams/count_1w.txt and count_2w.txt.
"""
import collections
import hashlib
import json
import math
import os
import pathlib
import random
import re
import sys

root = pathlib.Path(__file__).resolve().parent.parent / "corpus"
WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
SENT = re.compile(r'(?<=[.!?])["\')\]]*\s+')
FUNCTION = (
    "the of and to a in is that it for as with was on be by this are or an at from not have has had but "
    "which they you we he she his her their its can will would could should may might must been being were "
    "do does did if then than so such there these those what when where who how why all any some each many "
    "more most other also only just very into over about after before between through during under while "
    "because however therefore although though since until i me my your our us them him no nor yet both "
    "either neither one two often always never still even well too much own same out up down off again here"
).split()
MARKS = list(",.;:!?()\"'-/") + ["—", "–", "…", "%", "$", "&"]


def stats(xs):
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# ---- word frequencies -------------------------------------------------------
# How much of each table to read. A page that has to download them wants
# them small; MODELESE_WORDS and MODELESE_PAIRS say how small.
UNI_N = int(os.environ.get("MODELESE_WORDS", "100000"))
PAIR_N = int(os.environ.get("MODELESE_PAIRS", "0"))
uni, total = {}, 0
for i, line in enumerate(open(root / "ngrams" / "count_1w.txt", encoding="utf-8")):
    if i >= UNI_N:
        break
    w, c = line.split("\t")
    uni[w] = int(c)
    total += int(c)
pair = {}
for line in open(root / "ngrams" / "count_2w.txt", encoding="utf-8"):
    w, c = line.rstrip("\n").split("\t")
    pair[w.lower()] = pair.get(w.lower(), 0) + int(c)
if PAIR_N:
    pair = dict(sorted(pair.items(), key=lambda kv: -kv[1])[:PAIR_N])
FLOOR = math.log(10 / total)


def logp(w):
    return math.log(uni[w] / total) if w in uni else FLOOR


# ---- features ---------------------------------------------------------------
def dense(text, tags):
    # How a text is laid out -- its line breaks, its paragraph sizes, its
    # Markdown -- says where it was collected, not how it was written, so
    # none of that is measured.
    text = " ".join(text.split())
    f = {}
    words = WORD.findall(text)
    lower = [w.lower() for w in words]
    n = max(len(words), 1)
    sents = [len(WORD.findall(s)) for s in SENT.split(text.replace("\n", " "))]
    sents = [s for s in sents if s]

    m, sd = stats(sents)
    f["rhythm:sentence_mean"] = m
    f["rhythm:sentence_cv"] = sd / m if m else 0
    f["rhythm:short"] = sum(1 for s in sents if s <= 6) / max(len(sents), 1)
    f["rhythm:long"] = sum(1 for s in sents if s >= 30) / max(len(sents), 1)
    f["rhythm:flat"] = sum(1 for a, b in zip(sents, sents[1:]) if abs(a - b) <= 3) / max(len(sents) - 1, 1)

    lm, lsd = stats([len(w) for w in words])
    f["words:length_mean"], f["words:length_sd"] = lm, lsd
    f["words:long"] = sum(1 for w in words if len(w) > 6) / n
    windows = [len(set(lower[i:i + 50])) / 50 for i in range(0, max(len(lower) - 50, 1), 25)] if len(lower) >= 50 else [len(set(lower)) / n]
    f["words:spread"] = sum(windows) / len(windows)
    counts = collections.Counter(lower)
    f["words:once"] = sum(1 for c in counts.values() if c == 1) / n
    f["words:top"] = max(counts.values(), default=0) / n
    pairs = collections.Counter(zip(lower, lower[1:]))
    f["words:repeated_pairs"] = sum(c for c in pairs.values() if c > 1) / max(len(lower) - 1, 1)
    triples = collections.Counter(zip(lower, lower[1:], lower[2:]))
    f["words:repeated_triples"] = sum(c for c in triples.values() if c > 1) / max(len(lower) - 2, 1)

    for w in FUNCTION:
        f["function:" + w] = counts.get(w, 0) / n

    chars = max(len(text), 1)
    for c in MARKS:
        f["marks:" + c] = text.count(c) / chars * 100
    f["marks:upper"] = sum(1 for c in text if c.isupper()) / chars
    f["marks:digit"] = sum(1 for c in text if c.isdigit()) / chars
    f["marks:contraction"] = sum(1 for w in words if "'" in w or "’" in w) / n
    f["marks:lower_start"] = sum(1 for s in SENT.split(text) if s[:1].islower()) / max(len(sents), 1)

    lp = sorted(logp(w) for w in lower)
    m, sd = stats(lp)
    f["surprise:mean"], f["surprise:sd"] = m, sd
    f["surprise:p10"] = lp[len(lp) // 10] if lp else 0
    f["surprise:p90"] = lp[len(lp) * 9 // 10] if lp else 0
    f["surprise:unknown"] = sum(1 for w in lower if w not in uni) / n
    cond = []
    for a, b in zip(lower, lower[1:]):
        c = pair.get(a + " " + b)
        cond.append(math.log(c / uni[a]) if c and a in uni else logp(b) + math.log(0.4))
    m, sd = stats(cond)
    f["surprise:pair_mean"], f["surprise:pair_sd"] = m, sd
    f["surprise:pair_known"] = sum(1 for a, b in zip(lower, lower[1:]) if a + " " + b in pair) / max(len(lower) - 1, 1)

    t = max(len(tags), 1)
    for tag, c in collections.Counter(tags).items():
        f["tags:" + tag] = c / t
    for (a, b), c in collections.Counter(zip(tags, tags[1:])).items():
        f[f"tags:{a} {b}"] = c / t
    opens = [b for a, b in zip(["."] + tags, tags) if a == "."]
    for tag, c in collections.Counter(opens).items():
        f["tags:^" + tag] = c / max(len(opens), 1)
    return f


def grams(text):
    words = re.findall(r"[a-z0-9]+(?:['’][a-z]+)?", text.lower())
    return {" ".join(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}


def load(spec):
    """[(label, generator, dense features, phrase set)] for a split, or half of one."""
    name, _, half = spec.partition(":")
    key = (name, half)
    if key in CACHE:
        return CACHE[key]
    docs = []
    for label in ("human", "model"):
        if not (root / name / label).is_dir():
            continue  # a register with no model side
        tags = {}
        # Without script/tags having run, a directory has no tag features.
        if (root / name / label / "tags.jsonl").exists():
            for line in open(root / name / label / "tags.jsonl", encoding="utf-8"):
                row = json.loads(line)
                tags[row["name"]] = row["tags"]
        for p in sorted((root / name / label).glob("*.md")):
            # A generated document splits with the human one it is paired to.
            pair, _, local = p.stem.partition("~")
            if half and int(hashlib.sha1((pair + ".md").encode()).hexdigest()[:2], 16) % 2 != int(half):
                continue
            text = p.read_text(encoding="utf-8")
            generator = "human" if label == "human" else local or re.sub(r"^[a-z_]+-(.*)-[0-9a-f]{8}-.*$", r"\1", p.name)
            if generator == p.name:
                generator = "llama"  # generate.py's first model, which adds no suffix
            docs.append((label, generator, dense(text, tags.get(p.name, [])), grams(text)))
    CACHE[key] = docs
    return docs


CACHE = {}


# ---- the model --------------------------------------------------------------
def fit(train, groups, epochs=8):
    """Returns score(doc) for a logistic model over the chosen feature groups."""
    use_phrases = "phrases" in groups
    names = sorted({k for _, _, d, _ in train for k in d if k.split(":")[0] in groups})
    # Keep a feature only if enough documents have it, so a rare tag pair
    # cannot become a fingerprint of one source.
    seen = collections.Counter(k for _, _, d, _ in train for k in d)
    names = [k for k in names if seen[k] >= len(train) * 0.02]
    mean, sd = {}, {}
    for k in names:
        mean[k], sd[k] = stats([d.get(k, 0.0) for _, _, d, _ in train])

    phrases = {}
    if use_phrases:
        n = collections.Counter(label for label, *_ in train)
        df = {"human": collections.Counter(), "model": collections.Counter()}
        for label, _, _, g in train:
            df[label].update(g)
        scored = []
        for gram in set(df["human"]) | set(df["model"]):
            h, m = df["human"].get(gram, 0), df["model"].get(gram, 0)
            if h + m >= 25:
                w = math.log((m + 1) / (n["model"] + 2)) - math.log((h + 1) / (n["human"] + 2))
                scored.append((abs(w) * math.log(h + m), gram))
        scored.sort(reverse=True)
        phrases = {gram: i for i, (_, gram) in enumerate(scored[:2000])}

    index = {k: i for i, k in enumerate(names)}
    base = len(names)

    def vector(d, g):
        xs = [(index[k], (v - mean[k]) / sd[k]) for k, v in d.items() if k in index and sd[k] > 0]
        xs += [(index[k], -mean[k] / sd[k]) for k in names if k not in d and sd[k] > 0]
        xs += [(base + phrases[p], 1.0) for p in g if p in phrases]
        return xs

    rows = [(1.0 if label == "model" else 0.0, vector(d, g)) for label, _, d, g in train]
    pos = sum(y for y, _ in rows)
    weight = {1.0: len(rows) / (2 * pos), 0.0: len(rows) / (2 * (len(rows) - pos))}
    w = [0.0] * (base + len(phrases))
    bias = 0.0
    rng = random.Random(3)
    step = 0
    for epoch in range(epochs):
        rng.shuffle(rows)
        for y, xs in rows:
            step += 1
            lr = 0.05 / (1 + step / 20000)
            z = bias + sum(w[i] * v for i, v in xs)
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            g = (p - y) * weight[y]
            bias -= lr * g
            for i, v in xs:
                w[i] -= lr * (g * v + 1e-4 * w[i])

    def score(d, g):
        return bias + sum(w[i] * v for i, v in vector(d, g))

    # Everything a scorer written elsewhere needs to give the same number.
    score.params = {
        "bias": bias,
        "features": [{"name": k, "mean": mean[k], "sd": sd[k], "weight": w[index[k]]} for k in names if sd[k] > 0],
        "phrases": {p: w[base + i] for p, i in phrases.items()},
    }
    top = sorted(((abs(w[i]), k, w[i]) for k, i in index.items()), reverse=True)[:12]
    return score, top


def evaluate(score, docs):
    h = sorted(score(d, g) for label, _, d, g in docs if label == "human")
    m = [score(d, g) for label, _, d, g in docs if label == "model"]
    out = []
    for fpr in (0.01, 0.05):
        threshold = h[min(len(h) - 1, int((1 - fpr) * len(h)))]
        out.append(sum(1 for s in m if s > threshold) / len(m))
    wins = 0.0
    hs = sorted(h)
    import bisect
    for s in m:
        wins += bisect.bisect_left(hs, s) + 0.5 * (bisect.bisect_right(hs, s) - bisect.bisect_left(hs, s))
    return wins / (len(h) * len(m)), out[0], out[1], len(h), len(m)


ALL = ["rhythm", "words", "function", "marks", "surprise", "tags", "phrases"]
METRICS = ALL[:-1]


def report(name, score, docs):
    auc, at1, at5, nh, nm = evaluate(score, docs)
    print(f"{name:54} {auc:5.3f} {at1:9.1%} {at5:9.1%}   ({nh} human, {nm} model)", flush=True)


if __name__ == "__main__":
    print(f"{'experiment':54} {'AUC':>5} {'@1% FP':>9} {'@5% FP':>9}", flush=True)
    raid0, raid1 = load("raid:0"), load("raid:1")
    hc3_train, hc3_test = load("train"), load("test")

    print("\n-- trained on half of RAID, scored on the other half", flush=True)
    for label, groups in [("all features", ALL), ("metrics only (no phrases)", METRICS)]:
        score, top = fit(raid0, groups)
        report(label, score, raid1)
        if label == "metrics only (no phrases)":
            metrics_score, metrics_top = score, top
        if label == "all features":
            all_score = score
    print("   heaviest metrics:", ", ".join(f"{k} ({w:+.2f})" for _, k, w in metrics_top), flush=True)

    print("\n-- the same models, scored on a corpus they never saw (HC3)", flush=True)
    report("RAID-trained, all features -> HC3", all_score, hc3_test)
    report("RAID-trained, metrics only -> HC3", metrics_score, hc3_test)
    for label, groups in [("HC3-trained, all features -> RAID", ALL), ("HC3-trained, metrics only -> RAID", METRICS)]:
        score, _ = fit(hc3_train, groups)
        report(label, score, raid1)

    print("\n-- trained on both, scored on each", flush=True)
    for label, groups in [("all features", ALL), ("metrics only", METRICS)]:
        score, _ = fit(raid0 + hc3_train, groups)
        report(f"both-trained, {label} -> RAID", score, raid1)
        report(f"both-trained, {label} -> HC3", score, hc3_test)

    print("\n-- one generator held out of training entirely, scored on it (metrics only / all)", flush=True)
    humans1 = [d for d in raid1 if d[0] == "human"]
    for held in ("chatgpt", "gpt4", "llama-chat", "mistral-chat", "cohere-chat", "mpt", "gpt2", "cohere"):
        train = [d for d in raid0 if d[1] != held]
        unseen = humans1 + [d for d in raid0 + raid1 if d[1] == held]
        for label, groups in [("metrics only", METRICS), ("all features", ALL)]:
            score, _ = fit(train, groups)
            report(f"without {held}, {label}", score, unseen)
