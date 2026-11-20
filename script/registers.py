#!/usr/bin/env python3
"""How often the model is wrong about people, register by register.

    python3 script/registers.py

A threshold set on one kind of human writing is the wrong threshold for
another: forum answers, abstracts, and commit messages sit in different
places on the same score. This trains on RAID and HC3 together, sets the
threshold where 5% of their held-out human text is flagged, and then asks
what share of pre-2022 issues, pull requests, and commit messages
(github.py) that threshold flags.

It then moves the threshold to where each register's own false-positive rate
is 5%, and reports what share of the benchmarks' model text still clears it:
what holding the line for that register costs.

Both are run twice: with the benchmarks alone, and with half of each
register added to training as human text, scored on the other half.
"""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model  # noqa: E402

held = model.load("raid:1") + model.load("test")
REGISTERS = ("issues", "prs", "commits")


def run(title, train):
    score, _ = model.fit(train, model.ALL)
    human = sorted(score(d, g) for label, _, d, g in held if label == "human")
    generated = [score(d, g) for label, _, d, g in held if label == "model"]
    threshold = human[int(0.95 * len(human))]
    caught = sum(1 for s in generated if s > threshold) / len(generated)
    print(f"\n-- {title}")
    print(f"benchmarks: threshold flags 5.0% of {len(human)} human documents, catches {caught:.1%} of {len(generated)} model documents")
    print(f"{'register':10} {'documents':>9} {'flagged at that threshold':>26} {'caught at its own 5%':>22}")
    for name in REGISTERS:
        # The half of each register no model here was trained on.
        scores = sorted(score(d, g) for label, _, d, g in model.load(name + ":1") if label == "human")
        flagged = sum(1 for s in scores if s > threshold) / len(scores)
        own = scores[min(len(scores) - 1, int(0.95 * len(scores)))]
        kept = sum(1 for s in generated if s > own) / len(generated)
        print(f"{name:10} {len(scores):9} {flagged:26.1%} {kept:22.1%}", flush=True)


benchmarks = model.load("raid:0") + model.load("train")
if "--paired" not in sys.argv:
    run("trained on the benchmarks alone", benchmarks)
    run("with half of each register added as human text",
        benchmarks + [d for name in REGISTERS for d in model.load(name + ":0") if d[0] == "human"])


def paired():
    """Each local generator in turn, held out of training (generate.py).

    Half of every register's human text and half of the other generators'
    versions go into training with the benchmarks. The held-out generator's
    documents in the other half are then scored against that half's human
    text, at the register's own 5% and 1% thresholds: the catch rate inside
    a register, on a generator the model never met.
    """
    half = {(name, h): model.load(f"{name}:{h}") for name in REGISTERS for h in "01"}
    generators = sorted({g for docs in half.values() for label, g, *_ in docs if label == "model"})
    print("\n-- inside each register: its held-out humans against one generator's versions of the same")
    print(f"{'held out':10} {'register':9} {'human':>6} {'model':>6} {'caught at 5% FP':>16} {'caught at 1% FP':>16}")
    for out in generators + ["none"]:
        train = benchmarks + [d for (_, h), docs in half.items() if h == "0" for d in docs if d[1] != out]
        score, _ = model.fit(train, model.ALL)
        for name in REGISTERS:
            human = sorted(score(d, g) for label, _, d, g in half[name, "1"] if label == "human")
            generated = [score(d, g) for label, gen, d, g in half[name, "1"] if label == "model" and out in (gen, "none")]
            if not generated:
                continue
            at = lambda fpr: sum(1 for s in generated if s > human[min(len(human) - 1, int((1 - fpr) * len(human)))]) / len(generated)
            print(f"{out:10} {name:9} {len(human):6} {len(generated):6} {at(0.05):16.1%} {at(0.01):16.1%}", flush=True)


paired()
