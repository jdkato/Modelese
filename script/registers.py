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
    """The registers' own model side (generate.py), which no training here saw.

    The human half that was held out sets each register's 5% threshold, and
    the generated documents in that register are scored against it: the
    catch rate inside a register, on a generator the model never met.
    """
    train = benchmarks + [d for name in REGISTERS for d in model.load(name + ":0") if d[0] == "human"]
    score, _ = model.fit(train, model.ALL)
    print("\n-- inside each register: its held-out humans against a local model's versions of the same")
    print(f"{'register':10} {'human':>6} {'model':>6} {'caught at 5% FP':>16} {'caught at 1% FP':>16}")
    for name in REGISTERS:
        human = sorted(score(d, g) for label, _, d, g in model.load(name + ":1") if label == "human")
        generated = [score(d, g) for label, _, d, g in model.load(name) if label == "model"]
        if not generated:
            continue
        at = lambda fpr: sum(1 for s in generated if s > human[min(len(human) - 1, int((1 - fpr) * len(human)))]) / len(generated)
        print(f"{name:10} {len(human):6} {len(generated):6} {at(0.05):16.1%} {at(0.01):16.1%}", flush=True)


paired()
