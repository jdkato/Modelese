#!/usr/bin/env python3
"""Scores Vale styles as detectors of model-written text.

    python3 script/bench.py [--split test] [--only FILE] NAME=STYLESPATH:STYLE[,STYLE...] ...

A style is a detector once its alerts are added up: a document's score is the
weight of what fired on it, and a threshold on the score makes the call. This
runs each named configuration over corpus/<split>/human and .../model (see
corpus.py) and reports what the commercial detectors report: the share of
model text caught at a fixed rate of false alarms on human text.

Two scores are reported, since which is better depends on the style: the
weighted count of alerts (suggestion 1, warning 2, error 3), and that count
per 100 words.

Set VALE to the binary to use.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

WEIGHT = {"suggestion": 1, "warning": 2, "error": 3}
root = pathlib.Path(__file__).resolve().parent.parent
vale = os.environ.get("VALE", "vale")

args = sys.argv[1:]
split = "test"
if args[:1] == ["--split"]:
    split, args = args[1], args[2:]
only = None
if args[:1] == ["--only"]:
    # A file of `label/name` lines: score just those documents.
    only = set(pathlib.Path(args[1]).read_text().split())
    args = args[2:]
if not args:
    sys.exit(__doc__)


def lint(styles_path, styles, directory):
    """Returns {file name: weighted alert count} for every file in directory."""
    with tempfile.TemporaryDirectory() as tmp:
        ini = pathlib.Path(tmp) / ".vale.ini"
        ini.write_text(
            f"StylesPath = {styles_path}\nMinAlertLevel = suggestion\n\n[*.md]\nBasedOnStyles = {styles}\n"
        )
        out = subprocess.run(
            [vale, "--config", str(ini), "--no-global", "--no-exit", "--output=JSON", str(directory)],
            capture_output=True, text=True, check=False,
        )
    try:
        alerts = json.loads(out.stdout)
    except json.JSONDecodeError:
        sys.exit(f"vale failed on {directory}:\n{out.stdout[:500]}{out.stderr[:500]}")
    for name, found in alerts.items():
        if only is not None and f"{pathlib.Path(directory).name}/{pathlib.Path(name).name}" not in only:
            continue
        for check in {a["Check"] for a in found}:
            fired.setdefault(check, {}).setdefault(pathlib.Path(directory).name, 0)
            fired[check][pathlib.Path(directory).name] += 1
    return {pathlib.Path(k).name: sum(WEIGHT.get(a["Severity"], 1) for a in v) for k, v in alerts.items()}


def rate_at(human, model, fpr):
    """Share of model scores above the threshold that fpr of human scores exceed."""
    ranked = sorted(human)
    threshold = ranked[min(len(ranked) - 1, int((1 - fpr) * len(ranked)))]
    return sum(1 for s in model if s > threshold) / len(model), threshold


def auc(human, model):
    pairs = sorted([(s, 0) for s in human] + [(s, 1) for s in model])
    total, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        total += (i + j + 1) / 2 * sum(1 for k in range(i, j) if pairs[k][1])
        i = j
    return (total - len(model) * (len(model) + 1) / 2) / (len(human) * len(model))


# fired[check][label] counts the documents each rule fired on.
fired = {}

docs = {}
for label in ("human", "model"):
    directory = root / "corpus" / split / label
    if not directory.is_dir():
        sys.exit(f"{directory} is missing; run script/corpus.py")
    docs[label] = {
        p.name: len(p.read_text(encoding="utf-8").split())
        for p in directory.glob("*.md")
        if only is None or f"{label}/{p.name}" in only
    }

print(f"{split}: {len(docs['human'])} human, {len(docs['model'])} model documents")
print(f"{'configuration':22} {'score':14} {'AUC':>5} {'caught @1% FP':>14} {'caught @5% FP':>14} {'flagged human/model':>20}")
for arg in args:
    name, _, spec = arg.partition("=")
    styles_path, _, styles = spec.rpartition(":")
    counts = {label: lint(styles_path, styles.replace(",", ", "), root / "corpus" / split / label) for label in docs}
    for score in ("count", "per 100 words"):
        values = {}
        for label, files in docs.items():
            values[label] = [
                counts[label].get(f, 0) * (100 / max(words, 1) if score != "count" else 1) for f, words in files.items()
            ]
        at1, _ = rate_at(values["human"], values["model"], 0.01)
        at5, _ = rate_at(values["human"], values["model"], 0.05)
        any_h = sum(1 for v in values["human"] if v > 0) / len(values["human"])
        any_m = sum(1 for v in values["model"] if v > 0) / len(values["model"])
        print(f"{name:22} {score:14} {auc(values['human'], values['model']):5.2f} {at1:14.1%} {at5:14.1%} {any_h:9.1%} / {any_m:.1%}")

# Each rule on its own: the share of documents it fires on. A rule earns its
# level here, since one that fires on human text costs the whole score.
print(f"\n{'rule':34} {'human':>7} {'model':>7} {'ratio':>6}")
for check, by in sorted(fired.items(), key=lambda kv: -kv[1].get("model", 0)):
    h = by.get("human", 0) / len(docs["human"])
    m = by.get("model", 0) / len(docs["model"])
    print(f"{check:34} {h:7.2%} {m:7.2%} {m / h if h else float('inf'):6.1f}")
