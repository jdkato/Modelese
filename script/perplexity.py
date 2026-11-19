#!/usr/bin/env python3
"""A perplexity detector, for comparison: GPT-2's surprise at each document.

    python3 script/perplexity.py [--split test] [--n 600]

The first statistical detectors scored a text by how predictable a language
model found it, on the reasoning that a model writes what a model expects.
This is that baseline, run locally: `llama-perplexity` with a GPT-2 model
file at corpus/models/gpt2.Q8_0.gguf, over --n documents of each label drawn
from corpus/<split>.

The tool scores the second half of each 64-token window, so a document needs
128 tokens; only those of 110 words or more are drawn. The names drawn are
written to corpus/<split>/perplexity.txt, so bench.py can score a style on
exactly the same documents (`--only`).
"""
import concurrent.futures
import pathlib
import random
import re
import subprocess
import sys

args = sys.argv[1:]
opt = {"--split": "test", "--n": "600"}
while args:
    opt[args[0]] = args[1]
    args = args[2:]

root = pathlib.Path(__file__).resolve().parent.parent / "corpus"
model = root / "models" / "gpt2.Q8_0.gguf"
split = root / opt["--split"]
FINAL = re.compile(r"Final estimate: PPL = ([0-9.]+)")


def perplexity(path):
    out = subprocess.run(
        ["llama-perplexity", "-m", str(model), "-f", str(path), "-c", "64", "-b", "64", "-ngl", "0", "-t", "2", "--no-warmup"],
        capture_output=True, text=True, check=False,
    )
    found = FINAL.search(out.stdout + out.stderr)
    return float(found.group(1)) if found else None


random.seed(11)
scores = {}
drawn = []
for label in ("human", "model"):
    files = sorted(p for p in (split / label).glob("*.md") if len(p.read_text(encoding="utf-8").split()) >= 110)
    random.shuffle(files)
    files = files[: int(opt["--n"])]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        values = list(pool.map(perplexity, files))
    scores[label] = [v for v in values if v is not None]
    drawn += [f"{label}/{p.name}" for p, v in zip(files, values) if v is not None]
(split / "perplexity.txt").write_text("\n".join(drawn) + "\n")


def rate_at(human, model, fpr):
    # Low perplexity is the model's side, so the threshold is a floor.
    ranked = sorted(human)
    threshold = ranked[max(0, int(fpr * len(ranked)) - 1)]
    return sum(1 for s in model if s < threshold) / len(model)


def auc(human, model):
    wins = sum(1 for m in model for h in human if m < h) + 0.5 * sum(1 for m in model for h in human if m == h)
    return wins / (len(human) * len(model))


median = lambda xs: sorted(xs)[len(xs) // 2]
h, m = scores["human"], scores["model"]
print(f"{opt['--split']}: {len(h)} human, {len(m)} model documents of 110 words or more")
print(f"median perplexity: human {median(h):.1f}, model {median(m):.1f}")
print(f"GPT-2 perplexity       AUC {auc(h, m):.2f}   caught @1% FP {rate_at(h, m, 0.01):.1%}   caught @5% FP {rate_at(h, m, 0.05):.1%}")
