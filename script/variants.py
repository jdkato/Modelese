#!/usr/bin/env python3
"""Scores the model with some feature groups left out, on both benchmarks.

    python3 script/variants.py rhythm,words,function,marks,surprise,phrases

Trains on half of RAID plus HC3's training half, scores the other halves.
Run it under MODELESE_WORDS / MODELESE_PAIRS to shrink the frequency tables.
"""
import os
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model  # noqa: E402

groups = sys.argv[1].split(",")
score, _ = model.fit(model.load("raid:0") + model.load("train"), groups)
label = f"{'+'.join(groups)} (words {os.environ.get('MODELESE_WORDS', 'all')}, pairs {os.environ.get('MODELESE_PAIRS', 'all')})"
model.report(label[:54] + " -> RAID", score, model.load("raid:1"))
model.report(label[:54] + " -> HC3", score, model.load("test"))
