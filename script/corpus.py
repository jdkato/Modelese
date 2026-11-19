#!/usr/bin/env python3
"""Builds corpus/ from HC3: human and ChatGPT answers to the same questions.

    python3 script/corpus.py

HC3 (Guo et al., 2023; CC BY-SA 4.0) pairs each question with answers people
wrote before ChatGPT existed and answers ChatGPT gave. Pairing matters: both
sides are on the same topics, so a word that separates them is a habit of the
writer and not a subject.

Each answer of fifty words or more becomes one Markdown file, in `train` or
`test` by a hash of its text. Rules are mined and tuned on `train`; the
numbers a rule or the package reports come from `test`.

    corpus/<split>/<human|model>/<n>.md
"""
import hashlib
import json
import pathlib
import re
import shutil
import urllib.request

SUBSETS = ["wiki_csai", "open_qa", "finance", "medicine"]
URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{}.jsonl"

root = pathlib.Path(__file__).resolve().parent.parent / "corpus"
raw = root / "hc3"
raw.mkdir(parents=True, exist_ok=True)

for split in ("train", "test"):
    shutil.rmtree(root / split, ignore_errors=True)

counts = {}
for subset in SUBSETS:
    path = raw / f"{subset}.jsonl"
    if not path.exists():
        print(f"fetching {subset}")
        urllib.request.urlretrieve(URL.format(subset), path)

    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        for label, answers in (("human", row["human_answers"]), ("model", row["chatgpt_answers"])):
            for text in answers:
                if not text or len(text.split()) < 50:
                    continue
                # One paragraph per line in the source; a blank line between
                # them is what Markdown needs to read them the same way.
                # Some answers carry their line breaks as the two characters
                # `\\n`, left over from how they were collected.
                text = text.replace("\\n", "\n")
                text = re.sub(r"\s*\n\s*", "\n\n", text.strip()) + "\n"
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                split = "train" if int(digest[:2], 16) % 2 == 0 else "test"
                out = root / split / label
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{subset}-{digest[:12]}.md").write_text(text, encoding="utf-8")
                counts[(split, label)] = counts.get((split, label), 0) + 1

for (split, label), n in sorted(counts.items()):
    print(f"{split:5} {label:5} {n}")
