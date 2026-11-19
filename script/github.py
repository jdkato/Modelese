#!/usr/bin/env python3
"""Collects human technical writing from GitHub, written before 2022.

    python3 script/github.py

Issues, pull request descriptions, and commit messages are what the people
who use Vale write, and none of the benchmark corpora hold any. Anything
dated before 2022 predates the chat models, so it is human by construction.
This takes the earliest issues and pull requests of a few long-lived
projects, and their commits up to the end of 2021, through the `gh` CLI:

    corpus/issues/human/<repo>-<number>.md
    corpus/prs/human/<repo>-<number>.md
    corpus/commits/human/<repo>-<sha>.md
    corpus/<kind>/prompts.jsonl      each one's repository and title

There is no model side: these measure how often a detector is wrong about
people in each register, which is the number a threshold has to be set by.

Code blocks, quoted replies, HTML comments, and trailers are removed, since
they are not the writer's prose. What is left has to run to fifty words.
"""
import json
import pathlib
import re
import shutil
import subprocess

REPOS = [
    "kubernetes/kubernetes", "rust-lang/rust", "golang/go", "microsoft/vscode", "facebook/react",
    "python/cpython", "nodejs/node", "django/django", "rails/rails", "ansible/ansible",
]
COMMIT_REPOS = REPOS + ["torvalds/linux", "git/git"]
PAGES = 4

root = pathlib.Path(__file__).resolve().parent.parent / "corpus"


def api(path):
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=False)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return []


def prose(text):
    text = text.replace("\r", "")
    # A fence left open, as a cut-off reply leaves one, runs to the end.
    text = re.sub(r"```.*?(?:```|\Z)", " ", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    lines = []
    for line in text.split("\n"):
        # Quoted replies, indented code, tables, and commit trailers.
        if re.match(r"\s*>|\s{4,}\S|\t|\s*\||[A-Z][A-Za-z-]+: \S+@|(Signed-off-by|Reviewed-by|Change-Id|Fixes|Closes|Co-authored-by|git-svn-id|Reviewed-on|Run-TryBot|Auto-Submit|PR-URL|Cc|Acked-by|Tested-by|Link):", line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def keep(kind, name, text, repo, title):
    text = prose(text or "")
    if len(text.split()) < 50:
        return 0
    out = root / kind / "human"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    # What the document is about, for generate.py to ask a model for the same.
    with (root / kind / "prompts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"name": name, "repo": repo, "title": title.strip()}) + "\n")
    return 1


for kind in ("issues", "prs", "commits"):
    shutil.rmtree(root / kind, ignore_errors=True)

counts = {"issues": 0, "prs": 0, "commits": 0}
for repo in COMMIT_REPOS:
    slug = repo.replace("/", "-")
    if repo in REPOS:
        for page in range(1, PAGES + 1):
            for item in api(f"repos/{repo}/issues?state=all&sort=created&direction=asc&per_page=100&page={page}"):
                if not isinstance(item, dict) or item.get("created_at", "9") >= "2022":
                    continue
                kind = "prs" if "pull_request" in item else "issues"
                counts[kind] += keep(kind, f"{slug}-{item['number']}", item.get("body"), repo, item.get("title") or "")
    for page in range(1, PAGES + 1):
        for item in api(f"repos/{repo}/commits?until=2021-12-31T00:00:00Z&per_page=100&page={page}"):
            if isinstance(item, dict):
                message = item["commit"]["message"]
                subject, _, body = message.partition("\n")
                counts["commits"] += keep("commits", f"{slug}-{item['sha'][:12]}", message, repo, subject)
    print(f"{repo:24} issues {counts['issues']:5}  prs {counts['prs']:5}  commits {counts['commits']:5}", flush=True)
