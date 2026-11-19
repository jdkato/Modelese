#!/usr/bin/env python3
"""Writes the model side of the GitHub registers with a local chat model.

    python3 script/generate.py [--n 150]

For --n of the human documents in each of corpus/issues, prs, and commits
(github.py), a model is given the same repository and title and asked to
write the same kind of thing. The result goes through the cleaning the human
text went through, into corpus/<kind>/model/ under the same name, so the two
sides are paired.

The model is a GGUF file at corpus/models/llama-3.2-3b-instruct.Q4_K_M.gguf,
served by `llama-server` for the length of the run. It is one small
generator, chosen because it runs here; nothing in training has seen it.
"""
import concurrent.futures
import importlib.util
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

root = pathlib.Path(__file__).resolve().parent.parent / "corpus"
spec = importlib.util.spec_from_file_location("github", pathlib.Path(__file__).with_name("github.py"))
n = int(sys.argv[2]) if sys.argv[1:2] == ["--n"] else 150
PORT = 8791

ASK = {
    "issues": 'Write the body of a GitHub issue for the {repo} project. Its title is "{title}". Write only the body, in 80 to 200 words.',
    "prs": 'Write the description of a GitHub pull request for the {repo} project. Its title is "{title}". Write only the description, in 80 to 200 words.',
    "commits": 'Write a git commit message for the {repo} project. Its subject line is "{title}". Write the whole message, subject line first, with a body of 80 to 200 words.',
}


def prose(text):
    # The cleaning github.py applies, without running its collection.
    source = pathlib.Path(__file__).with_name("github.py").read_text()
    scope = {"re": __import__("re")}
    exec(source[source.index("def prose(text):"):source.index("def keep(")], scope)
    return scope["prose"](text)


server = subprocess.Popen(
    ["llama-server", "-m", str(root / "models" / "llama-3.2-3b-instruct.Q4_K_M.gguf"), "--port", str(PORT), "-c", "4096", "-np", "4", "-ngl", "99"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
try:
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            break
        except OSError:
            time.sleep(1)

    for kind, ask in ASK.items():
        out = root / kind / "model"
        out.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in (root / kind / "prompts.jsonl").open(encoding="utf-8")]
        rows = sorted((r for r in rows if len(r["title"]) > 8), key=lambda r: r["name"][::-1])[:n]
        def write(row):
            if (out / f"{row['name']}.md").exists():
                return 1
            body = json.dumps({
                "messages": [{"role": "user", "content": ask.format(**row)}],
                "max_tokens": 400, "temperature": 0.8,
            }).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
            try:
                reply = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
            except (OSError, KeyError, ValueError):
                return 0
            text = prose(reply)
            if len(text.split()) < 50:
                return 0
            (out / f"{row['name']}.md").write_text(text + "\n", encoding="utf-8")
            return 1

        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            done = sum(pool.map(write, rows))
        print(f"{kind}: {done} of {len(rows)} written", flush=True)
finally:
    server.terminate()
