# Modelese

A [Vale](https://vale.sh) style for the way language models write.

Legalese is how contracts read and journalese is how headlines read. Modelese
is how an unedited model answer reads: it tells you a point is important
before making it, sends you to a professional it never names, says the answer
depends on a variety of factors, and wraps up with "Overall". A person can
write this way too. The style flags the writing, not the writer.

Every rule here was found by measurement and ships with its numbers.

## Install

> Modelese requires Vale v3.23.0 or later.

```ini
Packages = https://github.com/jdkato/Modelese/releases/latest/download/Modelese.zip

[*.md]
BasedOnStyles = Modelese
```

```console
$ vale sync
```

## Rules

Rates are the share of documents each rule fires on, in held-out human and
model answers to the same questions.

| Rule | Flags | Human | Model |
| ---- | ----- | ----: | ----: |
| `Importance` | "It's important to note that", "carefully consider" | 1.5% | 45% |
| `Deferral` | "consult with a financial advisor", "it's always a good idea to" | 0.5% | 25% |
| `Announce` | "Here are a few", "There are several ways" | 1.9% | 23% |
| `Transitions` | One sentence in five opening on However, Additionally, Overall | 2.2% | 23% |
| `Variety` | "a variety of factors", "depending on several" | 1.2% | 17% |
| `Wrap` | "Overall,", "In conclusion,", "If you have any questions" | 1.2% | 16% |
| `Monotone` | Six or more sentences all near the same length | 2.0% | 13% |
| `Helpers` | "This can help", "it may be helpful to" | 1.3% | 13% |
| `SameStart` | Close to half of the sentences opening on one word | 1.7% | 8% |

Two rules are for commit messages, pull requests, and issues. Their rates are
on pre-2022 human text and on four local models' versions of the same
documents. The rules were written from one model's text and measured on text
they were not written from.

| Rule | Flags | Human | Model |
| ---- | ----- | ----: | ----: |
| `Narration` | "This commit introduces", "The changes made" | 0% to 2.7% | 60% to 98% of commits and pull requests |
| `Pitch` | "improve the overall", "aims to", "ensures that the" | 1.3% to 4.0% | 57% to 87% of commits and pull requests, 29% to 63% of issues |

## As a detector

Add up a document's alerts and the style is a detector. On 2,601 human and
4,838 model answers it had not seen:

| Score | AUC | Caught at 1% false positives | Caught at 5% |
| ----- | --: | ---------------------------: | -----------: |
| Weighted alert count | 0.83 | 46.9% | 53.6% |
| Alerts per 100 words | 0.83 | 39.7% | 60.0% |

A trained classifier does better on the same task. What a style gives you
instead is the reason for every point of the score, on your own machine, and a
way to fix it: each alert names the phrase and says what to write.

Do not use the score to decide who wrote something. At the 1% setting, one
human answer in a hundred is still flagged.

## The fitted score

The rules are the part a writer reads. `script/model.py` measures the rest of
what separates the two sides, as numbers a rule could compute: how much
sentence lengths vary, how often words and word pairs repeat, how expected
each word is, how often each function word and punctuation mark appears, and
which phrases lean which way. A regularized linear model weighs them, so a
score is still a sum of named parts.

Trained on half of RAID and half of HC3 together, and scored on the halves it
had not seen, at 5% false positives:

| Scored on | Caught | AUC |
| --------- | -----: | --: |
| RAID (eleven generators, eight domains) | 80.8% | 0.945 |
| HC3 | 92.4% | 0.986 |

The RAID paper reports 85.0%, 79.6%, and 66.5% for the three detectors it
ranks highest on the same kind of text, measured on its full test set.

What the numbers do not cover:

- **A generator it has not seen.** With one generator held out of training,
  it catches between 36% and 84% of that generator's text.
- **A kind of writing it has not seen.** Trained on the benchmarks alone, the
  5% threshold flags 10% of pre-2022 GitHub issues and 22% of commit messages.
  With a few hundred of each added to training as human text, it flags 2% to
  3% of the ones held out.
- **Model text inside those registers.** `script/registers.py --paired` holds
  one local generator out of training at a time and scores its versions of
  real issues, pull requests, and commit messages. At 5% false positives:

  | Held out | Issues | Pull requests | Commits |
  | -------- | -----: | ------------: | ------: |
  | Llama 3.2 3B | 59% | 73% | 88% |
  | Qwen 2.5 3B | 90% | 74% | 86% |
  | Gemma 2 2B | 65% | 71% | 92% |
  | Phi 3.5 mini | 81% | 86% | 100% |

  These are small local models and 27 to 67 documents a cell. The figures
  say nothing yet about larger hosted models.

`script/export.py` writes the score as data for a page to compute, without
the tagger's features and with small word tables. It catches 84.9% of the
benchmarks' held-out model text at 5% false positives, and 70.6% at 1%.

## How the rules are made

The corpus is [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3):
answers people wrote before ChatGPT existed, paired with answers ChatGPT gave
to the same questions. Pairing keeps topic out of it, so a phrase that
separates the two sides is a habit and not a subject.

```console
$ python3 script/corpus.py     # fetch HC3 and split it into train and test
$ python3 script/raid.py       # sample RAID across its eight domains
$ python3 script/github.py     # pre-2022 issues, pull requests, and commits
$ python3 script/generate.py   # a local model's versions of the same
$ python3 script/mine.py       # rank phrases by model rate over human rate, on train
$ python3 script/bench.py "Modelese=$PWD:Modelese"   # score the style, on test
$ python3 script/model.py      # fit and score the measured features
$ ./test.sh                    # every rule's own cases
```

`mine.py` produces a list to read. A rule is written by hand from it, given a
case it must fire on and one it must leave alone, and kept only if `bench.py`
shows it holding up on the half of the corpus it was not found in.

## Limits

- The rules were mined from HC3, whose model side is ChatGPT from early 2023
  answering questions. On RAID they catch far less: 19% at 5% false
  positives.
- The rhythm rules (`Monotone`, `SameStart`, `Transitions`) score a whole
  document and say nothing about one under six sentences.

## License

The rules and scripts are MIT. HC3 is CC BY-SA 4.0 and RAID is MIT; both are
downloaded by the scripts and are not part of this repository.
