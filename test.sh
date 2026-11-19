#!/bin/sh
#
# Every rule carries its own cases in a `tests:` block: one it must fire on
# and one it must leave alone. `--coverage` fails the run on any rule that no
# case makes fire, since a Vale rule that matches nothing loads, runs, and
# reports success.
set -eu
cd "$(dirname "$0")"
"${VALE:-vale}" test --coverage Modelese
