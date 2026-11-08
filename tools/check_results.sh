#!/bin/bash
# Recompute every table from the per-edge files shipped in this repository,
# without network access, model weights or web caches, and compare the new
# logs with results/. Needs Python 3.9+ (standard library only).
#
#   bash tools/check_results.sh          -> check/results/, then a diff summary
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config.sh"; source "$ROOT/config_extend.sh"
# shellcheck disable=SC1091
source "$ROOT/lib/common.sh"; source "$ROOT/pipeline.sh"
PY="${PY:-python3}"; SCR="$ROOT/scripts"
WORK="$ROOT/check/work"; RES="$ROOT/check/results"
rm -rf "$ROOT/check"; mkdir -p "$WORK" "$RES"
find "$ROOT/derived_data" -type f -exec cp {} "$WORK/" \;
cp "$ROOT"/results/csv/*.csv "$WORK/"
cp "$ROOT/results/label_coverage.json" "$RES/"
export WORK RES
step_analysis > "$ROOT/check/analysis.log" 2>&1
echo "recomputed logs in check/results/"
n_same=0; n_diff=0
for f in "$RES"/*.txt "$RES"/*.md; do
    b="$(basename "$f")"; [[ -f "$ROOT/results/$b" ]] || continue
    # ignore lines that carry absolute paths or timestamps
    if diff -q <(grep -v -E '^/|work/|<repo>|wrote |[0-9]{4}-[0-9]{2}-[0-9]{2}' "$f") \
               <(grep -v -E '^/|work/|<repo>|wrote |[0-9]{4}-[0-9]{2}-[0-9]{2}' "$ROOT/results/$b") >/dev/null; then
        n_same=$((n_same+1))
    else
        n_diff=$((n_diff+1)); echo "  differs: $b"
    fi
done
echo "identical: $n_same   different: $n_diff"
