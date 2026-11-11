#!/bin/bash
# Three follow-up checks on the finished run. All reuse work/ as run_all.sh
# left it (texts, labels, names, the harder-option questions).
#
#   1. Relation check: does the sentence that names the disease also state the
#      relation? Cue words for every edge; optionally a model judge.
#   2. Two sheets for a person: a blinded audit of named sentences, and a
#      recall check of contraindications the matcher misses in the label.
#   3. Routing test: the harder-option questions with the passage cut from the
#      drug's whole FDA label, so prose, label and routed answers can be
#      compared on the same questions.
#
#   bash relcheck.sh quick      login node, minutes, no model: cue check,
#                               RBG intervals, audit and recall sheets
#   bash relcheck.sh submit     SLURM: the judge (Qwen2.5-7B) and the label-page
#                               runs for the three 7-8B models
#   bash relcheck.sh status     which jobs have finished
#   bash relcheck.sh report     judge table and routing table
#   bash relcheck.sh audit      score the filled-in audit and recall sheets
#                               (results/*_merged.csv: one *_label column per annotator)
#
#   bash relcheck.sh --local submit   run the model jobs in this shell
#   bash relcheck.sh --smoke ...      the same on the smoke fixture

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config.sh"
# shellcheck disable=SC1091
[[ -f "$ROOT/config_extend.sh" ]] && source "$ROOT/config_extend.sh"
# shellcheck disable=SC1091
source "$ROOT/lib/common.sh"

MODE="slurm"; SMOKE=0; ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) MODE="local" ;;
        --smoke) SMOKE=1; MODE="local" ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done
export SMOKE
CMD="${ARGS[0]:-}"
[[ -z "$CMD" ]] && { sed -n '2,22p' "$0"; exit 0; }

if [[ "$SMOKE" == "1" ]]; then BASE="$ROOT/smoke"; else BASE="$ROOT"; fi
WORK="${WORK:-$BASE/work}"; RES="${RES:-$BASE/results}"; LOGS="${LOGS:-$BASE/logs}"
SCR="$ROOT/scripts"
mkdir -p "$WORK/done" "$RES" "$LOGS"
export WORK RES LOGS PUBMED_TERMS

JUDGE_MODEL="${JUDGE_MODEL:-$LARGE_MODEL}"
ROUTE_MODELS="${ROUTE_MODELS-$LARGE_MODEL allenai/OLMo-2-1124-7B-Instruct ibm-granite/granite-3.1-8b-instruct}"
AUDIT_N="${AUDIT_N:-40}"       # named sentences per relation, for Wikipedia and the label
RECALL_N="${RECALL_N:-50}"
TIME_RC="${TIME_RC:-12:00:00}"
if [[ "$SMOKE" == "1" ]]; then
    JUDGE_MODEL="$BASE/tiny_model_b"; ROUTE_MODELS="$BASE/tiny_model_b"
    AUDIT_N=5; RECALL_N=5; PER_BAND=10; CPUS=2
fi

setup_env
have "$WORK/matched_edges.csv" "$WORK/me_bal_union.csv" \
    || die "run_all.sh has not finished: $WORK/matched_edges.csv or me_bal_union.csv is missing"
if [[ "$SMOKE" == "1" && ! -d "$WORK/label_cache" ]]; then
    # the fixture has no labels (they need the network): fake them from the
    # cached articles so every code path runs
    log "smoke: building a stand-in label_cache from wiki_cache"
    "$PY" - "$WORK" <<'PY'
import json, os, re, sys
w = sys.argv[1]
src, dst = os.path.join(w, "wiki_cache"), os.path.join(w, "label_cache")
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if not f.endswith(".txt"):
        continue
    t = open(os.path.join(src, f), encoding="utf-8", errors="replace").read()
    name = f[:-4]
    half = len(t) // 2
    json.dump({"drug": name, "found": True, "full": t, "indications": t[:half],
               "contraindications": t[half:], "warnings": ""},
              open(os.path.join(dst, re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:150] + ".json"), "w"))
PY
fi
[[ -d "$WORK/label_cache" ]] || die "no $WORK/label_cache; run: bash run_all.sh --from extend"

RC_ARGS=(--edges matched_edges.csv --wiki-cache wiki_cache --pubmed-cache pubmed_cache
         --labels label_cache --names-cache gbif_cache --ancestors mondo_child_parent.csv)

run_task () {      # run_task judge|route MODEL
    local task="$1" model="$2" tag th="${SLURM_CPUS_PER_TASK:-$CPUS}"
    tag="$(tag_of "$model")"
    local marker="$WORK/done/rc_${task}_${tag}.ok"
    [[ -f "$marker" ]] && { log "have $task $tag"; return 0; }
    offline_models
    case "$task" in
        judge)
            ( cd "$WORK" && py relcheck.py "${RC_ARGS[@]}" --judge "$model" \
                --threads "$th" --batch-size 8 --out "relcheck_judged.csv" ) \
                > "$RES/relcheck_judge.txt" 2>&1 || return 1 ;;
        route)
            ( cd "$WORK" && py abstain_probe.py --scored me_bal_union.csv \
                --wiki-cache wiki_cache --extra-text pubmed_cache \
                --ancestors mondo_child_parent.csv --names-cache gbif_cache \
                --per-band "$PER_BAND" --options 5 --distractors sibling \
                --conditions passage --page-labels label_cache \
                --model "$model" --threads "$th" --batch-size 4 --seed "$SEED" \
                --out "abstain_${tag}_label.csv" ) \
                > "$RES/route_${tag}.txt" 2>&1 || return 1 ;;
        *) die "unknown task $task" ;;
    esac
    touch "$marker"
    log "done $task $tag"
}

jobs_list () {
    echo "judge $JUDGE_MODEL"
    local m
    for m in $ROUTE_MODELS; do echo "route $m"; done
}

case "$CMD" in
    _task) run_task "${ARGS[1]}" "${ARGS[2]}" ;;
    quick)
        log "cue check, audit and recall sheets"
        ( cd "$WORK" && py relcheck.py "${RC_ARGS[@]}" --audit "$AUDIT_N" \
            --recall "$RECALL_N" --out relcheck_sentences.csv ) \
            > >(tee "$RES/relcheck_quick.txt") 2>&1 || die "relcheck.py failed"
        cp "$WORK/audit_sheet.csv" "$WORK/recall_sheet.csv" "$RES/" 2>/dev/null
        log "RBG intervals for the paper's odds ratios"
        {
            for a in "matched_edges.csv tier Wikipedia" "me_bal_union.csv tier_second PubMed" \
                     "me_bal_union.csv tier_union Wiki+PubMed" "me_label_own.csv tier Label-own" \
                     "me_label_any.csv tier Label-whole" "me_label_allthree.csv tier All-three"; do
                set -- $a
                [[ -s "$WORK/$1" ]] && "$PY" "$SCR/mh_rbg.py" "$WORK/$1" "$2" indication contraindication "$3"
            done
            [[ -s "$WORK/het_edges.csv" ]] && "$PY" "$SCR/mh_rbg.py" "$WORK/het_edges.csv" tier treats palliates Hetionet
        } | tee "$RES/mh_rbg.txt"
        log "fill in $RES/audit_sheet.csv and $RES/recall_sheet.csv, then: bash relcheck.sh audit" ;;
    submit)
        while read -r task model; do
            if [[ "$MODE" == "slurm" ]]; then
                jid=$(submit "rc_${task}_$(tag_of "$model")" "$TIME_RC" "" \
                      "bash $ROOT/relcheck.sh _task $task $model") || die "sbatch failed"
                log "submitted $task $(tag_of "$model")  job $jid"
            else
                run_task "$task" "$model" || die "$task $model failed"
            fi
        done < <(jobs_list) ;;
    status)
        while read -r task model; do
            t="$(tag_of "$model")"
            if [[ -f "$WORK/done/rc_${task}_${t}.ok" ]]; then echo "done     $task $t"
            else echo "pending  $task $t"; fi
        done < <(jobs_list) ;;
    report)
        if [[ -s "$WORK/relcheck_judged.csv" ]]; then
            ( cd "$WORK" && py relcheck.py "${RC_ARGS[@]}" --reuse --out relcheck_judged.csv ) \
                > "$RES/relcheck_report.txt" 2>&1
            sed -n '/DOES THE NAMING/,$p' "$RES/relcheck_report.txt"
        else
            echo "=== no judge output yet"
        fi
        tags=()
        for m in $ROUTE_MODELS; do
            t="$(tag_of "$m")"
            [[ -s "$WORK/abstain_${t}_label.csv" && -s "$WORK/abstain_${t}_clean.csv" ]] && tags+=("$t")
        done
        if [[ ${#tags[@]} -gt 0 ]]; then
            ( cd "$WORK" && py route_report.py --models "${tags[@]}" --dir . ) \
                | tee "$RES/route_report.txt"
        else
            echo "=== no routing output yet"
        fi ;;
    audit)
        # filled sheets: one *_label column per annotator (or your_label)
        f="${ARGS[1]:-}"; rf="${ARGS[2]:-}"
        [[ -z "$f" ]] && { f="$RES/audit_sheet_merged.csv"; [[ -s "$f" ]] || f="$RES/audit_sheet.csv"; }
        [[ -z "$rf" ]] && { rf="$RES/recall_sheet_merged.csv"; [[ -s "$rf" ]] || rf="$RES/recall_sheet.csv"; }
        [[ -s "$f" ]] || die "no filled sheet at $f"
        key="$WORK/audit_key.csv"
        # with the judge done, score against the judged key as well
        if [[ -s "$WORK/relcheck_judged.csv" ]]; then
            "$PY" - "$key" "$WORK/relcheck_judged.csv" <<'PY'
import csv, sys
key, judged = sys.argv[1], sys.argv[2]
j = {(r["id"], r["corpus"]): r["judge"] for r in csv.DictReader(open(judged))}
rows = list(csv.DictReader(open(key)))
for r in rows:
    r["judge"] = j.get((r["id"], r["corpus"]), r.get("judge", ""))
w = csv.DictWriter(open(key, "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
PY
        fi
        rargs=(); [[ -s "$rf" ]] && rargs=(--score-recall "$rf")
        py relcheck.py --score-audit "$f" --key "$key" "${rargs[@]}" | tee "$RES/audit_score.txt" ;;
    *) die "unknown command $CMD (see bash relcheck.sh --help)" ;;
esac
