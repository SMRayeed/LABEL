#!/bin/bash
# One entry point for the whole pipeline, from an empty folder to the
# results tables.
#
#   bash run_all.sh              login node: fetch, build, then submit the
#                                model jobs and a final analysis job to SLURM
#   bash run_all.sh --local      everything in this shell, no SLURM (needs
#                                network and enough memory for a 7-8B model)
#   bash run_all.sh --smoke      offline test on a small synthetic fixture
#                                with a tiny random model; a few minutes
#   bash run_all.sh --status     what is built and what is missing
#
#   --from STEP    start at STEP and run the rest
#   --only STEP    run STEP alone
#   STEP is one of: download prepare text extend models analysis
#
# Every step skips outputs already on disk, so an interrupted run resumes
# by running the same command again.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/config.sh"
# shellcheck disable=SC1091
[[ -f "$ROOT/config_extend.sh" ]] && source "$ROOT/config_extend.sh"
# shellcheck disable=SC1091
source "$ROOT/lib/common.sh"
# shellcheck disable=SC1091
source "$ROOT/pipeline.sh"

STEPS=(download prepare text extend models analysis)
MODE="slurm"; FROM=""; ONLY=""; SMOKE=0; STATUS=0; TASK=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)  MODE="local" ;;
        --smoke)  SMOKE=1; MODE="local" ;;
        --status) STATUS=1 ;;
        --from)   FROM="$2"; shift ;;
        --only)   ONLY="$2"; shift ;;
        --task)   TASK=("$2" "$3"); shift 2 ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) die "unknown argument: $1 (see bash run_all.sh --help)" ;;
    esac
    shift
done
export SMOKE

if [[ "$SMOKE" == "1" ]]; then
    BASE="$ROOT/smoke"
else
    BASE="$ROOT"
fi
WORK="${WORK:-$BASE/work}"
DATA="${DATA:-$BASE/data}"
RES="${RES:-$BASE/results}"
LOGS="${LOGS:-$BASE/logs}"
SCR="$ROOT/scripts"
mkdir -p "$WORK/done" "$DATA" "$RES" "$LOGS"
export WORK DATA RES LOGS

if [[ "$STATUS" == "1" ]]; then
    status_report
    exit 0
fi

setup_env

if [[ "$SMOKE" == "1" ]]; then
    smoke_setup
fi

if [[ ${#TASK[@]} -eq 2 ]]; then
    offline_models
    run_model_task "${TASK[0]}" "${TASK[1]}"
    exit $?
fi

selected=()
if [[ -n "$ONLY" ]]; then
    selected=("$ONLY")
else
    started=0
    [[ -z "$FROM" ]] && started=1
    for s in "${STEPS[@]}"; do
        [[ "$s" == "$FROM" ]] && started=1
        [[ $started -eq 1 ]] && selected+=("$s")
    done
fi
[[ ${#selected[@]} -gt 0 ]] || die "unknown step '${FROM}${ONLY}'; use one of: ${STEPS[*]}"

log "pipeline     $(basename "$ROOT")"
log "mode         $MODE"
log "steps        ${selected[*]}"
log "work dir     $WORK"

SUBMITTED=0
for s in "${selected[@]}"; do
    declare -F "step_$s" >/dev/null || die "unknown step: $s"
    log "======== step: $s"
    "step_$s" || die "step '$s' failed; fix the cause and rerun: bash run_all.sh --from $s"
    if [[ "$SUBMITTED" == "1" ]]; then
        log "model jobs and the analysis job are queued; results land in $RES"
        log "follow them with:  squeue -u \$USER   and   ls $LOGS"
        exit 0
    fi
done
log "done. Results are in $RES"
