# W3PHIAI pipeline: what each step runs. Sourced by run_all.sh.
#
# Paper: "LABEL: Literature Asymmetry between Efficacy and Limits in Drug
# Knowledge and Language Models."

PKG_FILES=(pkg_ind_d2d.csv pkg_con_d2dis.csv pkg_off_d2dis.csv)
COLS=(--col-source x_name --col-target y_name --col-relation display_relation)

# --------------------------------------------------------------- download
step_download () {
    if [[ "$SMOKE" == "1" ]]; then
        log "smoke test: the fixture stands in for the downloads"
        return 0
    fi
    need_network download
    fetch_url "$PRIMEKG_URL"      "$DATA/kg.csv"
    fetch_url "$MONDO_URL"        "$DATA/mondo.obo"
    fetch_url "$DOID_URL"         "$DATA/doid.obo"
    fetch_url "$HETIO_EDGES_URL"  "$DATA/hetionet-v1.0-edges.sif.gz"
    fetch_url "$HETIO_NODES_URL"  "$DATA/hetionet-v1.0-nodes.tsv"
    {
        echo "# data files used by this run"
        echo "# downloaded $(date -u '+%Y-%m-%d %H:%M UTC')"
        for f in kg.csv mondo.obo doid.obo hetionet-v1.0-edges.sif.gz hetionet-v1.0-nodes.tsv; do
            printf '%s  %s\n' "$(sha256sum "$DATA/$f" | cut -c1-64)" "$f"
        done
        grep -m1 '^data-version' "$DATA/mondo.obo" | sed 's/^/mondo.obo /'
        grep -m1 '^data-version' "$DATA/doid.obo"  | sed 's/^/doid.obo  /'
    } > "$RES/data_versions.txt"
    log "wrote $RES/data_versions.txt"
}

# --------------------------------------------------------------- prepare
step_prepare () {
    have "$DATA/kg.csv" "$DATA/mondo.obo" || die "data/ is missing kg.csv or mondo.obo; run: bash run_all.sh --from download"
    if ! have "$WORK/pkg_ind_d2d.csv" "$WORK/pkg_con_d2dis.csv" "$WORK/pkg_off_d2dis.csv" "$WORK/pkg_pc.csv"; then
        step_log prep_primekg py prep_primekg.py --kg "$DATA/kg.csv" --outdir "$WORK" || return 1
    fi
    if ! have "$WORK/mondo_child_parent.csv"; then
        step_log mondo_parents py mondo_parents.py --obo "$DATA/mondo.obo" \
            --primekg "$WORK/pkg_pc.csv" --out "$WORK/mondo_child_parent.csv" || return 1
    fi
    if ! have "$WORK/het_all.csv"; then
        step_log prep_hetionet py prep_hetionet.py \
            --edges "$DATA/hetionet-v1.0-edges.sif.gz" \
            --nodes "$DATA/hetionet-v1.0-nodes.tsv" \
            --metaedges CtD,CpD --prefix het --outdir "$WORK" || return 1
    fi
    if ! have "$WORK/doid_child_parent.csv"; then
        step_log doid_parents py mondo_parents.py --obo "$DATA/doid.obo" --prefix DOID \
            --primekg "$WORK/het_all.csv" --out "$WORK/doid_child_parent.csv" || return 1
    fi
}

# --------------------------------------------------------------- text
# Fetches drug articles and redirect names from Wikipedia and abstracts from
# PubMed, then measures at what level the text names each edge's disease.
step_text () {
    need_network text
    reuse_caches
    check_pubmed_terms
    local web=(--contact "$CONTACT")
    local pmkey=()
    [[ -n "$NCBI_API_KEY" ]] && pmkey=(--api-key "$NCBI_API_KEY")

    # random probe sample, 600 edges per relation (Table 2, supplement)
    if ! have "$WORK/p1_edges_d2.csv"; then
        step_log build_probe_sample py gonogo_p1.py build --edges "${PKG_FILES[@]}" "${COLS[@]}" \
            --names redirects --any-names --ancestors mondo_child_parent.csv \
            --ancestor-depth 2 --n "$PROBE_N" --seed "$SEED" "${web[@]}" \
            --out-edges p1_edges_d2.csv --out-prompts p1_prompts_d2.jsonl || return 1
    fi

    # drug-matched sample: same number of each relation from every drug (Table 1)
    if ! have "$WORK/matched_indication.csv" "$WORK/matched_contraindication.csv"; then
        step_log drug_matched_sample py drug_matched_sample.py \
            --edges pkg_ind_d2d.csv pkg_con_d2dis.csv --a indication --b contraindication \
            --min "$MATCH_MIN" --per-drug "$MATCH_PER_DRUG" --seed "$SEED" \
            --out-prefix matched || return 1
    fi
    if ! have "$WORK/matched_edges.csv"; then
        step_log build_matched py gonogo_p1.py build \
            --edges matched_indication.csv matched_contraindication.csv "${COLS[@]}" \
            --names redirects --any-names --ancestors mondo_child_parent.csv \
            --ancestor-depth 2 --n 2000 --seed "$SEED" "${web[@]}" \
            --out-edges matched_edges.csv --out-prompts matched_prompts.jsonl || return 1
    fi

    # PubMed: single relevance-ranked query, then the balanced two-query corpus
    if ! have "$WORK/matched_edges_pm.csv"; then
        step_log second_corpus_single py second_corpus.py --scored matched_edges.csv \
            --corpus pubmed --names-cache gbif_cache --ancestors mondo_child_parent.csv \
            --ancestor-depth 2 --retmax 40 "${web[@]}" "${pmkey[@]}" \
            --out matched_edges_pm.csv || return 1
    fi
    if ! have "$WORK/matched_edges_pm_balanced.csv"; then
        step_log second_corpus_balanced py second_corpus.py --scored matched_edges.csv \
            --corpus pubmed --names-cache gbif_cache --ancestors mondo_child_parent.csv \
            --ancestor-depth 2 --retmax 40 --extra-terms "$PUBMED_TERMS" \
            "${web[@]}" "${pmkey[@]}" --out matched_edges_pm_balanced.csv || return 1
    fi
    local pair
    for pair in "matched_edges_pm.csv tier_union me_union.csv" \
                "matched_edges_pm.csv tier_second me_pm.csv" \
                "matched_edges_pm_balanced.csv tier_union me_bal_union.csv" \
                "matched_edges_pm_balanced.csv tier_second me_bal_pm.csv"; do
        set -- $pair
        have "$WORK/$3" || ( cd "$WORK" && py retier.py --in "$1" --use "$2" --out "$3" ) || return 1
    done

    # Hetionet control: two benefit relations
    if ! have "$WORK/het_edges.csv"; then
        step_log build_hetionet py gonogo_p1.py build --edges het_all.csv "${COLS[@]}" \
            --names redirects --any-names --ancestors doid_child_parent.csv \
            --ancestor-depth 2 --n "$PROBE_N" --seed "$SEED" "${web[@]}" \
            --out-edges het_edges.csv --out-prompts het_prompts.jsonl || return 1
    fi

    # supplement: floor and the contribution of each naming level
    local d
    for d in 1 2; do
        have "$RES/name_levels_depth$d.csv" && continue
        step_log "name_levels_depth$d" py name_tiers.py --edges "${PKG_FILES[@]}" "${COLS[@]}" \
            --names redirects --any-names --n "$PROBE_N" --seed "$SEED" \
            --ancestors mondo_child_parent.csv --ancestor-depth "$d" "${web[@]}" \
            --csv "$RES/name_levels_depth$d.csv" || warn "name level table depth $d failed; the main results do not need it"
    done

    # models, while the network is here
    if [[ "$SMOKE" != "1" ]]; then
        # shellcheck disable=SC2086
        fetch_models $(printf '%s\n' $PROBE_MODELS $ABSTAIN_MODELS | sort -u)
    fi
}

# --------------------------------------------------------------- extend
# Login node. Downloads the extra models and the FDA labels, then scores the
# labels with the same matcher as the prose corpora.
step_extend () {
    have "$WORK/matched_edges.csv" "$WORK/me_bal_union.csv" \
        || die "the text step has not finished; run: bash run_all.sh --from text"
    if [[ "$SMOKE" == "1" ]]; then
        log "smoke test: no network, so FDA labels are skipped"
        return 0
    fi
    need_network extend
    # shellcheck disable=SC2086
    fetch_models $SCALE_MODELS
    step_log fetch_labels py fetch_labels.py --edges matched_edges.csv \
        --cache label_cache || return 1
    step_log label_coverage py label_coverage.py --edges matched_edges.csv \
        --union me_bal_union.csv --labels label_cache --names-cache gbif_cache \
        --ancestors mondo_child_parent.csv --depth 2 --seed "$SEED" \
        --json "$RES/label_coverage.json" || return 1
}

# --------------------------------------------------------------- models
# Each line: task model walltime
is_gpu () { [[ " $GPU_MODELS " == *" $1 "* ]]; }
is_xl ()  { [[ "$1" == *14B* || "$1" == *32B* || "$1" == *70B* || "$1" == *72B* ]]; }
time_for () {
    if is_gpu "$1"; then echo "$TIME_GPU"
    elif is_small "$1" || [[ "$1" == *1.5B* ]]; then echo "$TIME_SMALL"
    elif is_xl "$1"; then echo "$TIME_XL"
    else echo "$TIME_LARGE"; fi
}

# submit to a GPU node, with the GPU environment and bfloat16 weights
submit_gpu () {
    local name="$1" time="$2"; shift 2
    sbatch --parsable --account="$ACCOUNT" --partition="$GPU_PARTITION" \
        --gpus-per-node=1 --cpus-per-task=8 --mem=64G --time="$time" \
        --job-name="$name" --output="$LOGS/%x_%j.out" --chdir="$ROOT" \
        --wrap="ENV_DIR=$GPU_ENV_DIR TORCH_INDEX=$GPU_TORCH_INDEX USE_GPU=1 $*"
}

model_jobs () {
    local m
    for m in $(printf '%s\n' $PROBE_MODELS $SCALE_MODELS | awk '!seen[$0]++'); do
        echo "probe $m $(time_for "$m")"
    done
    for m in $ABSTAIN_MODELS; do
        echo "abstain $m $(time_for "$m")"
    done
}

run_model_task () {
    local task="$1" model="$2" tag
    tag="$(tag_of "$model")"
    local marker="$WORK/done/${task}_${tag}.ok"
    if [[ -f "$marker" ]]; then log "have $task $tag"; return 0; fi
    local th="${SLURM_CPUS_PER_TASK:-$CPUS}"
    local gflags=()
    [[ "${USE_GPU:-0}" == "1" ]] && gflags=(--gpu --dtype bfloat16)
    log "task $task  model $model  threads $th ${gflags[*]}"
    case "$task" in
        probe)
            ( cd "$WORK" && py run_probe_cpu.py --edges p1_edges_d2.csv \
                --out "p1_edges_d2_${tag}.csv" --model "$model" \
                --threads "$th" --batch-size 8 "${gflags[@]}" ) || return 1 ;;
        abstain)
            ( cd "$WORK" && py abstain_probe.py --scored me_bal_union.csv \
                --wiki-cache wiki_cache --extra-text pubmed_cache \
                --ancestors mondo_child_parent.csv --names-cache gbif_cache \
                --per-band "$PER_BAND" --options 5 --distractors sibling \
                --model "$model" --threads "$th" --batch-size 4 \
                --seed "$SEED" --out "abstain_${tag}_clean.csv" "${gflags[@]}" ) || return 1 ;;
        *) die "unknown task $task" ;;
    esac
    touch "$marker"
}

step_models () {
    have "$WORK/p1_edges_d2.csv" "$WORK/me_bal_union.csv" \
        || die "the text step has not finished; run: bash run_all.sh --from text"
    local ids=() line task model time tag
    while read -r task model time; do
        tag="$(tag_of "$model")"
        [[ -f "$WORK/done/${task}_${tag}.ok" ]] && { log "have $task $tag"; continue; }
        if [[ "$MODE" == "slurm" ]]; then
            local jid
            if is_gpu "$model"; then
                jid=$(submit_gpu "w3_${task}_${tag}" "$time" \
                      "bash $ROOT/run_all.sh --task $task $model") || die "sbatch failed"
            else
                jid=$(submit "w3_${task}_${tag}" "$time" "" \
                      "bash $ROOT/run_all.sh --task $task $model") || die "sbatch failed"
            fi
            log "submitted $task $tag  job $jid  ($time)"
            ids+=("$jid")
        else
            offline_models
            run_model_task "$task" "$model" || return 1
        fi
    done < <(model_jobs)

    if [[ "$MODE" == "slurm" && ${#ids[@]} -gt 0 ]]; then
        local dep jid
        dep="$(IFS=:; echo "${ids[*]}")"
        jid=$(submit "w3_analysis" "$TIME_ANALYSIS" "$dep" \
              "bash $ROOT/run_all.sh --local --only analysis") || die "sbatch failed"
        log "submitted analysis job $jid, starts after ${ids[*]}"
        SUBMITTED=1
    fi
}

# --------------------------------------------------------------- analysis
step_analysis () {
    local small large
    small="$(tag_of "$SMALL_MODEL")"; large="$(tag_of "$LARGE_MODEL")"
    local ab=() m
    for m in $ABSTAIN_MODELS; do
        [[ -s "$WORK/abstain_$(tag_of "$m")_clean.csv" ]] && ab+=("abstain_$(tag_of "$m")_clean.csv")
    done

    # Table 1 and the degree strata (supplement): within-drug tests
    step_log dilution_wiki   py dilution_test.py --scored matched_edges.csv \
        --edges pkg_ind_d2d.csv pkg_con_d2dis.csv --col-source x_name --col-relation display_relation
    step_log dilution_pubmed py dilution_test.py --scored me_bal_pm.csv \
        --edges pkg_ind_d2d.csv pkg_con_d2dis.csv --col-source x_name --col-relation display_relation
    step_log dilution_union  py dilution_test.py --scored me_bal_union.csv \
        --edges pkg_ind_d2d.csv pkg_con_d2dis.csv --col-source x_name --col-relation display_relation
    step_log dilution_pubmed_single py dilution_test.py --scored me_pm.csv \
        --edges pkg_ind_d2d.csv pkg_con_d2dis.csv --col-source x_name --col-relation display_relation
    # negative control
    step_log dilution_hetionet py dilution_test.py --scored het_edges.csv \
        --edges het_all.csv --a treats --b palliates --col-source x_name --col-relation display_relation
    # Table 2, Wilson intervals, paired gain, harder-distractor check
    if have "$WORK/p1_edges_d2_${small}.csv" "$WORK/p1_edges_d2_${large}.csv"; then
        step_log w3_stats py w3_stats.py --small "p1_edges_d2_${small}.csv" \
            --large "p1_edges_d2_${large}.csv" --abstain "${ab[@]}"
    else
        warn "probe outputs missing; w3_stats skipped"
    fi
    # FDA labels (step extend): within-drug test on the label text
    local lf
    for lf in me_label_own me_label_any me_label_allthree; do
        have "$WORK/$lf.csv" && step_log "dilution_${lf#me_}" py dilution_test.py --scored "$lf.csv" \
            --edges pkg_ind_d2d.csv pkg_con_d2dis.csv --col-source x_name --col-relation display_relation
    done
    # retrieval by naming group, scale curve, label table
    local probes=() pool=()
    for m in $(printf '%s\n' $PROBE_MODELS $SCALE_MODELS | awk '!seen[$0]++'); do
        [[ -s "$WORK/p1_edges_d2_$(tag_of "$m").csv" ]] && probes+=("p1_edges_d2_$(tag_of "$m").csv")
    done
    for m in $ABSTAIN_MODELS; do
        [[ "$m" == "$SMALL_MODEL" ]] || pool+=("$(tag_of "$m")")
    done
    step_log w3_extras py w3_extras.py --abstain "${ab[@]}" --pool "${pool[@]}" \
        --probe "${probes[@]}" --labels-json "$RES/label_coverage.json" \
        --out "$RES/EXTRAS.md" --json "$RES/extras.json"

    # failure styles (paper) and declines by page type (supplement)
    step_log w3_failure py w3_failure.py --abstain "${ab[@]}" --pool "${pool[@]}"

    step_log summary py w3_summary.py --work "$WORK" --small-tag "$small" \
        --large-tag "$large" --out "$RES/SUMMARY.md"

    # per-edge outputs next to the logs
    mkdir -p "$RES/csv"
    local f
    for f in p1_edges_d2.csv "p1_edges_d2_${small}.csv" "p1_edges_d2_${large}.csv" \
             matched_edges.csv me_bal_union.csv me_bal_pm.csv me_union.csv me_pm.csv \
             het_edges.csv mondo_child_parent.csv "${ab[@]}" "${probes[@]}" \
             me_label_own.csv me_label_any.csv me_label_allthree.csv; do
        [[ -s "$WORK/$f" ]] && cp "$WORK/$f" "$RES/csv/"
    done

    # figures (they plot the numbers reported in the paper)
    if [[ -d "$ROOT/paper" ]]; then
        ( cd "$ROOT/paper" && "$PY" build_figures.py ) > "$RES/figures.txt" 2>&1 \
            || warn "figures not rebuilt; see $RES/figures.txt"
    fi
    
    return 0
}

# --------------------------------------------------------------- status
status_report () {
    local f
    echo "work dir: $WORK"
    for f in "$DATA/kg.csv" "$DATA/mondo.obo" "$DATA/doid.obo" \
             "$DATA/hetionet-v1.0-edges.sif.gz" "$DATA/hetionet-v1.0-nodes.tsv" \
             "$WORK/pkg_ind_d2d.csv" "$WORK/mondo_child_parent.csv" "$WORK/het_all.csv" \
             "$WORK/doid_child_parent.csv" "$WORK/p1_edges_d2.csv" "$WORK/matched_edges.csv" \
             "$WORK/matched_edges_pm.csv" "$WORK/matched_edges_pm_balanced.csv" \
             "$WORK/me_bal_union.csv" "$WORK/het_edges.csv" "$WORK/me_label_own.csv" \
             "$RES/label_coverage.json" "$RES/EXTRAS.md" "$RES/SUMMARY.md"; do
        if [[ -s "$f" ]]; then echo "  have     ${f#$BASE/}"; else echo "  missing  ${f#$BASE/}"; fi
    done
    echo "model tasks:"
    local task model time
    while read -r task model time; do
        if [[ -f "$WORK/done/${task}_$(tag_of "$model").ok" ]]; then
            echo "  done     $task $(tag_of "$model")"
        else
            echo "  pending  $task $(tag_of "$model")"
        fi
    done < <(model_jobs)
}

# --------------------------------------------------------------- smoke test
smoke_setup () {
    if ! have "$DATA/kg.csv"; then
        log "building the synthetic fixture"
        "$PY" "$ROOT/tests/make_fixture.py" --data "$DATA" --work "$WORK" \
            --terms "$PUBMED_TERMS" || die "fixture build failed"
    fi
    if [[ ! -d "$BASE/tiny_model_a" ]]; then
        log "building two tiny random models"
        "$PY" "$ROOT/tests/tiny_model.py" --text "$WORK/wiki_cache" --out "$BASE/tiny_model_a" --seed 1 \
            || die "tiny model build failed"
        "$PY" "$ROOT/tests/tiny_model.py" --text "$WORK/wiki_cache" --out "$BASE/tiny_model_b" --seed 2 \
            || die "tiny model build failed"
    fi
    SMALL_MODEL="$BASE/tiny_model_a"; LARGE_MODEL="$BASE/tiny_model_b"
    PROBE_MODELS="$SMALL_MODEL $LARGE_MODEL"
    ABSTAIN_MODELS="$SMALL_MODEL $LARGE_MODEL"
    PER_BAND=10; PROBE_N=60
    SCALE_MODELS=""; GPU_MODELS=""
    CPUS=2
}
