#!/usr/bin/env bash
# Phase 0 batch orchestrator — ingest → map_and_clean → quality_gate → load_to_db
# for every slug listed below. Fail-soft: a failure in one slug logs to
# data/db/_pipeline_skipped.log and moves on.

set -u
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOGDIR="data/db/_pipeline_logs"
mkdir -p "$LOGDIR"
SKIPLOG="data/db/_pipeline_skipped.log"
: > "$SKIPLOG"

SLUGS=(
  "joebeachcapital/horse-racing"
  "sheikhbarabas/horse-racing-results-uk-ireland-2005-to-2019"
  "takamotoki/jra-horse-racing-dataset"
  "zygmunt/horse-racing-dataset"
  "felipetappata/thoroughbred-races-in-argentina"
  "gdaley/horseracing-in-hk"
  "lantanacamara/hong-kong-horse-racing"
)

run_step() {
  local slug=$1 step=$2 log=$3
  shift 3
  echo "[$(date +%H:%M:%S)] $slug :: $step" | tee -a "$log"
  if ! "$@" >>"$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] $slug :: $step FAILED" | tee -a "$log"
    echo "$slug :: $step failed (see $log)" >>"$SKIPLOG"
    return 1
  fi
  return 0
}

for slug in "${SLUGS[@]}"; do
  safe_slug=${slug//\//__}
  log="$LOGDIR/$safe_slug.log"
  : > "$log"

  echo "=== $slug ===" | tee -a "$log"

  run_step "$slug" "ingest" "$log" \
    $PY scripts/db/ingest_kaggle.py --dataset "$slug" || continue

  run_step "$slug" "map_and_clean" "$log" \
    $PY scripts/db/map_and_clean.py \
      --input "data/staging/$safe_slug" \
      --map "$slug" || continue

  run_step "$slug" "quality_gate" "$log" \
    $PY scripts/db/quality_gate.py \
      --input "data/cleaned/$safe_slug" || continue

  if [[ -f "data/cleaned/$safe_slug/accepted/all.parquet" ]]; then
    run_step "$slug" "load_to_db" "$log" \
      $PY scripts/db/load_to_db.py \
        --input "data/cleaned/$safe_slug/accepted/" || continue
  else
    echo "$slug :: no accepted parquet (quality_gate rejected all)" >>"$SKIPLOG"
    continue
  fi

  echo "[$(date +%H:%M:%S)] $slug :: DONE" | tee -a "$log"
done

echo "=== Phase 0 pipeline finished ==="
echo "Skip log:"
cat "$SKIPLOG"
