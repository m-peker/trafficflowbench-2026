#!/usr/bin/env bash
# Retrain the four Task 1 tree members on the CURRENT (2026-09-11 rebuilt) release.
#   TFB_DATA=<release> TFB_WORK=<work dir with cache/ built by reproduce.py> bash train/train_trees.sh
# Seeds are fixed inside the scripts (row samples seed=1, alt seed=11); LightGBM
# defaults otherwise. Early stopping on the last 45 train days. CPU only, ~8-10 h at 8 threads.
set -e
cd "$(dirname "$0")/../src"
export TFB_THREADS=${TFB_THREADS:-8}   # original run: 12; LightGBM output depends slightly on it
python -W ignore task1_run.py      > "$TFB_WORK/train_own.log"    2>&1   # out/t1_model_<panel>.pkl
python -W ignore pooled_t1.py train > "$TFB_WORK/train_pooled.log" 2>&1   # out/t1_pooled.pkl
python -W ignore t1_members.py family > "$TFB_WORK/train_family.log" 2>&1 # out/t1_family_<family>.pkl
python -W ignore t1_members.py alt  > "$TFB_WORK/train_alt.log"    2>&1   # out/t1_alt_<panel>.pkl
echo "trees done"
