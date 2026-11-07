#!/usr/bin/env bash
# Task 2 of finals s101 / s103: retrain the two public Kaggle pipelines from their
# published code (copied unchanged in src_t2/, apart from env-configurable seeds).
#  1. anubisyy catboost-lgbm-nnls-pipeline-top-50-0-86517 (src_t2/pub): ongoing LGB
#     (train_ongoing.py, ~15 min CPU) and onset LGB + CatBoost (train_onset.py,
#     train_onset_catboost.py; GPU for CatBoost). Writes queue_lgb.txt, onset_lgb.txt,
#     onset.cbm -> copy to models/t2/pub_ongoing_lgb.txt, pub_onset_lgb.txt, pub_onset.cbm.
#  2. yeashusemwal trafficflowbench-trees-u-net-cnn-lb-0-86 (src_t2/yeash/v2): the 7
#     ongoing members, seed = version number, default arguments (~25-40 min each;
#     at most two at a time on 32 GB RAM) -> models/t2/t2_v*/task2_*.txt.
# Check: the retrained anubisyy ongoing model reproduces that notebook's published
# ongoing labels on 100% of 87,000 cells; the onset model on 99.97% of 14,500.
set -e
cd "$(dirname "$0")/.."
export TFB_DATA_ROOT="$TFB_DATA" TFB_OUTPUT_ROOT="$PWD/t2_train/pub" TFB_THREADS=8
(cd src_t2/pub && python -u train_ongoing.py && python -u train_onset.py && python -u train_onset_catboost.py)
cd src_t2/yeash/v2
for s in 11 12 13 14; do python -u task2_v11.py --out ../../../t2_train/t2_v$s --threads 8 --seed $s; done
for s in 15 16;       do python -u task2_v15.py --out ../../../t2_train/t2_v$s --threads 8 --seed $s; done
python -u task2_v18.py --out ../../../t2_train/t2_v20 --threads 8 --seed 20
