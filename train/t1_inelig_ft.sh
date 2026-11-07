#!/usr/bin/env bash
# The six Task 1 members of final s88: each shipped member (f1 h1 i1 j1 m1 n1)
# fine-tuned 6000 steps from its own checkpoint with finite-but-ineligible
# readings (pct_observed < 75) always visible as inputs (T1_INELIG_TRAIN=1).
# ~30 min per net on one GPU. Note: the archived qm1 is the best checkpoint of a
# run that a shutdown stopped at step 5000 (selection every 500 steps).
cd "$(dirname "$0")/../src"
B="T1_VAL_DAYS=7 T1_INELIG_TRAIN=1"
run() { env $B $2 python -u t1_nn.py train $1 --init $3 --steps 6000 --seed $4; }
run qf1 "T1_TCROP=96 T1_LCROP=64 T1_BS=10" f1 101
run qh1 "T1_TCROP=96 T1_LCROP=64 T1_BS=10" h1 102
run qi1 "T1_TCROP=96 T1_LCROP=64 T1_BS=10" i1 103
run qj1 "T1_TCROP=96 T1_LCROP=64 T1_BS=10" j1 104
run qm1 "T1_TCROP=96 T1_LCROP=64 T1_BS=6" m1 105
run qn1 "T1_TCROP=192 T1_LCROP=96 T1_BS=4" n1 106
