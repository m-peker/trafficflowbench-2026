#!/usr/bin/env bash
# The 12 gap nets of final s95: each shipped gap net fine-tuned 3000 steps from
# its own checkpoint with its own gap recipe, plus the finite-but-ineligible
# readings as inputs (T1_INELIG_TRAIN=1). Seeds 201-212.
cd "$(dirname "$0")/../src"
B="T1_VAL_DAYS=7 T1_INELIG_TRAIN=1"
run() { env $B $2 python -u t1_nn.py train $1 --init $3 --steps 3000 --seed $4; }
run gf1q  "T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.5" gf1 201
run gh1q  "T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.5" gh1 202
run gi1q  "T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.5" gi1 203
run gm1q  "T1_TCROP=96 T1_LCROP=64 T1_BS=8 T1_GAPP=0.5" gm1 204
run gn1q  "T1_TCROP=192 T1_LCROP=96 T1_BS=4 T1_GAPP=0.5" gn1 205
run ge1q  "T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.6 T1_GAP_EVENT=1" ge1 206
run gr1q  "T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.6 T1_GAP_EVENT=1 T1_RAMPS=1" gr1 207
run geh1q "T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.6 T1_GAP_EVENT=1" geh1 208
run gei1q "T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.6 T1_GAP_EVENT=1" gei1 209
run gem1q "T1_TCROP=96 T1_LCROP=64 T1_BS=8 T1_GAPP=0.6 T1_GAP_EVENT=1" gem1 210
run gen1q "T1_TCROP=192 T1_LCROP=96 T1_BS=4 T1_GAPP=0.6 T1_GAP_EVENT=1" gen1 211
run xs1q  "T1_TCROP=192 T1_LCROP=96 T1_BS=4 T1_GAPP=0.3 T1_GAP_EVENT=1 T1_RAMPS=1" xs1 212
