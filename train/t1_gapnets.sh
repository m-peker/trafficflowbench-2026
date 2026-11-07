#!/usr/bin/env bash
# Gap-aware finetunes of the shipped members (T1_GAPP=0.5): s70 showed one such
# net (gf1) as the gap fallback is worth +0.0011. Structure, not seed, is where
# ensemble diversity comes from, so the parents differ: h1/i1 (f1's shape),
# m1 (dilations to 32, width 192), n1 (192-step context).
cd "$(dirname "$0")/../src"
C="T1_VAL_DAYS=7 T1_GAPP=0.5"
run() { echo "[$(date +%m-%d\ %H:%M)] start $1"; env $C $2 python -u t1_nn.py train $1 --init $3 --steps 6000 --seed $4 > out/t1_nn_$1.log 2>&1; echo "[$(date +%m-%d\ %H:%M)] done $1: $(grep '^step' out/t1_nn_$1.log | tail -1)"; }
run gh1 "T1_TCROP=96 T1_LCROP=64 T1_BS=10" h1 11
run gi1 "T1_TCROP=96 T1_LCROP=64 T1_BS=10" i1 12
run gm1 "T1_TCROP=96 T1_LCROP=64 T1_BS=8" m1 13
run gn1 "T1_TCROP=192 T1_LCROP=96 T1_BS=4" n1 14
echo "[$(date +%m-%d\ %H:%M)] all done"
