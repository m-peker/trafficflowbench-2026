#!/usr/bin/env bash
# Event-centred gap finetunes (ge1's recipe: T1_GAP_EVENT=1, T1_GAPP=0.6, 8000 steps)
# from the other shipped parents, for structural diversity in the gap fallback.
cd "$(dirname "$0")/../src"
B="T1_VAL_DAYS=7 T1_GAPP=0.6 T1_GAP_EVENT=1"
run() { echo "[$(date +%m-%d\ %H:%M)] start $1"; env $B $2 python -u t1_nn.py train $1 --init $3 --steps 8000 --seed $4 > out/t1_nn_$1.log 2>&1; echo "[$(date +%m-%d\ %H:%M)] done $1: $(grep '^step' out/t1_nn_$1.log | tail -1)"; }
run geh1 "T1_TCROP=96 T1_LCROP=64 T1_BS=10" h1 31
run gei1 "T1_TCROP=96 T1_LCROP=64 T1_BS=10" i1 32
run gem1 "T1_TCROP=96 T1_LCROP=64 T1_BS=8" m1 33
run gen1 "T1_TCROP=192 T1_LCROP=96 T1_BS=4" n1 34
echo "[$(date +%m-%d\ %H:%M)] all done"
