#!/usr/bin/env bash
# Event-centred gap training (T1_GAP_EVENT=1), without and with the ramp counts.
cd "$(dirname "$0")/../src"
B="T1_VAL_DAYS=7 T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.6 T1_GAP_EVENT=1"
echo "[$(date +%m-%d\ %H:%M)] start ge1"
env $B python -u t1_nn.py train ge1 --init f1 --steps 8000 --seed 21 > out/t1_nn_ge1.log 2>&1
echo "[$(date +%m-%d\ %H:%M)] done ge1: $(grep '^step' out/t1_nn_ge1.log | tail -1)"
echo "[$(date +%m-%d\ %H:%M)] start gr1"
env $B T1_RAMPS=1 python -u t1_nn.py train gr1 --init f1 --steps 8000 --seed 22 > out/t1_nn_gr1.log 2>&1
echo "[$(date +%m-%d\ %H:%M)] done gr1: $(grep '^step' out/t1_nn_gr1.log | tail -1)"
echo "[$(date +%m-%d\ %H:%M)] all done"
