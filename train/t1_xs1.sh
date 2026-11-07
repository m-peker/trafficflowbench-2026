#!/bin/sh
# xs1: the from-scratch twelfth gap net of final s83 (~4.9 h on one GPU).
# Not a finetune: trained from random init with event-centred blackouts on 30%
# of crops, ramp inputs, 16 h x 96-link crops, width 160, 13 blocks.
cd "$(dirname "$0")/../src"
T1_VAL_DAYS=7 T1_WIDTH=160 T1_TCROP=192 T1_LCROP=96 T1_BS=4 \
T1_DILS=1,2,4,8,16,32,1,2,4,8,16,32,1 T1_RAMPS=1 T1_GAPP=0.3 T1_GAP_EVENT=1 \
python -u t1_nn.py train xs1 --steps 24000 --seed 101
