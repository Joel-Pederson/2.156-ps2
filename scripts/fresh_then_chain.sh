#!/bin/bash
# A fresh 300,000-update pass at the mixed cushion, then chain_mix.sh passes on its output.
# Run from ps2/.  Usage: fresh_then_chain.sh TAG SEED STOP
TAG=$1; SEED=$2; STOP=$3
P="$HOME/miniforge3/envs/ps2/bin/python"; D=data/designs
FAST_MARGINS=0.06,0.04,0.04,0.04 FAST_DOUBLES=1176 FAST_TRIPLES=588 FAST_WORKERS=3 caffeinate -i "$P" -u fast_anneal.py \
  $D/start_poolpick_new.npy 0.06 300000 5e-5 1e-6 "$SEED" 0.5 2.0 "$D/fast_fresh_$TAG.npy" 0.0 > "data/fast_fresh_$TAG.log" 2>&1 < /dev/null \
  && exec scripts/chain_mix.sh "$TAG" "$D/fast_fresh_$TAG.npy" $((SEED + 1)) "$STOP"
