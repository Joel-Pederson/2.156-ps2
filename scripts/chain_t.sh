#!/bin/bash
# Annealing passes on one chain until STOP (epoch seconds); each pass starts from the last one's
# output. Run from ps2/.  Usage: chain.sh TAG START SEED STOP
TAG=$1; CUR=$2; SEED=$3; STOP=$4
P="$HOME/miniforge3/envs/ps2/bin/python"; D=data/designs
p=1
while [ "$(date +%s)" -lt "$STOP" ]; do
  OUT=$D/fast_${TAG}_p$p.npy
  FAST_DOUBLES=1176 FAST_TRIPLES=1176 FAST_WORKERS=3 caffeinate -i "$P" -u fast_anneal.py \
    "$CUR" 0.06 150000 3e-5 1e-6 "$SEED" 0.5 1.5 "$OUT" 0.0 > "data/fast_${TAG}_p$p.log" 2>&1 < /dev/null || break
  CUR=$OUT; SEED=$((SEED + 100)); p=$((p + 1))
done
