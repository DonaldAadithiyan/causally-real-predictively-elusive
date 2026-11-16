#!/usr/bin/env bash
# Round-2 WP2 chain (synthetic). Started after the synth regen finished.
set -u
cd "$(dirname "$0")/../.."
L=runs/p126_round2/logs
while [ ! -f runs/p126_round2/arrays/synth_seed2_repro.json ] || [ ! -f runs/p126_round2/arrays/synth_seed0_repro.json ]; do sleep 30; done
python p126/r2_wp2.py verify > $L/wp2_verify.log 2>&1
A=""; B=""
i=0
for r in 1.0 1.5 2.0 3.0 5.0; do for p in 0.97 0.995 0.999; do
  if [ "$r" = "5.0" ] && [ "$p" = "0.995" ]; then continue; fi
  if [ $((i % 2)) -eq 0 ]; then A="$A $r:$p:0"; else B="$B $r:$p:0"; fi; i=$((i+1))
done; done
python p126/r2_wp2.py make $A > $L/wp2_make_a.log 2>&1 &
python p126/r2_wp2.py make $B > $L/wp2_make_b.log 2>&1 &
wait
SA=""; SB=""; i=0
for f in runs/p126_round2/wp2/settings/*.json; do t=$(basename $f .json); ds=${t%_seed*}; s=${t##*_seed}
  if [ $((i % 2)) -eq 0 ]; then SA="$SA $ds:$s"; else SB="$SB $ds:$s"; fi; i=$((i+1)); done
python p126/r2_wp2.py analyze synth:0 synth:1 synth:2 $SA > $L/wp2_analyze_a.log 2>&1 &
python p126/r2_wp2.py analyze $SB > $L/wp2_analyze_b.log 2>&1 &
python p126/r2_wp2.py link synth:0 synth:1 synth:2 > $L/wp2_link.log 2>&1 &
wait
python p126/r2_wp2.py wp3 synthR1.0P0.995:0 synth:0 > $L/wp2_wp3.log 2>&1
echo WP2_CHAIN_DONE >> $L/wp2_chain.log
