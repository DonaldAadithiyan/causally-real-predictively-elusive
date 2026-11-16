#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
L=runs/p126_round2/logs; M=runs/p126_round2/wp6
todo="ETTm1:0:stepzero ETTm2:0:stepzero electricity:0:stepzero,wp1,A,C,wp3 traffic:0:stepzero,wp1,A,C,wp3 electricity:1:stepzero,wp1,A,C,wp3 traffic:1:stepzero,wp1,A,C,wp3 traffic:2:stepzero,wp1,A,C,wp3 electricity:2:stepzero,wp1,A,C,wp3"
done_list=""
while true; do
  left=0
  for item in $todo; do
    ds=${item%%:*}; rest=${item#*:}; s=${rest%%:*}; parts=${rest#*:}
    case " $done_list " in *" $ds:$s "*) continue;; esac
    left=$((left+1))
    if [ -f $M/${ds}_seed${s}_meta.json ]; then
      python p126/r2_wp6_post.py $parts $ds:$s >> $L/wp6_post.log 2>&1
      done_list="$done_list $ds:$s"
    fi
  done
  [ $left -eq 0 ] && break
  sleep 60
done
echo WP6_POST_DONE >> $L/wp6_post.log
