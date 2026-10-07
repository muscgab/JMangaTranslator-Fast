#!/bin/bash
# Modern-GPU latency run (user 2026-10-07): 500 Manga109-s boxes, batch=1, one system after another on one GPU.
# Usage: TAG=rtx3060 bash run_bench.sh   -> /root/bench/out/$TAG/{system}.jsonl + .summary.json; ends in BENCH_DONE.
set -uo pipefail
B=${B:-/root/bench}; P=$B/proj; O=$B/out/${TAG:?}; mkdir -p $O $B/logs
PY=${PY:-python3}                                       # AutoDL 4090: envs/mt/bin/python
SRV=${SRV:-$B/llama.cpp/build/bin/llama-server}
GGUF=${GGUF:-$B/models/Galtransl-v4-4B-2601.gguf}
ts() { date +%F_%T; }
cd $P
SYSTEMS=${SYSTEMS:-ours_eager ours_graph nano gal}
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv > $O/gpu.csv
lscpu | grep -E "Model name|^CPU\(s\)" > $O/cpu.txt
nvidia-smi --query-gpu=timestamp,pstate,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,clocks_event_reasons.active --format=csv -lms 500 > $O/gpu_trace_${SYSTEMS// /_}.csv 2>&1 &   # power / clocks / util / throttle reasons every 0.5 s
DMON=$!
for s in $SYSTEMS; do
  [ $s = gal ] && continue
  echo "[$(ts)] $s"
  $PY benchmarks/speed_modern/bench.py --system $s --out $O > $B/logs/bench_$s.log 2>&1 || echo "[$(ts)] $s FAILED"
  tail -1 $B/logs/bench_$s.log | cut -c1-400
done
if [[ " $SYSTEMS " == *" gal "* ]]; then
echo "[$(ts)] gal: llama-server (1 slot, same flags as the Titan Xp run)"
$SRV -m $GGUF -ngl 99 -np 1 -c 4096 --host 127.0.0.1 --port 18080 \
  --seed 20261006 > $B/logs/llama_server.log 2>&1 &
SPID=$!
up=0; for i in $(seq 1 120); do curl -sf http://127.0.0.1:18080/health >/dev/null && { up=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 2; done
if [ $up = 1 ]; then
  $PY benchmarks/speed_modern/bench.py --system gal --out $O > $B/logs/bench_gal.log 2>&1 || echo "[$(ts)] gal FAILED"
  tail -1 $B/logs/bench_gal.log | cut -c1-400
else
  echo "[$(ts)] llama-server did not come up"; tail -5 $B/logs/llama_server.log
fi
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
fi
kill $DMON 2>/dev/null
echo "[$(ts)] BENCH_DONE"
