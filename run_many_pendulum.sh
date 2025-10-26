#!/bin/bash
MAX_JOBS=6

run_job() {
  echo "Launching: $1"
  eval "$1" &
}

joblist=(
    "python parseval_reg/parseval_reg/main.py --env gym_pendulum_drift --algorithm base --repeat_idx 0"
    "python parseval_reg/parseval_reg/main.py --env gym_pendulum_drift --algorithm base --repeat_idx 1"
    "python parseval_reg/parseval_reg/main.py --env gym_pendulum_drift --algorithm base --repeat_idx 2"
    "python parseval_reg/parseval_reg/main.py --env gym_pendulum_drift --algorithm parseval --repeat_idx 0"
    "python parseval_reg/parseval_reg/main.py --env gym_pendulum_drift --algorithm parseval --repeat_idx 1"
    "python parseval_reg/parseval_reg/main.py --env gym_pendulum_drift --algorithm parseval --repeat_idx 2"
)

for job in "${joblist[@]}"; do
  # Wait if too many jobs running
  while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
    sleep 5
  done
  run_job "$job"
  sleep 10
done

wait
echo "✅ All jobs completed!"




