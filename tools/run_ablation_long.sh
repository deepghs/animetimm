#!/bin/bash
# Longer-horizon confirmation of the three arms that matter.
#
#   bce          is the short-training advantage of ASL just faster convergence?
#   asl_matched  constant gamma = 4.5, the control
#   pasl_gamma   per-tag gamma; the one finding that is real but that the
#                current dbv4-facing metrics would reject
#
# 12,000 steps is 2.4x the first sweep, same seed / data order / schedule shape.
#
# Paths come from the environment so this runs outside the author's machine:
#   DBSURVEY_ROOT          checkout of the danbooru-label-quality-survey repo
#   DBSURVEY_AUDIT_IMAGES  the 150 audited thumbnails
#   ABLATION_OUT           scratch directory for checkpoints and json
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
OUT=${ABLATION_OUT:-/data/narugo1992/loss_ablation_long}
mkdir -p ${OUT}/logs
STEPS=${STEPS:-12000}
for ARM in bce asl_matched pasl_gamma; do
  [ -f ${OUT}/${ARM}.json ] && { echo "skip ${ARM}"; continue; }
  echo "=== ${ARM} @ $(date -u) ==="
  accelerate launch --num_processes 8 --mixed_precision bf16 \
      tools/loss_ablation.py --arm ${ARM} --steps ${STEPS} \
      --batch-size 64 --num-workers 22 --eval-batches 300 --seed 0 \
      --out ${OUT} > ${OUT}/logs/${ARM}.log 2>&1
  echo "  exit=$? at $(date -u)"
done
echo "ALL LONG ARMS DONE $(date -u)"
accelerate launch --num_processes 8 --mixed_precision bf16 \
    tools/posthoc_eval.py --dir ${OUT} --eval-batches 300 \
    > ${OUT}/logs/posthoc.log 2>&1
echo "LONG POSTHOC DONE exit=$? $(date -u)"
