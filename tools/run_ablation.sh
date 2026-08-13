#!/bin/bash
# Sequential loss ablation.  Identical seed, backbone, data order, schedule and
# step budget for every arm; only the loss differs.
#
# Paths come from the environment so this runs outside the author's machine:
#   DBSURVEY_ROOT          checkout of the danbooru-label-quality-survey repo
#   DBSURVEY_AUDIT_IMAGES  the 150 audited thumbnails
#   ABLATION_OUT           scratch directory for checkpoints and json
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
OUT=${ABLATION_OUT:-/data/narugo1992/loss_ablation}
mkdir -p ${OUT}/logs
STEPS=${STEPS:-5000}
for ARM in bce asl asl_matched pasl_gamma pasl_topk pasl_dilig; do
  if [ -f ${OUT}/${ARM}.json ]; then echo "skip ${ARM} (done)"; continue; fi
  echo "=== ${ARM} @ $(date -u) ==="
  accelerate launch --num_processes 8 --mixed_precision bf16 \
      tools/loss_ablation.py --arm ${ARM} --steps ${STEPS} \
      --batch-size 64 --num-workers 22 --eval-batches 300 --seed 0 \
      --out ${OUT} > ${OUT}/logs/${ARM}.log 2>&1
  echo "  exit=$? at $(date -u)"
done
echo "ALL ARMS DONE $(date -u)"
