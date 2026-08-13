#!/bin/bash
# Seed replication for the two arms that matter, so a difference can be told
# from seed noise.  Run after run_ablation.sh has picked a winner.
#   ARMS="bce pasl_topk" SEEDS="1 2" bash tools/run_ablation_seeds.sh
#
# Paths come from the environment so this runs outside the author's machine:
#   DBSURVEY_ROOT          checkout of the danbooru-label-quality-survey repo
#   DBSURVEY_AUDIT_IMAGES  the 150 audited thumbnails
#   ABLATION_OUT           scratch directory for checkpoints and json
set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
OUT=${ABLATION_OUT:-/data/narugo1992/loss_ablation_seeds}
mkdir -p ${OUT}/logs
STEPS=${STEPS:-5000}
for SEED in ${SEEDS:-1 2}; do
  for ARM in ${ARMS:-bce}; do
    TAG=${ARM}_s${SEED}
    [ -f ${OUT}/${TAG}.json ] && { echo "skip ${TAG}"; continue; }
    echo "=== ${TAG} @ $(date -u) ==="
    accelerate launch --num_processes 8 --mixed_precision bf16 \
        tools/loss_ablation.py --arm ${ARM} --steps ${STEPS} --seed ${SEED} \
        --batch-size 64 --num-workers 22 --eval-batches 300 \
        --out ${OUT} > ${OUT}/logs/${TAG}.log 2>&1
    # the harness names outputs by arm; keep them apart per seed
    for EXT in json pt; do
      [ -f ${OUT}/${ARM}.${EXT} ] && mv ${OUT}/${ARM}.${EXT} ${OUT}/${TAG}.${EXT}
    done
    [ -f ${OUT}/${ARM}_counts.npz ] && mv ${OUT}/${ARM}_counts.npz ${OUT}/${TAG}_counts.npz
    echo "  done $(date -u)"
  done
done
echo "SEEDS DONE $(date -u)"
