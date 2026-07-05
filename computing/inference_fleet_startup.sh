#!/usr/bin/env bash
# Startup script for an L4 inference worker VM (rts-infer-N), plan Phase 3.
#
# Runs at boot on each g2-standard-96 (8x L4). Reads its parameters from instance
# metadata (set by create_inference_fleet.sh), pulls the self-contained
# rts-infer image, and launches one queue worker per GPU. The workers claim
# shards from the GCS queue (inference/claim.py) and write probability COGs;
# the fleet auto-balances against the A100 master via the shared queue.
#
# Auth: relies on the VM service account (via the metadata server) for both
# Artifact Registry pull and GCS read/write — no key files. The SA must have
# roles/artifactregistry.reader + roles/storage.objectAdmin on the buckets.
#
# Idempotent-ish: safe to re-run (docker pull + fresh `docker run` per GPU). The
# queue's done-markers + atomic claims make duplicate/parallel workers safe.
#
# NOTE (pre-flight gate): validate this end-to-end on ONE real rts-infer-1 before
# creating the rest of the fleet (plan Phase 3) — image pull, GPU visibility,
# bucket auth, and 8 workers claiming with zero runtime patches.
set -euo pipefail

log() { echo "[fleet-startup $(date -u +%H:%M:%S)] $*"; }

meta() {
  curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

IMAGE=$(meta docker-image)
RUN_BASE=$(meta run-base)
QUAD_INDEX=$(meta quad-index)
S2_INDEX=$(meta s2-index)
PACKAGES=$(meta packages)          # comma-separated gs:// package dirs
GPUS_PER_VM=$(meta gpus-per-vm)
DL_WORKERS=$(meta dataloader-workers)
PROJECT=$(curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/project/project-id")

log "image=$IMAGE base=$RUN_BASE gpus=$GPUS_PER_VM project=$PROJECT"

# --package A --package B ... from the comma-separated list.
PKG_ARGS=""
IFS=',' read -ra PKG_ARR <<< "$PACKAGES"
for p in "${PKG_ARR[@]}"; do PKG_ARGS="$PKG_ARGS --package $p"; done

# Wait for the NVIDIA runtime to be ready (DLVM installs drivers on first boot).
for i in $(seq 1 60); do
  if nvidia-smi >/dev/null 2>&1; then break; fi
  log "waiting for nvidia-smi ($i/60)"; sleep 10
done
nvidia-smi -L || { log "FATAL: no GPUs visible"; exit 1; }

# Authenticate docker to Artifact Registry via the VM service account.
gcloud auth configure-docker us-west1-docker.pkg.dev --quiet
docker pull "$IMAGE"

# One worker container per GPU. --gpus device=$g pins the container to its L4
# (which the NVIDIA runtime renumbers to cuda:0 inside — do NOT also set
# CUDA_VISIBLE_DEVICES=$g: that indexes the *visible* set, so for g>=1 it hides
# the only GPU and torch sees zero devices; caught by the 2026-07-05 pre-launch
# audit). --worker-id keeps per-VM/per-GPU contribution visible in the monitor.
# GOOGLE_CLOUD_PROJECT lets google-cloud-storage bill list/read.
for g in $(seq 0 $((GPUS_PER_VM - 1))); do
  log "launching worker on GPU $g"
  docker run -d --restart=on-failure:3 \
    --name "rts-worker-$g" \
    --gpus "device=$g" \
    -e GOOGLE_CLOUD_PROJECT="$PROJECT" \
    "$IMAGE" \
    scripts/run_inference_worker.py \
      --base "$RUN_BASE" \
      --quad-index "$QUAD_INDEX" \
      --s2-index "$S2_INDEX" \
      $PKG_ARGS \
      --device cuda \
      --num-workers "$DL_WORKERS" \
      --worker-id "$(hostname):gpu$g"
done

# Per-VM health check: all containers up a few seconds after launch.
sleep 20
UP=$(docker ps --filter "name=rts-worker-" --filter "status=running" -q | wc -l)
log "health: $UP/$GPUS_PER_VM worker containers running"
if [ "$UP" -lt "$GPUS_PER_VM" ]; then
  log "WARN: only $UP/$GPUS_PER_VM workers up — inspect 'docker logs rts-worker-*'"
fi
log "startup complete"
