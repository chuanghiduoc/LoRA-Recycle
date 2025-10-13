#!/bin/bash
#
# Quick test for mapper only (for debugging)
#

set -e

DATASET=${1:-miniimagenet}
LORA_ID=${2:-0}

# Detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export LORA_RECYCLE_ROOT="$PROJECT_ROOT"
export LORA_HUB_DIR="${PROJECT_ROOT}/lorahub"

# Create config
CONFIG_JSON=$(cat <<EOF
{
  "dataset": "$DATASET",
  "backbone": "base_clip_16",
  "resolution": 224,
  "rank": 4,
  "way_test": 5,
  "instance_per_class": 5,
  "lora_num": 100,
  "prune_layer": [-1],
  "prune_ratio": [0.0],
  "mask_ratio": -1,
  "quick_test": false,
  "use_mask": false,
  "output_dir": "${PROJECT_ROOT}/pre_datapool_test/${DATASET}"
}
EOF
)

export LORA_MAPREDUCE_CONFIG="$CONFIG_JSON"

echo "Testing mapper with LoRA $LORA_ID..."
echo "$LORA_ID" | python3 "${SCRIPT_DIR}/mapper.py"

