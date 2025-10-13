#!/bin/bash
#
# Local Test Script for LoRA-Recycle MapReduce
# Tests mapper and reducer locally without Hadoop
#
# Usage:
#   ./test_local.sh [dataset] [num_loras]
#
# Example:
#   ./test_local.sh flower 2
#

set -e

# ================== Configuration ==================
DATASET=${1:-miniimagenet}
NUM_LORAS=${2:-2}  # Test with only 2 LoRAs

BACKBONE="base_clip_16"
RESOLUTION=224
RANK=4
WAY_TEST=5
INSTANCE_PER_CLASS=5

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Auto-detect if running in WSL and adjust paths accordingly
if grep -qi microsoft /proc/version 2>/dev/null; then
    # Running in WSL - ensure we're using /mnt/c path
    if [[ "$PROJECT_ROOT" != /mnt/c/* ]]; then
        PROJECT_ROOT="/mnt/c/LoRA-Recycle"
    fi
fi

LORA_HUB_DIR="${PROJECT_ROOT}/lorahub"
OUTPUT_DIR="${PROJECT_ROOT}/pre_datapool_test/${DATASET}"

echo "=============================================="
echo "  LoRA-Recycle Local Test"
echo "=============================================="
echo "Dataset:          $DATASET"
echo "Number of LoRAs:  $NUM_LORAS (test mode)"
echo "Output Directory: $OUTPUT_DIR"
echo "=============================================="

# ================== Validation ==================
if [ ! -d "${LORA_HUB_DIR}/${DATASET}_${BACKBONE}/${WAY_TEST}way" ]; then
    echo "❌ ERROR: LoRAs not found in ${LORA_HUB_DIR}/${DATASET}_${BACKBONE}/${WAY_TEST}way"
    echo "Please train LoRAs first using: python train_100_loras.py --dataset $DATASET --num_loras 10 --max_batches 10"
    exit 1
fi

echo "✅ Found LoRAs"

# ================== Create Config ==================
CONFIG_JSON=$(cat <<EOF
{
  "dataset": "$DATASET",
  "backbone": "$BACKBONE",
  "resolution": $RESOLUTION,
  "rank": $RANK,
  "way_test": $WAY_TEST,
  "instance_per_class": $INSTANCE_PER_CLASS,
  "lora_num": $NUM_LORAS,
  "prune_layer": [-1],
  "prune_ratio": [0.0],
  "mask_ratio": -1,
  "quick_test": false,
  "output_dir": "$OUTPUT_DIR"
}
EOF
)

echo ""
echo "Configuration:"
echo "$CONFIG_JSON"
echo ""

# Export environment variables
export LORA_MAPREDUCE_CONFIG="$CONFIG_JSON"
export LORA_RECYCLE_ROOT="$PROJECT_ROOT"
export LORA_HUB_DIR="$LORA_HUB_DIR"

# ================== Generate Input ==================
echo "Step 1: Generating input..."

INPUT_FILE="/tmp/lora_ids_test.txt"
> $INPUT_FILE

for i in $(seq 0 $((NUM_LORAS - 1))); do
    echo $i >> $INPUT_FILE
done

echo "✅ Generated $NUM_LORAS LoRA IDs"

# ================== Run Mapper ==================
echo ""
echo "Step 2: Running mapper locally..."
echo "⏳ This may take a few minutes..."
echo ""

MAPPER_OUTPUT="/tmp/mapper_output_${DATASET}.txt"

cat $INPUT_FILE | python3 "${SCRIPT_DIR}/mapper.py" > $MAPPER_OUTPUT 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Mapper completed successfully"
    echo ""
    echo "Mapper output:"
    grep -v "^\[Mapper\]" $MAPPER_OUTPUT | head -n 10
else
    echo "❌ Mapper failed!"
    echo "Check logs in $MAPPER_OUTPUT"
    exit 1
fi

# ================== Run Reducer ==================
echo ""
echo "Step 3: Running reducer locally..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

REDUCER_OUTPUT="/tmp/reducer_output_${DATASET}.txt"

grep -v "^\[Mapper\]" $MAPPER_OUTPUT | python3 "${SCRIPT_DIR}/reducer.py" > $REDUCER_OUTPUT 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Reducer completed successfully"
    echo ""
    echo "Reducer output:"
    cat $REDUCER_OUTPUT
else
    echo "❌ Reducer failed!"
    echo "Check logs in $REDUCER_OUTPUT"
    exit 1
fi

# ================== Verify Results ==================
echo ""
echo "Step 4: Verifying results..."

if [ -d "$OUTPUT_DIR" ]; then
    echo "✅ Output directory exists: $OUTPUT_DIR"
    
    # Count files
    NUM_UNMASKED=$(find "$OUTPUT_DIR/unmasked" -type f 2>/dev/null | wc -l)
    NUM_MASKED=$(find "$OUTPUT_DIR/masked" -type f 2>/dev/null | wc -l)
    
    echo "   Unmasked images: $NUM_UNMASKED"
    echo "   Masked images:   $NUM_MASKED"
    
    if [ $NUM_UNMASKED -gt 0 ] && [ $NUM_MASKED -gt 0 ]; then
        echo "✅ Images generated successfully!"
    else
        echo "⚠️  WARNING: No images found in output directory"
    fi
    
    # Show summary
    if [ -f "$OUTPUT_DIR/generation_summary.json" ]; then
        echo ""
        echo "Generation summary:"
        cat "$OUTPUT_DIR/generation_summary.json"
    fi
else
    echo "❌ Output directory not found!"
    exit 1
fi

# ================== Summary ==================
echo ""
echo "=============================================="
echo "✅ Local test completed successfully!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. If test looks good, run full MapReduce job:"
echo "     ./run_mapreduce.sh $DATASET 100 10"
echo ""
echo "  2. Or test with more LoRAs:"
echo "     ./test_local.sh $DATASET 5"
echo ""

