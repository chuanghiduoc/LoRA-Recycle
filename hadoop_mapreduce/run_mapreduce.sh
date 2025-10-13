#!/bin/bash
#
# Hadoop MapReduce Job Submission Script for LoRA-Recycle PreGenerate
# 
# Usage:
#   ./run_mapreduce.sh [dataset] [num_loras] [num_mappers]
#
# Example:
#   ./run_mapreduce.sh flower 100 10
#

set -e  # Exit on error

# ================== Configuration ==================
DATASET=${1:-miniimagenet}
NUM_LORAS=${2:-100}
NUM_MAPPERS=${3:-10}  # Number of parallel mappers

BACKBONE="base_clip_16"
RESOLUTION=224
RANK=4
WAY_TEST=5
INSTANCE_PER_CLASS=5

# Paths
# Auto-detect if running in WSL and adjust paths accordingly
if grep -qi microsoft /proc/version 2>/dev/null; then
    # Running in WSL - use /mnt/c path
    LORA_RECYCLE_ROOT="${LORA_RECYCLE_ROOT:-/mnt/c/LoRA-Recycle}"
else
    # Running in native Linux
    LORA_RECYCLE_ROOT="${LORA_RECYCLE_ROOT:-/home/baotrong/LoRA-Recycle}"
fi
LORA_HUB_DIR="${LORA_RECYCLE_ROOT}/lorahub"
OUTPUT_DIR="${LORA_RECYCLE_ROOT}/pre_datapool_mapreduce/${DATASET}"

# Hadoop paths
HDFS_INPUT="/lora_recycle/input"
HDFS_OUTPUT="/lora_recycle/output"
HDFS_LORAHUB="/lora_recycle/lorahub"

# ================== Validation ==================
echo "=============================================="
echo "  LoRA-Recycle MapReduce Job Submission"
echo "=============================================="
echo "Dataset:           $DATASET"
echo "Number of LoRAs:   $NUM_LORAS"
echo "Number of Mappers: $NUM_MAPPERS"
echo "Output Directory:  $OUTPUT_DIR"
echo "=============================================="

# Check if Hadoop is running
if ! jps | grep -q "NameNode"; then
    echo "❌ ERROR: Hadoop NameNode is not running!"
    echo "Please start Hadoop first: start-dfs.sh && start-yarn.sh"
    exit 1
fi

echo "✅ Hadoop is running"

# Check if LoRAs exist
if [ ! -d "${LORA_HUB_DIR}/${DATASET}_${BACKBONE}/${WAY_TEST}way" ]; then
    echo "❌ ERROR: LoRAs not found in ${LORA_HUB_DIR}/${DATASET}_${BACKBONE}/${WAY_TEST}way"
    echo "Please train LoRAs first using: python train_100_loras.py --dataset $DATASET --num_loras $NUM_LORAS"
    exit 1
fi

echo "✅ Found LoRAs in ${LORA_HUB_DIR}/${DATASET}_${BACKBONE}/${WAY_TEST}way"

# ================== Prepare HDFS ==================
echo ""
echo "Step 1: Preparing HDFS directories..."

# Clean up old output
hdfs dfs -rm -r -f $HDFS_OUTPUT 2>/dev/null || true
hdfs dfs -rm -r -f $HDFS_INPUT 2>/dev/null || true

# Create directories
hdfs dfs -mkdir -p $HDFS_INPUT
hdfs dfs -mkdir -p $HDFS_LORAHUB

echo "✅ HDFS directories created"

# ================== Note about LoRAs ==================
echo ""
echo "Step 2: Checking LoRA accessibility..."
echo "ℹ️  LoRAs will be read from local filesystem: ${LORA_HUB_DIR}/${DATASET}_${BACKBONE}/${WAY_TEST}way"
echo "ℹ️  Mappers must have access to this path (shared filesystem or NFS)"
echo ""
echo "In this setup:"
echo "  - Input splits are distributed via HDFS"
echo "  - LoRAs are accessed from local/shared filesystem (faster)"
echo "  - Generated images are collected via reducer"
echo ""
echo "✅ LoRA path verified"

# ================== Generate Input File ==================
echo ""
echo "Step 3: Generating input file (LoRA IDs)..."

INPUT_FILE="/tmp/lora_ids_${DATASET}.txt"
> $INPUT_FILE  # Clear file

for i in $(seq 0 $((NUM_LORAS - 1))); do
    echo $i >> $INPUT_FILE
done

echo "✅ Generated $NUM_LORAS LoRA IDs in $INPUT_FILE"

# Upload to HDFS
hdfs dfs -put -f $INPUT_FILE $HDFS_INPUT/lora_ids.txt
echo "✅ Input file uploaded to HDFS"

# ================== Create Config JSON ==================
# Create JSON in single line to avoid issues with newlines in environment variables
CONFIG_JSON="{\"dataset\":\"$DATASET\",\"backbone\":\"$BACKBONE\",\"resolution\":$RESOLUTION,\"rank\":$RANK,\"way_test\":$WAY_TEST,\"instance_per_class\":$INSTANCE_PER_CLASS,\"lora_num\":$NUM_LORAS,\"prune_layer\":[-1],\"prune_ratio\":[0.0],\"mask_ratio\":-1,\"quick_test\":false,\"use_mask\":false,\"output_dir\":\"$OUTPUT_DIR\"}"

echo ""
echo "Step 4: Configuration:"
echo "$CONFIG_JSON"

# ================== Submit MapReduce Job ==================
echo ""
echo "Step 5: Submitting MapReduce job..."
echo "⏳ This will take approximately $((NUM_LORAS / NUM_MAPPERS * 15 / 60)) minutes..."
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get Python user site-packages path
USER_SITE_PACKAGES=$(python3 -m site --user-site)
USER_BASE=$(python3 -c "import site; print(site.USER_BASE)")

echo ""
echo "Python environment:"
echo "  User site-packages: $USER_SITE_PACKAGES"
echo "  User base: $USER_BASE"
echo ""

# Escape JSON for passing as environment variable
CONFIG_JSON_ESCAPED=$(echo "$CONFIG_JSON" | sed 's/"/\\"/g')

# Submit job using Hadoop Streaming
# Note: Number of mappers is determined by input splits, not by parameter
# We can control it by splitting input file or using -D mapreduce.job.maps (hint, not enforced)
mapred streaming \
  -D mapreduce.job.name="LoRA-Recycle-PreGenerate-${DATASET}" \
  -D mapreduce.job.maps=${NUM_MAPPERS} \
  -D mapreduce.job.reduces=1 \
  -D mapreduce.map.memory.mb=2048 \
  -D mapreduce.reduce.memory.mb=2048 \
  -D mapreduce.map.java.opts=-Xmx1536m \
  -D mapreduce.reduce.java.opts=-Xmx1536m \
  -D mapreduce.task.timeout=1800000 \
  -D mapreduce.input.lineinputformat.linespermap=$((NUM_LORAS / NUM_MAPPERS)) \
  -files "${SCRIPT_DIR}/mapper.py,${SCRIPT_DIR}/reducer.py" \
  -input "${HDFS_INPUT}/lora_ids.txt" \
  -output "${HDFS_OUTPUT}" \
  -mapper "/usr/bin/python3 mapper.py" \
  -reducer "/usr/bin/python3 reducer.py" \
  -cmdenv LORA_MAPREDUCE_CONFIG="$CONFIG_JSON" \
  -cmdenv LORA_RECYCLE_ROOT="$LORA_RECYCLE_ROOT" \
  -cmdenv LORA_HUB_DIR="$LORA_HUB_DIR" \
  -cmdenv PYTHONPATH="$LORA_RECYCLE_ROOT:$USER_SITE_PACKAGES" \
  -cmdenv PYTHONUSERBASE="$USER_BASE" \
  -cmdenv HOME="/home/baotrong" \
  -cmdenv HF_HOME="/home/baotrong/.cache/huggingface" \
  -cmdenv TRANSFORMERS_CACHE="/home/baotrong/.cache/huggingface/transformers"

# Check if job succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✅ MapReduce job completed successfully!"
    echo "=============================================="
    
    # ================== Download Results ==================
    echo ""
    echo "Step 6: Downloading results from HDFS..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Download from HDFS
    hdfs dfs -get "${HDFS_OUTPUT}/*" "${OUTPUT_DIR}/"
    
    echo "✅ Results downloaded to $OUTPUT_DIR"
    
    # ================== Show Summary ==================
    if [ -f "${OUTPUT_DIR}/generation_summary.json" ]; then
        echo ""
        echo "=============================================="
        echo "  Generation Summary"
        echo "=============================================="
        cat "${OUTPUT_DIR}/generation_summary.json"
        echo ""
    fi
    
    echo ""
    echo "🎉 All done! You can now use the pre-generated data for training:"
    echo "   python main.py --dataset $DATASET --pre_datapool_path $OUTPUT_DIR"
    echo ""
    
else
    echo ""
    echo "=============================================="
    echo "❌ MapReduce job failed!"
    echo "=============================================="
    echo "Check logs with: yarn logs -applicationId <application_id>"
    exit 1
fi

