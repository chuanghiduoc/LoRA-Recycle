#!/bin/bash
#
# Setup Environment for LoRA-Recycle Hadoop MapReduce
# This script prepares the environment for running MapReduce jobs
#

set -e

echo "=============================================="
echo "  LoRA-Recycle Hadoop Environment Setup"
echo "=============================================="

# ================== Check Prerequisites ==================
echo ""
echo "Step 1: Checking prerequisites..."

# Check if Hadoop is installed
if ! command -v hadoop &> /dev/null; then
    echo "❌ ERROR: Hadoop is not installed!"
    echo "Please install Hadoop 3.x first"
    exit 1
fi

echo "✅ Hadoop found: $(hadoop version | head -n 1)"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed!"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if required Python packages are installed
echo ""
echo "Step 2: Checking Python dependencies..."

REQUIRED_PACKAGES="torch torchvision transformers timm safetensors pillow"
MISSING_PACKAGES=""

for pkg in $REQUIRED_PACKAGES; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo "❌ Missing Python packages:$MISSING_PACKAGES"
    echo ""
    echo "Install them with:"
    echo "  pip install$MISSING_PACKAGES"
    exit 1
fi

echo "✅ All required Python packages are installed"

# ================== Check Hadoop Services ==================
echo ""
echo "Step 3: Checking Hadoop services..."

if ! jps | grep -q "NameNode"; then
    echo "⚠️  WARNING: NameNode is not running!"
    echo "Starting Hadoop services..."
    start-dfs.sh
    start-yarn.sh
    sleep 5
fi

echo "✅ Hadoop services status:"
jps

# ================== Setup Project Root ==================
echo ""
echo "Step 4: Setting up project root..."

# Detect project root (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Auto-detect if running in WSL and adjust paths accordingly
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Detected WSL environment"
    # Running in WSL - ensure we're using /mnt/c path
    if [[ "$PROJECT_ROOT" != /mnt/c/* ]]; then
        PROJECT_ROOT="/mnt/c/LoRA-Recycle"
    fi
fi

echo "Project root: $PROJECT_ROOT"

# Export environment variables
export LORA_RECYCLE_ROOT="$PROJECT_ROOT"
export LORA_HUB_DIR="${PROJECT_ROOT}/lorahub"

# Add to .bashrc for persistence
if ! grep -q "LORA_RECYCLE_ROOT" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# LoRA-Recycle Hadoop MapReduce" >> ~/.bashrc
    echo "export LORA_RECYCLE_ROOT=\"$PROJECT_ROOT\"" >> ~/.bashrc
    echo "export LORA_HUB_DIR=\"${PROJECT_ROOT}/lorahub\"" >> ~/.bashrc
    echo "✅ Added environment variables to ~/.bashrc"
else
    echo "✅ Environment variables already in ~/.bashrc"
fi

# ================== Create Necessary Directories ==================
echo ""
echo "Step 5: Creating necessary directories..."

mkdir -p "${PROJECT_ROOT}/lorahub"
mkdir -p "${PROJECT_ROOT}/pre_datapool"
mkdir -p "${PROJECT_ROOT}/hadoop_mapreduce/logs"

echo "✅ Directories created"

# ================== Test HDFS Access ==================
echo ""
echo "Step 6: Testing HDFS access..."

hdfs dfs -mkdir -p /lora_recycle/test 2>/dev/null || true
hdfs dfs -touchz /lora_recycle/test/test.txt 2>/dev/null || true

if hdfs dfs -test -e /lora_recycle/test/test.txt; then
    echo "✅ HDFS is accessible"
    hdfs dfs -rm -r /lora_recycle/test 2>/dev/null || true
else
    echo "❌ ERROR: Cannot access HDFS!"
    exit 1
fi

# ================== Make Scripts Executable ==================
echo ""
echo "Step 7: Making scripts executable..."

chmod +x "${SCRIPT_DIR}/mapper.py"
chmod +x "${SCRIPT_DIR}/reducer.py"
chmod +x "${SCRIPT_DIR}/run_mapreduce.sh"
chmod +x "${SCRIPT_DIR}/test_local.sh"

echo "✅ Scripts are executable"

# ================== Summary ==================
echo ""
echo "=============================================="
echo "✅ Environment setup complete!"
echo "=============================================="
echo ""
echo "Environment variables:"
echo "  LORA_RECYCLE_ROOT = $LORA_RECYCLE_ROOT"
echo "  LORA_HUB_DIR      = $LORA_HUB_DIR"
echo ""
echo "Next steps:"
echo "  1. Train LoRAs (if not already done):"
echo "     python train_100_loras.py --dataset miniimagenet --num_loras 10 --max_batches 10"
echo ""
echo "  2. Run MapReduce job:"
echo "     cd hadoop_mapreduce"
echo "     ./run_mapreduce.sh miniimagenet 10 5"
echo ""
echo "  3. Or test locally first:"
echo "     ./test_local.sh miniimagenet 2"
echo ""

