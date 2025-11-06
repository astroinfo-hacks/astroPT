#!/bin/bash
#PBS -N astropt_multimodal_full
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=16:ngpus=2:mem=32gb
#PBS -q gpu
#PBS -o /pbs/home/a/astroinfo09/logs/astropt_multimodal_full.out
#PBS -e /pbs/home/a/astroinfo09/logs/astropt_multimodal_full.err

# AstroPT Multimodal Full Dataset Training Script
# Based on successful DESI training configuration
# Designed to run disconnected in background with full error handling

echo "=== ASTROPT MULTIMODAL FULL DATASET TRAINING ==="
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

# Setup environment
cd /pbs/home/a/astroinfo09/astroPT
#source /pbs/throng/training/astroinfo2025/env/astroPT/bin/activate
source /pbs/throng/training/astroinfo2025/soft/astroPT/source_CC.sh

# Check GPU availability
echo "GPU Status:"
nvidia-smi

# Model configuration (scaling up from 3K test run)
MAX_ITERS=50000        # Production run: 50K iterations (~16x longer than test)
BATCH_SIZE=12          # Per GPU - effective batch size = 12*2*4 = 96 (reduced for memory)
GRAD_ACCUM=4           # Gradient accumulation steps
EVAL_INTERVAL=500      # Evaluate every 500 iterations
LOG_INTERVAL=50        # Log every 50 iterations
CHECKPOINT_INTERVAL=2500  # Save checkpoint every 2500 iterations

# Output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="/pbs/throng/training/astroinfo2025/work/msiudek/logs/astropt_multimodal_full_${TIMESTAMP}"

echo "Configuration:"
echo "  Max iterations: ${MAX_ITERS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRAD_ACCUM}"
echo "  Effective batch size: $((BATCH_SIZE * 2 * GRAD_ACCUM))"
echo "  Output directory: ${OUT_DIR}"
echo "  Estimated training time: 8-12 hours"

# Create output directory
mkdir -p ${OUT_DIR}

# Save this script for reference
cp $0 ${OUT_DIR}/launch_script.sh

# Launch training with 2 GPUs
echo ""
echo "=== LAUNCHING TRAINING ==="
echo "Command: torchrun --standalone --nproc_per_node=2 scripts/train_spectra_images.py"

torchrun --standalone --nproc_per_node=2 scripts/train_spectra_images.py \
    --out-dir ${OUT_DIR} \
    --train-split "train_batch_1+train_batch_2+train_batch_3+train_batch_4+train_batch_5+train_batch_6+train_batch_7+train_batch_8+train_batch_9+train_batch_10+train_batch_11+train_batch_12+train_batch_13+train_batch_14+train_batch_15+train_batch_16+train_batch_17+train_batch_18+train_batch_19+train_batch_20+train_batch_21+train_batch_22+train_batch_23+train_batch_24+train_batch_25" \
    --val-split "test_batch_1+test_batch_2" \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --max-iters ${MAX_ITERS} \
    --eval-interval ${EVAL_INTERVAL} \
    --log-interval ${LOG_INTERVAL} \
    --compile \
    --num-workers 0 2>&1 | tee ${OUT_DIR}/training.log

EXIT_CODE=$?

echo ""
echo "=== TRAINING COMPLETED ==="
echo "Exit code: ${EXIT_CODE}"
echo "Finished at: $(date)"

# Create completion summary
cat > ${OUT_DIR}/completion_summary.txt << EOF
AstroPT Multimodal Full Dataset Training Summary
===============================================

Started: $(date)
Configuration:
- Max iterations: ${MAX_ITERS}
- Batch size per GPU: ${BATCH_SIZE}  
- Effective batch size: $((BATCH_SIZE * 2 * GRAD_ACCUM))
- GPUs used: 2
- Exit code: ${EXIT_CODE}

Output files:
- Model checkpoints: ${OUT_DIR}/ckpt_*.pt
- Training log: ${OUT_DIR}/training.log
- Loss plots: ${OUT_DIR}/loss.png
- Visualizations: ${OUT_DIR}/visualizations/

Next steps if successful:
1. Extract embeddings: python scripts/euclid_desi_dataset/embeddings/extract_multimodal_embeddings.py --checkpoint ${OUT_DIR}/ckpt_best.pt
2. Run analysis: python scripts/euclid_desi_dataset/embeddings/multimodal_embedding_analysis.py
3. Compare with test training results

Status: $([ ${EXIT_CODE} -eq 0 ] && echo "SUCCESS" || echo "FAILED")
EOF

# Send notification (if mail is available)
if command -v mail >/dev/null 2>&1; then
    echo "AstroPT multimodal training completed with exit code ${EXIT_CODE}" | mail -s "Training Update" astroinfo09@example.com
fi

echo "Training summary saved to: ${OUT_DIR}/completion_summary.txt"
echo "Full log available at: ${OUT_DIR}/training.log"

exit ${EXIT_CODE}