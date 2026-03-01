#!/bin/bash
# MTEB Evaluation Script for RFC/CodeConvo Retrieval Tasks
#
# This script demonstrates a comprehensive evaluation setup with all available models,
# datasets, and directions. The nested loops below iterate through all combinations,
# but individual evaluations can also be run separately by:

#SBATCH --job-name=ir
#SBATCH --account=12345678
#SBATCH --time=00-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G


module load Miniconda3/22.11.1-1
export PS1=\$
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
conda deactivate &>/dev/null
echo "Conda environments: $(conda info --envs)"
echo "EBROOTMINCONDA3: ${EBROOTMINICONDA3}"

conda activate /path/to/.conda/envs/mteb

# Available datasets and directions to evaluate
DIRECTIONS=("i2c" "c2i")
NAMES=("rfc7657" "rfc8205" "rfc8335" "rfcs" "ids" "ids-supp")
SPLITS=("test")

# All fine-tuned models available for evaluation
ENCODERS=("bm25s")

for ENCODER in "${ENCODERS[@]}"; do
  for DIREC in "${DIRECTIONS[@]}"; do
    for NAME in "${NAMES[@]}"; do
      for SPLIT in "${SPLITS[@]}"; do
        echo "Running with encoder: $ENCODER and direction: $DIREC and name: $NAME and split: $SPLIT"
        python RFCAlign_IR_mteb.py \
        --model "$ENCODER" \
        --direction "$DIREC"\
        --split "$SPLIT" \
        --path "ir/$NAME/$DIREC/$SPLIT" \
        --name "$NAME" \
        --batch_size 8 \
        --topk 1000
      done
    done
  done
done