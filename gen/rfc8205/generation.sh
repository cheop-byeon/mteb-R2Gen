#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --account=12345678
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=120G

module load Miniconda3/22.11.1-1
export PS1=\$
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
conda deactivate &>/dev/null
echo "Conda environments: $(conda info --envs)"
echo "EBROOTMINCONDA3: ${EBROOTMINICONDA3}"

conda activate /path/to/.conda/envs/mteb

# Generate rationales using predictions from the specified retrieval model
python rfc_rationale_generator.py --model "jiebi/RFC-DRAlign-LN"