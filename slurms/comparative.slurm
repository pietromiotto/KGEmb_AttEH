#!/bin/bash
#SBATCH --job-name=KGEmb_comp_search
#SBATCH --partition=a100-gpu
#SBATCH --get-user-env	
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=account_name 
#SBATCH --mail-user=user_email
#SBATCH --mail-type=END,FAIL

# --- arguments ---
# $1 = Path to the HPO python script (Required)
# $2 = The rank to run (Required)
# $3 = Number of trials (Optional, defaults to 8)

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing arguments."
    echo "Usage: sbatch hpo_job.sbatch <script_path> <rank> [num_trials]"
    exit 1
fi

HPO_SCRIPT=$1
RANK_TO_RUN=$2
N_TRIALS_DEFAULT=8
N_TRIALS=${3:-$N_TRIALS_DEFAULT} #if $3 not given runs DEFAULT

#--------------------

# === Environment Variables ===
export OUT_DIR=""
export LOG_DIR="/KGEmb/logs"
export DATA_PATH="/KGEmb/data"

# === Name slurm outputs with timestamp and arguments ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${OUT_DIR}/slurm_HPO_Rank_${RANK_TO_RUN}_${TIMESTAMP}.out"
exec > "$OUTFILE" 2>&1

echo "=== STARTING HPO JOB ==="
echo "Script: $HPO_SCRIPT"
echo "Rank: $RANK_TO_RUN"
echo "Trials: $N_TRIALS"
echo "Output Log: $OUTFILE"
echo "========================"

# === Load CUDA ===
module load cuda

# === Activate Conda ===
eval "$(/miniconda3/bin/conda shell.bash hook)"
conda activate /miniconda3/envs/env_name

# === Navigate to the PROJECT ROOT directory ===
cd /KGEmb

# === Check PyTorch and GPU ===
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
"

# === Run the HPO script with its arguments ===
python -u "$HPO_SCRIPT" --rank "$RANK_TO_RUN" --trials "$N_TRIALS"

#Example usage:
#sbatch comparative.slurm KGEmb/hyperp_search/hpo_search.py 32 16
