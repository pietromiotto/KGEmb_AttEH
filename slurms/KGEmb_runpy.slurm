#!/bin/bash
#SBATCH --job-name=KGEmb_training
#SBATCH --partition=a100-gpu
#SBATCH --get-user-env	
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --account=account_name 
#SBATCH --mail-user=user_email
#SBATCH --mail-type=END,FAIL


# === Environment Variables ===
export OUT_DIR=""
export LOG_DIR="/KGEmb/logs"
export DATA_PATH="/KGEmb/data"
export WANDB_API_KEY="api_key_here_if_wandb_is_used"

# === Name slurm outputs with timestamp and arguments ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARGS_SLUG=$(echo "$@" | tr ' ' '_' | tr -cd '[:alnum:]_-')
OUTFILE="${OUT_DIR}/slurm_${TIMESTAMP}_${ARGS_SLUG}.out"
exec > "$OUTFILE" 2>&1

# === Load CUDA ===
module load cuda

# === Activate Conda ===
eval "$(/miniconda3/bin/conda shell.bash hook)"
conda activate /miniconda3/envs/env_name

# === Navigate to run.py directory ===
cd /KGEmb

# === Check PyTorch and GPU ===
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB') 
else:
    print('ERROR: No GPU detected')
"

# === WandB: Detect if ONLINE is possible and act consequently ===
python - <<EOF
import os
import wandb

try:
    os.environ["WANDB_MODE"] = "online"
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(project="test-project", mode="online")
    run.finish()
    print("WandB ONLINE confirmed")
    exit(0)
except Exception as e:
    print(f"WandB ONLINE failed: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    export WANDB_MODE=online
else
    export WANDB_MODE=offline
fi
echo "WANDB_MODE set to $WANDB_MODE"

# === Run with any passed arguments ===
python run.py "$@"

#EXAMPLE USAGE: 
#sbatch KGEmb_runpy.slurm  --dataset DD_graph --max_epochs 32 --model AttEH --rank 32


