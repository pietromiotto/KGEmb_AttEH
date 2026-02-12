#!/bin/bash
#SBATCH --job-name=KGEmb_testing
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

# === Handle model path argument ===
MODEL_SUBPATH="$1"
shift  # Shift is to access any additional arguments
EXTRA_ARGS="$@"  

MODEL_PATH="${LOG_DIR}/${MODEL_SUBPATH}"

# === Sanitize model path for file naming ===
ARGS_SLUG=$(echo "$MODEL_SUBPATH" | tr '/' '_')

# === Name slurm outputs with timestamp and arguments ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${OUT_DIR}/slurm_${TIMESTAMP}_${ARGS_SLUG}.out"
exec > "$OUTFILE" 2>&1
echo "Started SLURM job for model: $MODEL_SUBPATH"
echo "Additional test.py args: $EXTRA_ARGS"

# === Load CUDA ===
module load cuda

# === Activate Conda ===
eval "$(/miniconda3/bin/conda shell.bash hook)"
conda activate /miniconda3/envs/env_name

# === Navigate to test.py directory ===
cd /KGEmb

# === Check PyTorch and GPU ===
python -c "
import torch
print('------------------GPU-------------------------------')
print('PyTorch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB') 
else:
    print('ERROR: No GPU detected')
print('----------------------------------------------------')
"

# === Print Model Infos ===
python -c "
import json
config_path = '${MODEL_PATH}/config.json'
try:
    with open(config_path) as f:
        config = json.load(f)
    print('=== config.json ===')
    print(json.dumps(config, indent=4))
    print('===================')
except Exception as e:
    print(f'Could not read config.json: {e}')
"

# === Run test.py with model path and any extra args ===
python test.py --model_dir "$MODEL_PATH" $EXTRA_ARGS

#EXAMPLE USAGE:
#sbatch KGEmb_testpy.slurm --model_dir logs/time/dataset/model  --all 

