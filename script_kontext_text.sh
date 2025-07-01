#!/bin/bash -l
#SBATCH -J Flux-Garment-Remover-Test-Kontext
#SBATCH --array=1-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --time=0-00:05
#SBATCH --gres=gpu:1
#SBATCH --mem=6G

# Diagnostic and Analysis Phase - please leave these in.
echo "Starting job array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
scontrol show job $SLURM_JOB_ID
source ~/.bashrc
nvidia-smi # only if you requested gpus

# Compute Phase
conda activate flux_garment_remover

export HF_HOME=YOUR_HM_HOME
export HUGGINGFACE_TOKEN=YOUR_HF_TOKEN

python test_kontext_generation.py
