#!/bin/bash -l
#SBATCH -J segformer                 # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=2080-galvani   # Which partition will run your job
#SBATCH --time=0-00:01            # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1               # (optional) Requesting type and number of GPUs
#SBATCH --mem=6G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/mnt/lustre/work/ponsmoll/pba534/ffgarments/segmentation/slurm/%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/lustre/work/ponsmoll/pba534/ffgarments/segmentation/slurm/%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=cesar.diaz-blanco@student.uni-tuebingen.de  # Email to which notifications will be sent
#SBATCH --array=0-81  # Set the range as needed


# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
source ~/.bashrc
nvidia-smi # only if you requested gpus

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls

# Compute Phase
conda activate /mnt/lustre/work/ponsmoll/pba534/.conda/flux

export HF_HOME=/mnt/lustre/work/ponsmoll/pba534/hf_home

# python /mnt/lustre/work/ponsmoll/pba534/ffgarments/segmentation/segment_generated.py /mnt/lustre/work/ponsmoll/pba870/shared/00122_Outer
python /mnt/lustre/work/ponsmoll/pba534/ffgarments/segmentation/segment_dir.py /mnt/lustre/work/ponsmoll/pba534/ffgarments_datagen/data/4D-DRESS/ ${SLURM_ARRAY_TASK_ID}

conda deactivate
