#!/bin/sh
#SBATCH --job-name=lora
#SBATCH --output=seq.log
#SBATCH --error=seq.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem=80G
#SBATCH --cpus-per-task=20

srun singularity exec --nv ./containers/final.sif python3 "./sloT5.py"
