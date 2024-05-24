#!/bin/sh
#SBATCH --job-name=lorav2
#SBATCH --output=smallt5q.log
#SBATCH --error=smallt5q.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20

srun singularity exec --nv ./containers/final.sif python3 "./queryingslot5small.py"
