#!/bin/bash
#SBATCH --job-name=JobName
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH -p gpu2
#SBATCH -A {your_account_group}
#SBATCH --time=02:00:00
#SBATCH -o Jobname.out
#SBATCH -e Jobname.err

echo "Starting job on $(hostname) at $(date)"

module load cuda
module load conda

eval "$(conda shell.bash hook)"
conda activate myenv

python -u RNN/vanillarnn.py > RNN/vanillarnn.log 2>&1

echo "Job finished at $(date)"
