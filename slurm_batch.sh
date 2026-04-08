#! /bin/bash

#SBATCH --time=23:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_log.out

./.venv/bin/python3 -u -m scripts.benchmarks.nftf.vdp_group