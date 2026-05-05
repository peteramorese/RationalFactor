#! /bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_log.out

#./.venv/bin/python3 -u -m scripts.neurips.prop_nftf_comparison van_der_pol
#./.venv/bin/python3 -u -m scripts.neurips.prop_nftf_comparison cartpole
#./.venv/bin/python3 -u -m scripts.neurips.prop_nftf_comparison dubins_trailer
#./.venv/bin/python3 -u -m scripts.neurips.prop_nftf_comparison planar_quadrotor
#./.venv/bin/python3 -u -m scripts.neurips.prop_nftf_comparison quadcopter
#./.venv/bin/python3 -u -m scripts.neurips.prop_nftf_comparison aircraft

./.venv/bin/python3 -u -m scripts.neurips.filter_nftf_comparison van_der_pol
#./.venv/bin/python3 -u -m scripts.neurips.filter_nftf_comparison bad_sensor_van_der_pol
#./.venv/bin/python3 -u -m scripts.neurips.filter_nftf_comparison cartpole
#./.venv/bin/python3 -u -m scripts.neurips.filter_nftf_comparison dubins_trailer
#./.venv/bin/python3 -u -m scripts.neurips.filter_nftf_comparison planar_quadrotor
#./.venv/bin/python3 -u -m scripts.neurips.filter_nftf_comparison quadcopter
#./.venv/bin/python3 -u -m scripts.neurips.filter_nftf_comparison aircraft
