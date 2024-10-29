#!/bin/bash
#SBATCH -A b1139
#SBATCH -p b1139testnode
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --job-name="calibration"
#SBATCH --error=log/calibration.%j.err
#SBATCH --output=log/calibration.%j.out

module purge all

/projects/b1139/environments/emodpy-torch/bin/python run_calib.py

