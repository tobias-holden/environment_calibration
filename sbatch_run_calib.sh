#!/bin/bash
#SBATCH -A b1139
#SBATCH -p b1139
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --job-name="calibration"
#SBATCH --error=/projects/b1139/environmental_calibration/simulations/logs/calibration.%j.err
#SBATCH --output=/projects/b1139/environmental_calibration/simulations/logs/calibration.%j.out

module purge all
source /projects/b1139/environments/pytorch-1.11-emodpy-py39/bin/activate

cd /projects/b1139/environmental_calibration/simulations

python run_calib.py