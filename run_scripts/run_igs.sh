#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH -n1
#SBATCH -G1
#SBATCH -c1
#SBATCH -J igs
#SBATCH -o igs.out
#SBATCH -e igs.error
module purge
source /att/nobackup/jmframe/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
cd /att/nobackup/jmframe/lstm_camels/
echo 'calculating integrated gradients'
python -u ig_nwm_only.py
