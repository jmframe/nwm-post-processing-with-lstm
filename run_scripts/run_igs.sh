#!/bin/bash
module purge
source /home/jmframe/programs/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
cd /home/NearingLab/projects/jmframe/lstm_camels/
echo 'calculating integrated gradients'

python3 -u ig_nwm_only.py &> igs.out &
