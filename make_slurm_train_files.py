#!/att/nobackup/jframe/anaconda3/bin/python
# This is a script to generate slurm job scripts for the lstm. 

import os

# Delete existing job scripts
cmd = 'touch run_scripts/run_train_fake.slurm z_trash'
os.system(cmd)
cmd = 'mv run_scripts/run_train_* z_trash'
os.system(cmd)

# load in run seeds
seeds = [line.rstrip('\n') for line in open('run_scripts/seeds.txt')]

# open submit script file for writing
cmd = 'mv submit_train_jobs.sh z_trash'
os.system(cmd)
fname = 'submit_train_jobs.sh'

# Write the bash script header
with open(fname, 'w') as F:
    F.write('#!/bin/bash -f\n')
 
    # Loop through all the run sites and years, and add them thp the script
    for s in seeds:

        # job script file name
        temp_name = 'run_scripts/' + 'run_train.template'
        jname = 'run_scripts/run_train_'+s+'.slurm'
 
        # write to job script
        cmd = 'cp ' + temp_name + ' ' + jname
        os.system(cmd)

        cmd = 'sed -i "s/replace_seed/' + s + '/g" ' + jname
        os.system(cmd)

        # write to submit script 
        F.write('sbatch ' + jname + '\n')

## --- End Script ----------------------------------------------------
