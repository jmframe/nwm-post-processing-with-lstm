#!/att/nobackup/jframe/anaconda3/bin/python
# This is a script to generate slurm job scripts for the lstm. 
import sys
import os

# Delete existing job scripts
cmd = 'touch run_scripts/run_evaluate_fake.slurm z_trash'
os.system(cmd)
cmd = 'mv run_scripts/run_evaluate_*.slurm z_trash'
os.system(cmd)

# load in run seeds
runs = [line.rstrip('\n') for line in open('run_scripts/run_list.txt')]

# open submit script file for writing
cmd = 'mv submit_evaluate_jobs.sh z_trash'
os.system(cmd)
fname = 'submit_evaluate_jobs.sh'

# Write the bash script header
with open(fname, 'w') as F:
    F.write('#!/bin/bash -f\n')
 
    # Loop through all the run sites and years, and add them thp the script
    for rns in runs:

        # job script file name
        temp_name = 'run_scripts/run_evaluate.template'
        jname = 'run_scripts/run_evaluate_' + rns +'.slurm'
 
        # write to job script
        cmd = 'cp ' + temp_name + ' ' + jname
        os.system(cmd)

        cmd = 'sed -i "s/replace_run/' + rns + '/g" ' + jname
        os.system(cmd)

        # write to submit script 
        F.write('sbatch ' + jname + '\n')

## --- End Script ----------------------------------------------------
