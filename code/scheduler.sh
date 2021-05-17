#!/bin/bash
#MSUB -A p30135
#MSUB -q normal
#MSUB -l walltime=48:00:00
#MSUB -M nzweifel@u.northwestern.edu
#MSUB -j oe
#MSUB -N 2-obj_dataset
#MSUB -l nodes=1:ppn= 10

# Leave a blank line, like above, before you start your other commands

# with #MSUB, a # doesn't indicate a comment;
# it's part of the MSUB specification (and first line).
# In the rest of the script, # starts a comment

# add a project directory to your PATH (if needed)
export PATH=$PATH:/Final_Project

# load modules you need to use: these are just examples
# module load python/anaconda3.6
module load numpy/1.19.2

# Set your working directory 
# This sets it to the directory you're submitting from -- change as appropriate
# cd $PBS_O_WORKDIR
#echo $XINIT1
#echo $XINIT2
# After you change directories with the command above, all files below 
# are then referenced with respect to that directory

# A command you actually want to execute (example):
# Another command you actually want to execute, if needed (example):
# python error_function-quest.py
python3 run_quest_test.py ${STARTID}


