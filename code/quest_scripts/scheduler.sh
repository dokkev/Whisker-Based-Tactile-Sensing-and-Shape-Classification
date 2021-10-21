#!/bin/bash
#SBATCH -A p31427                                    ## Allocation
#SBATCH -p normal                                      ## Queue
#SBATCH -t 48:00:00                                 ## Walltime/duration of the job
#SBATCH -N 1                                         ## Number of Nodes
#SBATCH --ntasks-per-node=52                         ## Number of Cores (Processors)
#SBATCH --mail-user=dongkang2021@u.northwestern.edu  ## Designate email address for job communications
#SBATCH --job-name="natural_objects"                     ## Name of job
#SBATCH --constraint=quest10


## load modules you need to use: these are just examples
module load numpy/1.19.2

## run the script
python3 natural_objects.py 


