template= """#!/bin/bash

#SBATCH --time=2:00:00   # walltime
#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -J "eval squad"
#SBATCH --mem-per-cpu=4048M
#SBATCH --mail-user=herobd@gmail.com   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#xxSBATCH --qos=standby   
#xxSBATCH --requeue
#xxxSBATCH -C pascal

#130:00:00

export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

#Tesseract things 
module load libtiff/4.0 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fslhome/brianld/local/lib 
export PATH=$PATH:/fslhome/brianld/local/bin 
export TESSDATA_PREFIX=/fslhome/brianld/tesseract/tessdata

module remove miniconda3
module load cuda/10.1
module load cudnn/7.6
cd ~/pairing
source activate /fslhome/brianld/miniconda3/envs/new
"""
do_template="""
echo "{0}" 
python qa_eval.py -d SQuAD -c saved/{0}/checkpoint-latest.pth -g 0 -v 0 --autoregressive
"""

import sys
script = template
if len(sys.argv[1:])>0:
    for job_name in sys.argv[1:]:
        script+=do_template.format(job_name)
else:
    with open('tmp') as f:
        jobs = f.readlines()
    for job in jobs:
        job = [j for j in job.split(' ') if len(j)>0]
        assert len(job)==3
        job_name = job[0]
        script+=do_template.format(job_name)
with open('runs/run_squad_eval.pbs','w') as f:
    f.write(script)
    print('created: runs/run_squad_eval.pbs')

