template= """#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH  --cpus-per-task=4
#SBATCH -J "{}"
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=herobd@gmail.com   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -C pascal


export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

module remove miniconda3
module load cuda/10.1
module load cudnn/7.6
cd ~/pairing
source deactivate
source activate /fslhome/brianld/miniconda3/envs/new


export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth,ib
for rank in [[0..3]]
do
    srun -N 1 -n 1 --gpus 1 --exclusive  python  train.py --supercomputer -c configs/cf_{}.json -s saved/{}/checkpoint-latest.pth --rank $rank --worldsize 4 &
    if [ $rank==0 ]; then
        sleep 1.5
    fi
done

wait

"""

import sys

def create(job_name):
    script = template.format(job_name,job_name,job_name,job_name,job_name)
    script=script.replace('[[','{')
    script=script.replace(']]','}')

    with open('runs/run_{}.pbs'.format(job_name),'w') as f:
        f.write(script)
        print('created: runs/run_{}.pbs'.format(job_name))

for job_name in sys.argv[1:]:
    create(job_name)

