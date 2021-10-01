template= """#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks={}
#SBATCH --gpus-per-task=1
#SBATCH  --cpus-per-task=4
#SBATCH -J "{}"
#SBATCH --mem-per-cpu=4G
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
source activate /fslhome/brianld/miniconda3/envs/new


export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth,ib

set -eu
pids=()

for rank in [[0..{}]]
do
    srun -N 1 -n 1 --gpus 1 --exclusive  python  train.py --supercomputer -c configs/cf_{}.json -s saved/{}/checkpoint-latest.pth --rank $rank --worldsize 4 &
    pids+=($!)
    if [ $rank==0 ]; then
        sleep 2.5
    fi
done

sleep 120

SLEEP_TIME=30

while (( ${#pids[@]} )); do
  for pid_idx in "${!pids[@]}"; do
    pid=${pids[$pid_idx]}
    if ! kill -0 "$pid" 2>/dev/null; then # kill -0 checks for process existance
      # we know this pid has exited; retrieve its exit status
      echo "ended $pid"
      wait "$pid" || exit 1
      unset "pids[$pid_idx]"
    fi
  done
  sleep $SLEEP_TIME # in bash, consider a shorter non-integer interval, ie. 0.2
  if (( SLEEP_TIME<600 )); then
    #max of 5 minute sleep
    SLEEP_TIME=$((SLEEP_TIME+30))
  fi
done

"""

import sys
N=4
def create(job_name):
    script = template.format(N,job_name,N-1,job_name,job_name,job_name,job_name)
    script=script.replace('[[','{')
    script=script.replace(']]','}')

    with open('runs/run_{}.pbs'.format(job_name),'w') as f:
        f.write(script)
        print('created: runs/run_{}.pbs'.format(job_name))

for job_name in sys.argv[1:]:
    create(job_name)

