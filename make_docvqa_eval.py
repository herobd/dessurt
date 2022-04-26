template= """#!/bin/bash

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -J "eval {0} docvqa"
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
echo "VALID {0}" 
python qa_eval.py -d DocVQA -c saved/{0}/model_best.pth -g 0 -v 0
#THIS IS VALIDATION
"""




import sys
if len(sys.argv[1:])>0:
    jobs=sys.argv[1:]
else:
    with open('tmp') as f:
        jobs = f.read().split('\n')
for job_name in jobs:
    script = template.format(job_name)
    script+=do_template.format(job_name)
    file_name = 'runs/run_docVQA_eval_{}.pbs'.format(job_name)
    with open(file_name,'w') as f:
        f.write(script)
        print('created: {}'.format(file_name))

