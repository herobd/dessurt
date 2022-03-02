import json
import os
import sys
from change_checkpoint_reset_for_training import readRemoveWrite
from make_run import create

if len(sys.argv)==1:
    print('pretrained-checkpoint newID dirLocation [user]')
    exit()

pretrained_path=sys.argv[1]
#itera=int(sys.argv[2])
new_id=int(sys.argv[2])
new_dir_loc=sys.argv[3]
if len(sys.argv)>4:
    user = sys.argv[4]
else:
    user = None

path = pretrained_path.split('/')
name = path[-2]
file_name = path[-1]
iter_loc = file_name.find('iteration')
if iter_loc == -1:
    itera='latest'
else:
    itera = int(file_name[iter_loc+9:-4])
    if itera<1000000:
        itera = '{}k'.format( round(itera/1000))
    elif itera==1000000:
        itera = '1M'
    else:
        itera = '{:.2}m'.format( itera/1000000)
print('from '+itera)


broken = name.split('_')
old_id = broken[-1]
new_name='iamNER18c1noise_'+('_'.join(broken[1:-2]))+'_PTfrom{}i{}_{}'.format(old_id,itera,new_id)
print(new_name)

old_cf = 'configs/cf_{}.json'.format(name)
new_cf = 'configs/cf_{}.json'.format(new_name)

destination = os.path.join(new_dir_loc,new_name)
os.mkdir(destination)
readRemoveWrite(pretrained_path,destination)



with open(old_cf) as f:
    cf = json.load(f)

image_size = cf['model']['image_size']

new_dataset= {
        "data_set_name": "IAMNER",
        "data_dir": "../data/IAM",
        "cased": True,
        "full": True,
        "class_first": True,
        "use_noise": 0.09,
        "task": 18,
        "batch_size": 1,
        "num_workers": 6,
        "shuffle": True,
        "persistent_workers": True,
        "rescale_range": [0.8,1],
        "rescale_to_crop_size_first": True,
        "crop_params": {
            "crop_size": image_size,
            "pad": 0,
            "rot_degree_std_dev": 1
        },
        "questions": 1
            }
new_val =  {
        "shuffle": False,
        "batch_size": 3,
        "rescale_range": [0.9,0.9],
        "crop_params": {
            "crop_size": image_size,
            "pad": 0,
            "random": False
        }
    }

cf['name']=new_name
cf['data_loader']=new_dataset

cf['validation']=new_val

cf['model']['max_a_tokens'] = 740  #full 800 not required

#set validation
cf['trainer']['iterations']=300099
cf['trainer']['val_step']=10000
cf['trainer']['save_step']=14000000000
cf['trainer']["save_step_minor"]= 1024 
cf['trainer']['monitor_mode']='max'
cf['trainer']['monitor']='val_F_Measure_MACRO'

#set drop in LR
cf['trainer']["use_learning_schedule"]= "multi_rise then ramp_to_lower"
cf['trainer']["lr_down_start"]= 90000
cf['trainer']["ramp_down_steps"]= 10000
cf['trainer']["lr_mul"]= 0.1

if user is None or user=='brianld':
    cf['trainer']['save_dir']='saved/'
    if 'seperate_log_at' in cf['trainer']:
        del cf['trainer']['seperate_log_at']
else:
    cf['trainer']['save_dir']='/fslhome/{}/saved/'.format(user)
    cf['trainer']['seperate_log_at']='logs'


with open(new_cf,'w') as f:
    json.dump(cf,f,indent=4)
print(new_cf)
create(new_name)