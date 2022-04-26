import json
import sys

new_dataset= {
                "freq": 0.75,
                "config": {
                    "data_set_name": "IAMMixed",
                    "data_dir": "../data/IAM",
                    "rescale_range": [0.9,1],
                    "image_size": [1150,760],
                    "crop_params": {
                        "crop_size": [
                            1152,
                            768
                        ],
                        "pad": 0,
                        "rot_degree_std_dev": 1
                    },
                    "questions": 1,
                    "max_qa_len_in": 640,
                    "max_qa_len_out": 99999999999999,
                    "cased": True
                }
            }
new_val =  {
        "shuffle": False,
        "batch_size": 3,
        "num_workers": 4,
        "datasets": [
            {
                "freq": 1,
                "config": {
                    "data_set_name": "IAMQA",
                    "data_dir": "../data/IAM",
                    "rescale_to_crop_size_first": True,
                    "mode": "IAM_valid",
                    "rescale_range": [0.9,0.9],
                    "crop_params": {
                        "crop_size": [
                            1152,
                            768
                        ],
                        "pad": 0,
                        "rot_degree_std_dev": 0,
                        "random": False
                    },
                    "questions": 1,
                    "max_qa_len_in": 640,
                    "max_qa_len_out": 9999999999,
                    "cased": True
                }
            }
        ]
    }


name=sys.argv[1]
new_id=int(sys.argv[2])

#new_name='iamX+'+name

broken = name.split('_')
old_id = broken[-1]
new_name='iamX+'+('_'.join(broken[:-2]))+'_PTfrom{}i3m_{}'.format(old_id,new_id)

old_cf = 'configs/cf_{}.json'.format(name)
new_cf = 'configs/cf_{}.json'.format(new_name)



with open(old_cf) as f:
    cf = json.load(f)


cf['name']=new_name
cf['data_loader']['datasets'] = [new_dataset]+cf['data_loader']['datasets']

cf['validation']=new_val

cf['trainer']['iterations']=250000
cf['trainer']['val_step']=25000
cf['trainer']['save_step']=200000


with open(new_cf,'w') as f:
    json.dump(cf,f,indent=4)
print(new_cf)
