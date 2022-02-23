import json
import sys

new_dataset= {
                "freq": 1.5,
                "config": {
                    "data_set_name": "SQuAD",
                    "data_dir": "../data/SQuAD",
                    "rescale_range": [
                        0.9,
                        1.1
                    ],
                    "crop_params": {
                        "crop_size": [
                            1152,768
                        ],
                        "pad": 0,
                        "rot_degree_std_dev": 1
                    },
                    "questions": 1,
                    "image_size": [1150,760],
                    "cased": True
                }
            }
new_val =  {
        "shuffle": False,
        "batch_size": 3,
        "data_set_name": "MultipleDataset",
        "datasets": [
            {
                "freq": 1,
                "config": {
                    "data_set_name": "SQuAD",
                    "data_dir": "../data/SQuAD",
                    "rescale_range": [
                        1.0,
                        1.0
                    ],
                    "crop_params": {
                        "crop_size": [
                            1152,768
                        ],
                        "pad": 0,
                        "random": False
                    },
                    "questions": 1,
                    "image_size": [1150,760],
                    "cased": True
                }
            }

        ]
    }



name=sys.argv[1]
new_id=int(sys.argv[2])

broken = name.split('_')
old_id = broken[-1]
new_name='squad+'+('_'.join(broken[:-2]))+'_PTfrom{}i800k_{}'.format(old_id,new_id)

old_cf = 'configs/cf_{}.json'.format(name)
new_cf = 'configs/cf_{}.json'.format(new_name)



with open(old_cf) as f:
    cf = json.load(f)


cf['name']=new_name
cf['data_loader']['datasets'] = [new_dataset]+cf['data_loader']['datasets']

cf['validation']=new_val

cf['trainer']['iterations']=200000
cf['trainer']['val_step']=25000
cf['trainer']['save_step']=100000


with open(new_cf,'w') as f:
    json.dump(cf,f,indent=4)
print(new_cf)
