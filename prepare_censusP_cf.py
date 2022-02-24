import json
import sys

new_dataset= {
                "freq": 0.2,
                "config": {
                    "data_set_name": "CensusQA",
                    "data_dir": "../data/familysearch",
                    "rescale_range": [1,1.15],
                    "max_qa_len": 999999,
                    "questions":1,
                    "pretrain": True,
                    "rescale_to_crop_height_first": True,
                    "crop_params": {
                        "crop_size": [
                            1152,768
                        ],
                        "pad": 0,
                        "rot_degree_std_dev": 1,
                        "left": True
                    }
                }
            }
new_val =  {
        "shuffle": False,
        "batch_size": 3,
        "datasets": [
            {
                "freq": 1,
                "config": {
                    "data_set_name": "CensusQA",
                    "data_dir": "../data/familysearch",
                    "rescale_range": [1.1,1.1],
                    "max_qa_len": 999999,
                    "questions":1,
                    "pretrain": True,
                    "rescale_to_crop_height_first": True,
                    "crop_params": {
                        "crop_size": [
                            1152,768
                        ],
                        "pad": 0,
                        "rot_degree_std_dev": 1,
                        "left": True,
                        "random": False
                    }
                }
            }
        ]
    }



name=sys.argv[1]
new_id=int(sys.argv[2])

broken = name.split('_')
old_id = broken[-1]
new_name='censusP+'+('_'.join(broken[:-2]))+'_PTfrom{}i800k_{}'.format(old_id,new_id)

old_cf = 'configs/cf_{}.json'.format(name)
new_cf = 'configs/cf_{}.json'.format(new_name)



with open(old_cf) as f:
    cf = json.load(f)


cf['data_loader']['datasets'] = [new_dataset]+cf['data_loader']['datasets']

cf['validation']=new_val

cf['trainer']['iterations']=200000
cf['trainer']['val_step']=25000
cf['trainer']['save_step']=100000


with open(new_cf,'w') as f:
    json.dump(cf,f,indent=4)
print(new_cf)
