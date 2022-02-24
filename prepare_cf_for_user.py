import json
import sys


user = sys.argv[1]
print('For user: '+user)
for name in sys.argv[2:]:
    if name.endswith('json'):
        cf_path = name
        name = cf_path[11:-5]
    else:
        cf_path = 'configs/cf_{}.json'.format(name)
   

  

    with open(cf_path) as f:
        cf = json.load(f)
    
    if user=='brianld':
        cf['trainer']['save_dir']='saved/'
        if 'seperate_log_at' in cf['trainer']:
            del cf['trainer']['seperate_log_at']
    else:
        cf['trainer']['save_dir']='/fslhome/{}/saved/'.format(user)
        cf['trainer']['seperate_log_at']='logs'

    with open(cf_path,'w') as f:
        json.dump(cf,f,indent=4)
    print(cf_path)
