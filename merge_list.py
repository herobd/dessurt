import sys 

main_file = sys.argv[1]
new_file = sys.argv[2]

all_url={}
with open(main_file) as f:
    main = f.readlines()
    for line in main:
        tar,url = line.split(',')
        all_url[tar.strip()]=url.strip()

with open(new_file) as f:
    new = f.readlines()
    for line in new:
        tar,url = line.split(',')
        all_url[tar.strip()]=url.strip()


with open(main_file,'w') as f:
    for tar,url in all_url.items():
        f.write(tar+','+url+'\n')
        
