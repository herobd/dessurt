import sys
THRESHOLD=21
min_dist=99
with open('../data/fonts/clear_fonts.csv','w') as out:
    out.write('path,hasLower,hasNum\n')
    for infile in sys.argv[1:]:
        with open(infile) as f:
            entries = f.readlines()
        for line in entries:
            data = line.strip().split(',')
            #ugh, should have properly excaped those ,s
            dist = int(data[0])
            index = data[1]
            filepath = ','.join(data[2:])
            if dist<THRESHOLD:
                out.write('{},True,True\n'.format(filepath))
            if dist<min_dist:
                min_dist = dist

print('min dist: {}'.format(min_dist))

