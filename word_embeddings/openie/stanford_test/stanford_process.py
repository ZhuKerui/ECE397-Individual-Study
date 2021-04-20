import sys

ret = []
with open(sys.argv[1], 'r') as f_in:
    threshold = float(sys.argv[2])
    for line in f_in:
        content = line.strip().split('\t')
        if len(content) != 4:
            continue
        score = float(content[0])
        if score < threshold:
            continue
        ret.append('%s; %s; %s' % (content[1], content[2], content[3]))
        
with open(sys.argv[3], 'w') as f_out:
    f_out.write('\n'.join(ret))