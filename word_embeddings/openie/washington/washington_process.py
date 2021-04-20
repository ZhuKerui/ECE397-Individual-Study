import sys

ret = []
with open(sys.argv[1], 'r') as f_in:
    threshold = float(sys.argv[2])
    new_sent = True
    for line in f_in:
        line = line.strip()
        if new_sent:
            new_sent = False
            ret.append(line)
        elif not line:
            new_sent = True
            ret.append('')
        else:
            content = line.split(';')
            if len(content) != 4:
                continue
            score = float(content[0])
            if score < threshold:
                continue
            ret.append('%s; %s; %s' % (content[1].strip(), content[2].strip(), content[3].strip()))
        
with open(sys.argv[3], 'w') as f_out:
    f_out.write('\n'.join(ret))