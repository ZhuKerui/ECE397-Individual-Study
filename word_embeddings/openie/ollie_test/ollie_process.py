import sys

ret = []
with open(sys.argv[1], 'r') as f_in:
    new_sent = True
    threshold = float(sys.argv[2])
    for line in f_in:
        line = line.strip()
        if new_sent:
            new_sent = False
            ret.append(line)
        elif not line:
            new_sent = True
            ret.append('')
        elif line.split()[0] == 'No':
            continue
        else:
            score, content = line.split(':', 1)
            score = float(score)
            if score < threshold:
                continue
            content = content.strip(' ()').split(';')
            if len(content) != 3:
                continue
            ret.append('%s;%s;%s' % tuple(content))

with open(sys.argv[3], 'w') as f_out:
    f_out.write('\n'.join(ret))
