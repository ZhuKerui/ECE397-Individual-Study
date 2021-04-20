import sys

vocab = set(open(sys.argv[1]).read().strip().split())
with open(sys.argv[2]) as kw_in:
    possible_kw = []
    for line in kw_in:
        bad = False
        for word in line.split():
            if word not in vocab:
                bad = True
                break
        if not bad:
            possible_kw.append(line)
    with open(sys.argv[3], 'w', encoding='utf-8') as kw_out:
        kw_out.write(''.join(possible_kw))