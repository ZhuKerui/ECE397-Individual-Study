from pyopenie import OpenIE5
import sys


result = []
extractor = OpenIE5('http://localhost:9002')
with open(sys.argv[1], 'r') as f_in:
    for line in f_in:
        if not line.strip():
            continue
        result.append(line.strip())
        try:
            extractions = extractor.extract(line)
        except:
            print(line)
            continue
        for e in extractions:
            if not e['extraction']['arg2s']:
                continue
            confidence = e['confidence']
            subject = e['extraction']['arg1']['text']
            relation = e['extraction']['rel']['text']
            object = e['extraction']['arg2s'][0]['text']
            result.append('%f; %s; %s; %s' % (confidence, subject, relation, object))
        result.append('')
with open(sys.argv[2], 'w') as f_out:
    f_out.write('\n'.join(result))

