from openie import StanfordOpenIE
from time import time
import sys

relation_list = []
# with StanfordOpenIE(core_nlp_version = '4.2.0', install_dir_path = '/scratch/similar_relation/corenlp/') as client:
with StanfordOpenIE() as client:
    with open(sys.argv[1], 'r') as file_in:
        start_time = time()
        for line in file_in:
            line = line.strip()
            relation_list.append(line)
            for triple in client.annotate(line):
                sub = triple['subject']
                rel = triple['relation']
                obj = triple['object']
                if ';' not in sub and ';' not in rel and ';' not in obj:
                    relation_list.append('; '.join((sub, rel, obj)))
            relation_list.append('')
        print(time() - start_time)
        
with open(sys.argv[2], 'w', encoding='utf-8') as file_o:
    file_o.write('\n'.join(relation_list))


