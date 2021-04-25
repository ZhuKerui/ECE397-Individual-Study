# python sent_tokenize.py sent_file_in sent_file_out
import sys
from nltk import sent_tokenize

content = []
with open(sys.argv[1], 'r') as f_in:
    for line in f_in:
        content += sent_tokenize(line)

with open(sys.argv[2], 'w') as f_out:
    f_out.write('\n'.join(content))