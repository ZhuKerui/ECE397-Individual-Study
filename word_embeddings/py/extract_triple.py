# python extract_triple.py openie_result_file triple_file_out
import sys

triple = []
with open(sys.argv[1], 'r') as f_in:
    new_line = True
    for line in f_in:
        if new_line:
            new_line = False
        elif not line.strip():
            new_line = True
        else:
            triple.append(line)

with open(sys.argv[2], 'w') as f_out:
    f_out.write(''.join(triple))