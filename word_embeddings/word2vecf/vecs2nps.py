import numpy as np
import sys
import io

fh=io.open(sys.argv[1], 'r', encoding='utf-8')
foutname=sys.argv[2]
first=fh.readline()
size=map(int,first.strip().split())

wvecs=np.zeros((size[0],size[1]),float)

vocab=[]
for i,line in enumerate(fh):
    line = line.strip().split()
    vocab.append(line[0])
    wvecs[i,] = np.array(map(float,line[1:]))

np.save(foutname+".npy",wvecs)
with io.open(foutname+".vocab", "w", encoding='utf-8') as outf:
   outf.write(" ".join(vocab))
