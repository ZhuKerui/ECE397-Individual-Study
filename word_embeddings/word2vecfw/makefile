CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
#CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result -g

all: distance weight_train word2vecfw count_and_filter

count_and_filter: count_and_filter.c vocab.c io.c word2vecf_lib.c
	$(CC) vocab.c count_and_filter.c io.c word2vecf_lib.c -o count_and_filter $(CFLAGS)
weight_train : weight_train.c word2vecf_lib.c vocab.c io.c
	$(CC) weight_train.c word2vecf_lib.c vocab.c io.c -o weight_train $(CFLAGS)
word2vecfw : word2vecfw.c word2vecf_lib.c vocab.c io.c
	$(CC) word2vecfw.c word2vecf_lib.c vocab.c io.c -o word2vecfw $(CFLAGS)
distance : distance.c word2vecf_lib.c
	$(CC) distance.c word2vecf_lib.c -o distance $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf count_and_filter word2vecfw weight_train distance
