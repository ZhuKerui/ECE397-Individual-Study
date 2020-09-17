CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
#CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS = -lm -pthread -O0 -march=native -Wall -funroll-loops -Wno-unused-result -g

all: distance word2vecf count_and_filter

count_and_filter: count_and_filter.c vocab.c io.c
	$(CC) vocab.c count_and_filter.c io.c -o count_and_filter $(CFLAGS)
word2vecf : word2vecf.c vocab.c io.c
	$(CC) word2vecf.c vocab.c io.c -o word2vecf $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf count_and_filter word2vecf distance
