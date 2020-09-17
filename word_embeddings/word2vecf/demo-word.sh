make
time 
./count_and_filter -train ../../../dataset/dep_context.txt -cvocab cv -wvocab wv -min-count 20
./word2vecf -train ../../../dataset/dep_context.txt -wvocab wv -cvocab cv -output word_embedding.bin -size 200 -negative 15 -threads 10 -binary 1 -iters 3
./distance word_embedding.bin
