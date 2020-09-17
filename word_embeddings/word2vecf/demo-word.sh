make
time 
./count_and_filter -train context.txt -cvocab cv -wvocab wv -min-count 10
./word2vecf -train context.txt -wvocab wv -cvocab cv -output word_embedding -size 200 -negative 15 -threads 10
./distance word_embedding
