./count_and_filter -train ../../../dataset/dep_context.txt -cvocab cv -wvocab wv -rvocab rv -min-count 100
./word2vecfw -train ../../../dataset/dep_context.txt -wvocab wv -cvocab cv -output word_embedding.bin -binary 1 -size 200 -negative 15 -threads 10
./distance word_embedding.bin