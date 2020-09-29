./count_and_filter -train ../../../dataset/dep_context.txt -cvocab cv -wvocab wv -rvocab rv -min-count 100
./word2vecfw -train ../../../dataset/dep_context.txt -wvocab wv -cvocab cv -output word_embedding.bin -binary 1 -size 200 -negative 15 -threads 10
./distance word_embedding.bin

For test:
./count_and_filter -train ../../../dataset/sub_dep_context.txt -cvocab cv_t -wvocab wv_t -rvocab rv_t -min-count 10
./weight_train -train ../../../dataset/weight_train_file.txt -wvocab wv_t -cvocab cv_t -rvocab rv_t -rweight rw_t -output word_embedding.bin -iters 100 -binary 0 -size 200 -negative 15 -threads 1