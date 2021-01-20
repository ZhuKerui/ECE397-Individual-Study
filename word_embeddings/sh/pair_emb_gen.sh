#! /bin/bash

if [ ! -d "./data/outputs/pair_emb/" ]
then
    mkdir ./data/outputs/pair_emb/
fi

python3 ./relation_similar/pair_emb.py ./data/corpus/big_key.vocab ./data/outputs/co_occur/big_co_occur.csv ./data/corpus/big_reform.txt ./data/outputs/pair_emb/big_pair_train
cd ./data/tools/
./count_and_filter -train ../outputs/pair_emb/big_pair_train -wvocab ../outputs/pair_emb/pwv -cvocab ../outputs/pair_emb/pcv -min-count 5
./word2vecf -train ../outputs/pair_emb/big_pair_train -wvocab ../outputs/pair_emb/pwv -cvocab ../outputs/pair_emb/pcv -output ../outputs/pair_emb/big_pair_emb.txt -size 250 -negative 20 -binary 0 -threads 20 -iter 10