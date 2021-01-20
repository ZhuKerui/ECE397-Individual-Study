#! /bin/bash

if [ ! -d "./data/outputs/dependency_emb/" ]
then
    mkdir ./data/outputs/dependency_emb/
fi

python3 ./relation_similar/dep_generator.py ./data/corpus/big_key.vocab ./data/corpus/big_reform.txt ./data/outputs/dependency_emb/big_dep_train
cd ./data/tools/
./count_and_filter -train ../outputs/dependency_emb/big_dep_train -wvocab ../outputs/dependency_emb/wv -cvocab ../outputs/dependency_emb/cv -min-count 10
./word2vecf -train ../outputs/dependency_emb/big_dep_train -wvocab ../outputs/dependency_emb/wv -cvocab ../outputs/dependency_emb/cv -output ../outputs/dependency_emb/big_dep_emb.txt -size 200 -negative 15 -binary 0 -threads 20 -iter 10