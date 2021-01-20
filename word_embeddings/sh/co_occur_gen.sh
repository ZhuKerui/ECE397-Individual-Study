#! /bin/bash

if [ ! -d "./data/outputs/co_occur/" ]
then
    mkdir ./data/outputs/co_occur/
fi

python3 ./relation_similar/co_occur_generator.py ./data/corpus/big_key.vocab ./data/corpus/big_reform.txt ./data/outputs/co_occur/big_co_occur.csv