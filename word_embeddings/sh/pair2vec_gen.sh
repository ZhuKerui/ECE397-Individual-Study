if [ ! -d ./data/outputs/pair2vec/ ]
then
    mkdir ./data/outputs/pair2vec/
fi

python3 ./relation_similar/pair_generator.py ./data/corpus/big_key.vocab ./data/corpus/big_ctx.vocab ./data/corpus/big_reform.txt ./data/outputs/pair2vec/big_triplet