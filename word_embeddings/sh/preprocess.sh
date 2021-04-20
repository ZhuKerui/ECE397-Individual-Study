if [ -e data/triple.txt ]; then
    rm data/triple.txt
fi
if [ -e data/vocab.txt ]; then
    rm data/vocab.txt
fi
rm data/train_data*

cp data/stanford_triple.txt data/triple.txt
# echo '' >> data/triple.txt
# cat data/ollie_triple.txt >> data/triple.txt
python embeddings/preprocess.py data/triple.txt data/vocab.txt 10 data/train_data.txt data/train_data
