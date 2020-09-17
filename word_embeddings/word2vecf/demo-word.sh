make
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi
time 
./count_and_filter -train context.txt -cvocab cv -wvocab wv -min-count 100
./word2vecf -train context.txt -wvocab wv -cvocab cv -output word_embedding -size 200 -negative 15 -threads 10
./distance word_embedding