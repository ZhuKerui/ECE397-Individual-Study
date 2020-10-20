# Word2vecfw

> In the python console:

```bash
>>> from syntatic_dep import *
```

> In the System terminal:

```bash
$ make clean
$ make
```

## Weight Training

### Demo

> In the python console:

```bash
>>> from syntatic_dep import *
>>> gen_weight_training_data('../../dataset/dep_context.txt', '../../dataset/test_file/weight_train_file.txt', {'time complexity', 'space complexity', 'decoding complexity', 'hardware complexity', 'sample complexity'}, {'java', 'python', 'matlab', 'c++', 'javascript'})
```

> In the System terminal:

```bash
$ ./count_and_filter -train ../../../dataset/test_file/weight_train_file.txt -cvocab cv -wvocab wv -rvocab rv -min-count 100
$ ./weight_train -train ../../../dataset/test_file/weight_train_file.txt -wvocab wv -cvocab cv -rvocab rv -rweight rw -output word_embedding.bin -iters 100 -binary 1 -size 200 -negative 15 -threads 10
# For testing
$ ./weight_train -train ../../../dataset/test_file/weight_train_file.txt -wvocab wv -cvocab cv -rvocab rv -rweight rw -output word_embedding.txt -iters 800 -binary 0 -size 200 -negative 15 -threads 10 -batch 200
```

### Debug

> In the python console:

```bash
>>> gen_weight_training_data('../../dataset/dep_context.txt', '../../dataset/test_file/weight_train_file.txt', {'complexity'}, {'java'})
```

> In the System terminal:

```bash
$ ./count_and_filter -train ../../../dataset/test_file/weight_train_file.txt -cvocab cv_t -wvocab wv_t -rvocab rv_t -min-count 1
$ ./weight_train -train ../../../dataset/test_file/weight_train_file.txt -wvocab wv_t -cvocab cv_t -rvocab rv_t -rweight rw_t -output word_embedding.txt -iters 2 -binary 0 -size 10 -negative 1 -threads 1
```

## Weighted Dependency-Based Word Embedding

### Demo

> In the System terminal:

```bash
$ ./count_and_filter -train ../../../dataset/dep_context.txt -cvocab cv -wvocab wv -rvocab rv -min-count 100
$ ./word2vecfw -train ../../../dataset/dep_context.txt -wvocab wv -cvocab cv -rvocab rv -rweight rw -output word_embedding.bin -iters 100 -binary 1 -size 200 -negative 15 -threads 10
```

### Debug

> In the python console:

```bash
>>> extract_pairs('../../dataset/dep_context.txt', '../../dataset/sub_dep_context.txt', {'complexity', 'java'})
```

> In the System terminal:

```bash
$ ./count_and_filter -train ../../../dataset/sub_dep_context.txt -cvocab cv_t -wvocab wv_t -rvocab rv_t -min-count 1
$ ./word2vecfw -train ../../../dataset/sub_dep_context.txt -wvocab wv_t -cvocab cv_t -rvocab rv_t -rweight rw_t -output word_embedding.bin -iters 2 -binary 0 -size 20 -negative 1 -threads 1
```
