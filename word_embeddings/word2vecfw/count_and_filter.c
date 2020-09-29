//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "word2vecf_lib.h"
#include "vocab.h"
#include "io.h"


typedef float real;                    // Precision of float numbers


char train_file[MAX_STRING], output_file[MAX_STRING], cvocab_file[MAX_STRING], wvocab_file[MAX_STRING], rvocab_file[MAX_STRING];
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, use_position = 0;
long long layer1_size = 100;
long long train_words = 0, word_count_actual = 0, classes = 0, dumpcv = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  char context[MAX_STRING];
  char relation[MAX_STRING];
  FILE *fin;// = stdin;
  long long a, i;
  struct vocabulary *wv = CreateVocabulary();
  struct vocabulary *cv = CreateVocabulary();
  struct vocabulary *rv = CreateVocabulary();
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  while (1) {
    ReadWord(word, fin, MAX_STRING);
    ReadWord(context, fin, MAX_STRING);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 1000000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(wv,word);
    if (i == -1) {
      a = AddWordToVocab(wv,word);
      wv->vocab[a].cn = 1;
    } else wv->vocab[i].cn++;

    i = SearchVocab(cv,context);
    if (i == -1) {
      a = AddWordToVocab(cv,context);
      cv->vocab[a].cn = 1;
    } else cv->vocab[i].cn++;

    grab_relation(relation, context);
    i = SearchVocab(rv,relation);
    if (i == -1) {
      a = AddWordToVocab(rv,relation);
      rv->vocab[a].cn = 1;
    } else rv->vocab[i].cn++;

    EnsureVocabSize(wv);
    EnsureVocabSize(cv);
    EnsureVocabSize(rv);
  }
  SortAndReduceVocab(wv,min_count);
  SortAndReduceVocab(cv,min_count);
  SortAndReduceVocab(rv,min_count);
  printf("WVocab size: %lld\n", wv->vocab_size);
  printf("CVocab size: %lld\n", cv->vocab_size);
  printf("RVocab size: %lld\n", rv->vocab_size);
  printf("Words in train file: %lld\n", train_words);
  fclose(fin);
  SaveVocab(wv, wvocab_file);
  SaveVocab(cv, cvocab_file);
  SaveVocab(rv, rvocab_file);
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-wvocab <filename>\n");
    printf("\t\tword-vocabulary filename.\n");
    printf("\t-cvocab <filename>\n");
    printf("\t\tcontext-vocabulary filename.\n");
    printf("\t-rvocab <filename>\n");
    printf("\t\trelation-vocabulary filename.\n");
    printf("\nExamples:\n");
    printf("./count_and_filter -train ../../../dataset/dep_context.txt -cvocab cv -wvocab wv -rvocab rv -min-count 100\n\n");
    return 0;
  }
  wvocab_file[0] = 0;
  cvocab_file[0] = 0;
  rvocab_file[0] = 0;
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-wvocab", argc, argv)) > 0) strcpy(wvocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-cvocab", argc, argv)) > 0) strcpy(cvocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-rvocab", argc, argv)) > 0) strcpy(rvocab_file, argv[i + 1]);
  if (cvocab_file[0] == 0 || wvocab_file[0] == 0 || rvocab_file[0] == 0) {
     printf("-cvocab or -wvocab or -rvocab argument is missing\n\n");
     return 0;
  };
  LearnVocabFromTrainFile();
  return 0;
}
