// TODO: add total word count to vocabulary, instead of "train_words"
//
// Modifed by Yoav Goldberg, Jan-Feb 2014
// Removed:
//    hierarchical-softmax training
//    cbow
// Added:
//   - support for different vocabularies for words and contexts
//   - different input syntax
//
/////////////////////////////////////////////////////////////////
//
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

#include "word2vecf_lib.h"

long long GetFileSize(char *fname) {
  long long fsize;
  FILE *fin = fopen(fname, "rb");
  if (fin == NULL) {
    printf("ERROR: file not found! %s\n", fname);
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  fsize = ftell(fin);
  fclose(fin);
  return fsize;
}

void InitUnigramTable(struct vocabulary *v, int **unitable, int table_size) {
  int a, i;
  long long normalizer = 0;
  real d1, power = 0.75;
  (*unitable) = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < v->vocab_size; a++) normalizer += pow(v->vocab[a].cn, power);
  i = 0;
  d1 = pow(v->vocab[i].cn, power) / (real)normalizer;
  for (a = 0; a < table_size; a++) {
    (*unitable)[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(v->vocab[i].cn, power) / (real)normalizer;
    }
    if (i >= v->vocab_size) i = v->vocab_size - 1;
  }
}

void InitNet(struct vocabulary *wv, struct vocabulary *cv, real **word_vecs, real **context_vecs, long long layer1_size) {
  long long a, b;
  a = posix_memalign((void **)word_vecs, 128, (long long)wv->vocab_size * layer1_size * sizeof(real));
  if (*word_vecs == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (b = 0; b < layer1_size; b++) 
    for (a = 0; a < wv->vocab_size; a++)
        (*word_vecs)[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;

  a = posix_memalign((void **)context_vecs, 128, (long long)cv->vocab_size * layer1_size * sizeof(real));
  if (*context_vecs == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (b = 0; b < layer1_size; b++)
    for (a = 0; a < cv->vocab_size; a++)
      (*context_vecs)[a * layer1_size + b] = 0;
}

void InitRelationWeight(struct vocabulary *rv, real **relation_weight){
  long long a;
  a = posix_memalign((void **)relation_weight, 128, (long long)rv->vocab_size * sizeof(real));
  if (*relation_weight == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (a = 0; a < rv->vocab_size; a++)
    (*relation_weight)[a] = rand() / (real)RAND_MAX;
}

void dumpVec(char *dump_file, struct vocabulary *v, real *vecs, long long vec_dim, int binary){
  FILE *f = fopen(dump_file, "wb");
  fprintf(f, "%d %d\n", v->vocab_size, vec_dim);
  for (long a = 0; a < v->vocab_size; a++) {
      fprintf(f, "%s ", v->vocab[a].word);
      if (binary) for (long b = 0; b < vec_dim; b++) fwrite(&vecs[a * vec_dim + b], sizeof(real), 1, f);
      else for (long b = 0; b < vec_dim; b++) fprintf(f, "%lf ", vecs[a * vec_dim + b]);
      fprintf(f, "\n");
  }
  fclose(f);
}

void dumpKMean(long long classes, char *dump_file, struct vocabulary *wv, real *word_vecs, long long layer1_size){
  // Run K-means on the word vectors
  long a, b, c, d;
  int clcn = classes, iter = 10, closeid;
  int *centcn = (int *)malloc(classes * sizeof(int));
  int *cl = (int *)calloc(wv->vocab_size, sizeof(int));
  real closev, x;
  real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
  for (a = 0; a < wv->vocab_size; a++) cl[a] = a % clcn;
  for (a = 0; a < iter; a++) {
    printf("kmeans iter %d\n", a);
    for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
    for (b = 0; b < clcn; b++) centcn[b] = 1;
    for (c = 0; c < wv->vocab_size; c++) {
      for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += word_vecs[c * layer1_size + d];
      centcn[cl[c]]++;
    }
    for (b = 0; b < clcn; b++) {
      closev = 0;
      for (c = 0; c < layer1_size; c++) {
        cent[layer1_size * b + c] /= centcn[b];
        closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
      }
      closev = sqrt(closev);
      for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
    }
    for (c = 0; c < wv->vocab_size; c++) {
      closev = -10;
      closeid = 0;
      for (d = 0; d < clcn; d++) {
        x = 0;
        for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * word_vecs[c * layer1_size + b];
        if (x > closev) {
          closev = x;
          closeid = d;
        }
      }
      cl[c] = closeid;
    }
  }
  // Save the K-means classes
  FILE *fo = fopen(dump_file, "wb");
  for (a = 0; a < wv->vocab_size; a++) fprintf(fo, "%s %d\n", wv->vocab[a].word, cl[a]);
  fclose(fo);
  free(centcn);
  free(cent);
  free(cl);
}

int loadVec(char *fname, long long *words, long long *size, char **vocab, float **vecs, int normalize){
  long long a, b, word_num, vec_dim;
  char ch;
  float len;
  float *word_embed;
  char *word_vocab;

  FILE *f = fopen(fname, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }

  fscanf(f, "%lld", &word_num);
  fscanf(f, "%lld", &vec_dim);
  (*words) = word_num;
  (*size) = vec_dim;

  word_embed = (float *)malloc(word_num * vec_dim * sizeof(float));
  if (word_embed == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", word_num * vec_dim * sizeof(float) / 1048576, word_num, vec_dim);
    fclose(f);
    return -1;
  }
  (*vecs) = word_embed;

  word_vocab = (char *)malloc(word_num * MAX_STRING * sizeof(char));
  if (word_vocab == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", word_num * MAX_STRING * sizeof(char) / 1048576, word_num, MAX_STRING);
    fclose(f);
    return -1;
  }
  (*vocab) = word_vocab;

  for (b = 0; b < word_num; b++) {
    fscanf(f, "%s%c", &word_vocab[b * MAX_STRING], &ch);
    for (a = 0; a < vec_dim; a++) fread(&word_embed[a + b * vec_dim], sizeof(float), 1, f);
    if (normalize){
      len = 0;
      for (a = 0; a < vec_dim; a++) len += word_embed[a + b * vec_dim] * word_embed[a + b * vec_dim];
      len = sqrt(len);
      for (a = 0; a < vec_dim; a++) word_embed[a + b * vec_dim] /= len;
    }
  }
  fclose(f);
  return 0;
}

void grab_relation(char *relation, char *word){
  int a;
  strcpy(relation, word);
  for (a = 0; a < MAX_STRING; a++){
    if (relation[a] == 0) return;
    if (relation[a] == '_'){
      relation[a] = 0;
      if (!strcmp(relation, "prep")){
        // If the relation is 'prep', continue to include the preposiiton
        relation[a] = '_';
      }else{
        return;
      }
    }
  }
  relation[a-1] = 0;
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}
