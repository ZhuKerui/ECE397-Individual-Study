#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "vocab.h"
#include "io.h"

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define RELATION_NUM 30

typedef float real;                    // Precision of float numbers

#define N 5                  // number of closest words that will be shown

int64_t GetFileSize(char *fname);


// Used for sampling of negative examples.
// wc[i] == the count of context number i
// wclen is the number of entries in wc (context vocab size)
void InitUnigramTable(struct vocabulary *v, int32_t **unitable, int32_t table_size);

void InitNet(struct vocabulary *wv, struct vocabulary *cv, real **word_vecs, real **context_vecs, int64_t layer1_size);

void InitRelationWeight(struct vocabulary *rv, real **relation_weight);

void dumpVec(char *dump_file, struct vocabulary *v, real *vecs, int64_t layer1_size, int32_t binary);

void dumpKMean(int64_t classes, char *dump_file, struct vocabulary *wv, real *word_vecs, int64_t layer1_size);

int32_t loadVec(char *fname, int64_t *words, int64_t *size, char **vocab, float **vecs, int32_t normalize);

void grab_relation(char *relation, char *word);

int32_t ArgPos(char *str, int32_t argc, char **argv);
