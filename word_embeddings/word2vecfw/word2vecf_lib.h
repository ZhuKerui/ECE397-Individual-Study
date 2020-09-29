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

#define N 40                  // number of closest words that will be shown

long long GetFileSize(char *fname);


// Used for sampling of negative examples.
// wc[i] == the count of context number i
// wclen is the number of entries in wc (context vocab size)
void InitUnigramTable(struct vocabulary *v, int **unitable, int table_size);

void InitNet(struct vocabulary *wv, struct vocabulary *cv, real **word_vecs, real **context_vecs, long long layer1_size);

void InitRelationWeight(struct vocabulary *rv, real **relation_weight);

void dumpVec(char *dump_file, struct vocabulary *v, real *vecs, long long layer1_size, int binary);

void dumpKMean(long long classes, char *dump_file, struct vocabulary *wv, real *word_vecs, long long layer1_size);

int loadVec(char *fname, long long *words, long long *size, char **vocab, float **vecs, int normalize);

void grab_relation(char *relation, char *word);

int ArgPos(char *str, int argc, char **argv);
