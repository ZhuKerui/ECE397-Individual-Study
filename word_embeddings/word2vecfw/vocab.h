#ifndef _vocab_h
#define _vocab_h

#define MAX_STRING (50)

struct vocab_word {
  int64_t cn;
  char *word;
};

struct vocabulary {
   struct vocab_word *vocab;
   int32_t *vocab_hash;
   int64_t vocab_max_size; //1000
   int32_t vocab_size;
   int64_t word_count;
};


int32_t ReadWordIndex(struct vocabulary *v, FILE *fin);
int32_t GetWordHash(struct vocabulary *v, char *word);
int32_t SearchVocab(struct vocabulary *v, char *word);
int32_t AddWordToVocab(struct vocabulary *v, char *word);
void SortAndReduceVocab(struct vocabulary *v, int32_t min_count);
struct vocabulary *CreateVocabulary();
void SaveVocab(struct vocabulary *v, char *vocab_file);
struct vocabulary *ReadVocab(char *vocab_file);
void EnsureVocabSize(struct vocabulary *v);

#endif
