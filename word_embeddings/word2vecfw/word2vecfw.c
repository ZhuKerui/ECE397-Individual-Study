#include "word2vecf_lib.h"

char train_file[MAX_STRING];
char wvocab_file[MAX_STRING], cvocab_file[MAX_STRING], rvocab_file[MAX_STRING];
char output_file[MAX_STRING], dumpcv_file[MAX_STRING];
char rweight_file[MAX_STRING];

int binary = 0, debug_mode = 2, num_threads = 1;
long long layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *word_vecs, *context_vecs, *expTable;
clock_t start;
int numiters = 1;

struct vocabulary *wv;
struct vocabulary *cv;
struct vocabulary *rv;

real *relation_weight;

int negative = 15;
const int table_size = 1e8;
int *unitable;

// Read word,context pairs from training file, where both word and context are integers.
// We are learning to predict context based on word.
//
// Word and context come from different vocabularies, but we do not really care about that
// at this point.
void *TrainModelThread(void *id) {
  int ctxi = -1, wrdi = -1, rlti = -1; // ctxi: context index, wrdi: word index, rlti: relation index
  long long d;
  long long word_count = 0, last_word_count = 0;
  long long l1, l2, c, target, label;
  unsigned long long next_random = (unsigned long long)id;
  char relation[MAX_STRING];
  real f, g, weight;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb"); // fi: The input training file. 
                                      // For word embedding, there are 2 elements each line: target_word, relation_context
  long long start_offset = file_size / (long long)num_threads * (long long)id; // The starting point in the file for this thread
  long long end_offset = file_size / (long long)num_threads * (long long)(id+1); // The ending point in the file for this thread
  int iter;
  //printf("thread %d %lld %lld \n",id, start_offset, end_offset);

  /* The training process begins here */
  for (iter=0; iter < numiters; ++iter) {
    fseek(fi, start_offset, SEEK_SET);
    
    while (fgetc(fi) != '\n') { }; //TODO make sure its ok
    printf("thread %d %lld\n", id, ftell(fi));

    long long train_words = wv->word_count;
    while (1) { //HERE @@@
      // TODO set alpha scheduling based on number of examples read.
      // The conceptual change is the move from word_count to pair_count
      if (word_count - last_word_count > 10000) {
        word_count_actual += word_count - last_word_count;
        last_word_count = word_count;
        if ((debug_mode > 1)) {
          now=clock();
          printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                word_count_actual / (real)(numiters*train_words + 1) * 100,
                word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
          fflush(stdout);
        }
        // Update the learning rate according to the training process
        alpha = starting_alpha * (1 - word_count_actual / (real)(numiters*train_words + 1));
        if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
      }
      if (feof(fi) || ftell(fi) > end_offset) break; // If the end of the file or the end line in the file for this thread is reached, break the while loop
      for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
      wrdi = ReadWordIndex(wv, fi);
      ctxi = ReadWordIndex(cv, fi);
      word_count++;
      if (wrdi < 0 || ctxi < 0) continue;

      if (sample > 0) {
        real ran = (sqrt(wv->vocab[wrdi].cn / (sample * wv->word_count)) + 1) * (sample * wv->word_count) / wv->vocab[wrdi].cn;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        ran = (sqrt(cv->vocab[ctxi].cn / (sample * cv->word_count)) + 1) * (sample * cv->word_count) / cv->vocab[ctxi].cn;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
      }

      // NEGATIVE SAMPLING
      l1 = wrdi * layer1_size;
      for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = ctxi;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = unitable[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (cv->vocab_size - 1) + 1;
          if (target == ctxi) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        grab_relation(relation, cv->vocab[target].word);
        rlti = SearchVocab(rv, relation);
        if (rlti < 0) continue;
        weight = relation_weight[rlti];

        f = 0;
        for (c = 0; c < layer1_size; c++) f += word_vecs[c + l1] * context_vecs[c + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha * weight;
        else if (f < -MAX_EXP) g = (label - 0) * alpha * weight;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha * weight;
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * context_vecs[c + l2];
        for (c = 0; c < layer1_size; c++) context_vecs[c + l2] += g * word_vecs[c + l1];
      }
      // Learn weights input -> hidden
      for (c = 0; c < layer1_size; c++) word_vecs[c + l1] += neu1e[c];
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}


void TrainModel() {
  int idx;
  long a;
  long long words, size;
  real *relation_weight_temp;
  char *relation_vocab_temp;
  file_size = GetFileSize(train_file); // Get the file size, which will be later used for dividing the task into threads
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  wv = ReadVocab(wvocab_file); // Read the words into the word vocabulary
  cv = ReadVocab(cvocab_file); // Read the contexts into the context vocabulary
  rv = ReadVocab(rvocab_file); // Read the relations into the relation vocabulary

  relation_weight = (real *)calloc(rv->vocab_size, sizeof(real));

  InitNet(wv, cv, &word_vecs, &context_vecs, layer1_size); // Initialize the word vectors and the context vectors to random vectors
  
  // Initialize the relation weight table
  if (rweight_file[0] != 0){
    // If the user provide the relation weight file, load the weights
    loadVec(rweight_file, &words, &size, &relation_vocab_temp, &relation_weight_temp, 0);
    for (a = 0; a < words; a++){
      idx = SearchVocab(rv, &relation_vocab_temp[a * MAX_STRING]);
      if (idx != -1){
        relation_weight[idx] = relation_weight_temp[a];
      }
    }
  }else{
    // If the user doesn't provide the relation weight file, simply assign the weight of each relation to 1
    for (a = 0; a < rv->vocab_size; a++){
      relation_weight[a] = 1;
    }
  }
  
  InitUnigramTable(cv, &unitable, table_size); // Initialize the unigram table, which will be later used in negative sampling
  start = clock(); // Record the starting time
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a); // Create the threads and start training
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  if (classes == 0) {
    // Save the word vectors
    if (dumpcv_file[0] != 0) {
      dumpVec(dumpcv_file, cv, context_vecs, layer1_size, binary);
    }
    dumpVec(output_file, wv, word_vecs, layer1_size, binary);
  } else {
    dumpKMean(classes, output_file, wv, word_vecs, layer1_size);
  }
}


int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 15, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    //printf("\t-min-count <int>\n");
    //printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words and contexts. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value in the original word2vec was 1e-5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-iters <int>\n");
    printf("\t\tPerform i iterations over the data; default is 1\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-dumpcv filename\n");
    printf("\t\tDump the context vectors in file <filename>\n");
    printf("\t-wvocab filename\n");
    printf("\t\twords vocabulary file\n");
    printf("\t-cvocab filename\n");
    printf("\t\tcontexts vocabulary file\n");
    printf("\t-rvocab filename\n");
    printf("\t\trelations vocabulary file\n");
    printf("\nExamples:\n");
    printf("./word2vecf -train data.txt -wvocab wv -cvocab cv -output vec.txt -size 200 -negative 5 -threads 10 \n\n");
    return 0;
  }
  output_file[0] = 0;
  wvocab_file[0] = 0;
  cvocab_file[0] = 0;
  rvocab_file[0] = 0;
  dumpcv_file[0] = 0;
  rweight_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-wvocab", argc, argv)) > 0) strcpy(wvocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-cvocab", argc, argv)) > 0) strcpy(cvocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-rvocab", argc, argv)) > 0) strcpy(rvocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-rweight", argc, argv)) > 0) strcpy(rweight_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-dumpcv", argc, argv)) > 0) strcpy(dumpcv_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-iters", argc, argv)) > 0) numiters = atoi(argv[i+1]);

  if (output_file[0] == 0) { printf("must supply -output.\n\n"); return 0; }
  if (wvocab_file[0] == 0) { printf("must supply -wvocab.\n\n"); return 0; }
  if (cvocab_file[0] == 0) { printf("must supply -cvocab.\n\n"); return 0; }
  if (rvocab_file[0] == 0) { printf("must supply -rvocab.\n\n"); return 0; }
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
