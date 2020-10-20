#include "word2vecf_lib.h"

char train_file[MAX_STRING];
char wvocab_file[MAX_STRING], cvocab_file[MAX_STRING], rvocab_file[MAX_STRING];
char output_file[MAX_STRING], dumpcv_file[MAX_STRING];
char rweight_file[MAX_STRING];

int32_t binary = 0, debug_mode = 2, num_threads = 1;
int64_t layer1_size = 100;
int32_t batch_size = 100;
int64_t train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *word_vecs, *context_vecs, *expTable, *classification_vecs;
clock_t start;
int32_t numiters = 1;

struct vocabulary *wv;
struct vocabulary *cv;
struct vocabulary *rv;

real *relation_weight;

int32_t negative = 15;
const int32_t table_size = 1e8;
int32_t *unitable;

// Read word,context pairs from training file, where both word and context are integers.
// We are learning to predict context based on word.
//
// Word and context come from different vocabularies, but we do not really care about that
// at this point.
void *TrainModelThread(void *id) {
  int32_t ctxi = -1, wrdi = -1, rlti = -1; // ctxi: context index, wrdi: word index, rlti: relation index
  char class_str[2];
  int64_t word_count = 0, last_word_count = 0;
  int64_t l1, l2, l3, l4, a, b, c, d, target, label;
  int64_t block_size = rv->vocab_size * layer1_size;
  int64_t total_weight_num = rv->vocab_size * wv->vocab_size;
  uint64_t next_random = (uint64_t)id;
  char relation[MAX_STRING];
  real f, g;
  clock_t now;
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb"); // fi: The input training file. 
                                      // For word embedding, there are 2 elements each line: word_class, relation_context
  real *acc_weight_table = (real *)calloc(total_weight_num * layer1_size, sizeof(real));
  int32_t *weight_table_count = (int32_t *)calloc(total_weight_num, sizeof(int));
  int64_t start_offset = file_size / (int64_t)num_threads * (int64_t)id; // The starting point32_t in the file for this thread
  int64_t end_offset = file_size / (int64_t)num_threads * (int64_t)(id+1); // The ending point32_t in the file for this thread
  int32_t iter;
  //printf("thread %d %lld %lld \n",id, start_offset, end_offset);

  /* The training process begins here */
  for (iter=0; iter < numiters; ++iter) {
    fseek(fi, start_offset, SEEK_SET);
    
    while (fgetc(fi) != '\n') { }; //TODO make sure its ok
    // printf("thread %ld %ld\n", (int64_t)id, ftell(fi));

    int64_t train_words = wv->word_count;
    
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
      wrdi = ReadWordIndex(wv, fi);
      ctxi = ReadWordIndex(cv, fi);
      word_count++;
      if (wrdi < 0 || ctxi < 0) continue;

      if (sample > 0) {
        real ran = (sqrt(wv->vocab[wrdi].cn / (sample * wv->word_count)) + 1) * (sample * wv->word_count) / wv->vocab[wrdi].cn;
        next_random = next_random * (uint64_t)25214903917 + 11;
        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        ran = (sqrt(cv->vocab[ctxi].cn / (sample * cv->word_count)) + 1) * (sample * cv->word_count) / cv->vocab[ctxi].cn;
        next_random = next_random * (uint64_t)25214903917 + 11;
        if (ran < (next_random & 0xFFFF) / (real)65536) continue;
      }

      // NEGATIVE SAMPLING
      l1 = wrdi * layer1_size;
      for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = ctxi;
          label = 1;
        } else {
          next_random = next_random * (uint64_t)25214903917 + 11;
          target = unitable[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (cv->vocab_size - 1) + 1;
          if (target == ctxi) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        grab_relation(relation, cv->vocab[target].word);
        rlti = SearchVocab(rv, relation);
        if (rlti < 0) continue;

        l3 = wrdi * block_size + rlti * layer1_size;
        f = 0;
        for (c = 0; c < layer1_size; c++) f += word_vecs[c + l1] * context_vecs[c + l2];
        if (f > MAX_EXP) g = (label - 1);
        else if (f < -MAX_EXP) g = (label - 0);
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]);
        // Add the weight to the acc_weight_table according to the word and the relation
        for (c = 0; c < layer1_size; c++) acc_weight_table[c + l3] += g * context_vecs[c + l2] * alpha;
        for (c = 0; c < layer1_size; c++) context_vecs[c + l2] += g * word_vecs[c + l1] * alpha;
        weight_table_count[wrdi * rv->vocab_size + rlti] += 1;
      }


      // Learn weights input -> hidden
      if ((word_count - last_word_count) % batch_size == 0){
        // If we have processed a batch size of words, we update the word vector and the weights
        for (a = 0; a < wv->vocab_size; a++){
          // For each word in the word vocabulary
          l1 = a * layer1_size; // Locate the starting point32_t of the word vector
          for (b = 0; b < rv->vocab_size; b++){
            // For each relation contribution of the word
            l3 = a * block_size + b * layer1_size; // Locate the starting point32_t of the learned weights from the previous word embedding
            l4 = a * rv->vocab_size + b; // Locate the counter of the present relation in the present word
            if (weight_table_count[l4] == 0) continue; // If this relation didn't appear to this word, just skip it
            for (c = 0; c < layer1_size; c++) {
              // For each entry of the vector
              acc_weight_table[c + l3] /= weight_table_count[l4]; // Update the accumulated weight to the average value
              word_vecs[c + l1] += acc_weight_table[c + l3] * relation_weight[b]; // Add the product of the learned weight and 
                                                                                    // the weight of the relation to update the word vector
            }
          }
          // For now, all the contribution from every relation have been added to the present word vector
          grab_relation(class_str, wv->vocab[a].word); // Take the class of the present word
          label = atoi(class_str); // Transform the class to an integer
          f = 0; // Clear the loss
          for (c = 0; c < layer1_size; c++) f += word_vecs[c + l1] * classification_vecs[c]; // Calculate the loss
          // Calculate the back-propagation loss
          if (f > MAX_EXP) g = (label - 1);
          else if (f < -MAX_EXP) g = (label - 0);
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]);
          
          for (c = 0; c < layer1_size; c++){
            // Calculate the loss of each entry of the word vector
            neu1e[c] = g * classification_vecs[c];
            // Update the classification vector weight
            classification_vecs[c] += g * word_vecs[c + l1] * alpha;
            // Update the word vector according to the classification result
            word_vecs[c + l1] += neu1e[c] * alpha;
          }

          // Now, update the relation weights
          for (b = 0; b < rv->vocab_size; b++){
            // For each relation
            l3 = a * block_size + b * layer1_size; // Locate the starting point32_t of the learned weights from the previous word embedding
            for (c = 0; c < layer1_size; c++){
              relation_weight[b] += acc_weight_table[c + l3] * neu1e[c] * alpha;
            }
          }
        }

        for (a = 0; a < total_weight_num; a++){
          l3 = a * layer1_size;
          for (b = 0; b < layer1_size; b++){
            acc_weight_table[l3 + b] = 0;
          }
          weight_table_count[a] = 0;
        }
      }
    }
  }
  fclose(fi);
  free(neu1e);
  free(acc_weight_table);
  free(weight_table_count);
  pthread_exit(NULL);
}


void TrainModel() {
  int64_t a;
  file_size = GetFileSize(train_file); // Get the file size, which will be later used for dividing the task into threads
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  classification_vecs = (real *)malloc(layer1_size * sizeof(real));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  wv = ReadVocab(wvocab_file); // Read the words into the word vocabulary
  cv = ReadVocab(cvocab_file); // Read the contexts into the context vocabulary
  rv = ReadVocab(rvocab_file); // Read the relations into the relation vocabulary

  InitNet(wv, cv, &word_vecs, &context_vecs, layer1_size); // Initialize the word vectors and the context vectors to random vectors
  InitRelationWeight(rv, &relation_weight); // Initialize the relation weight table
  InitUnigramTable(cv, &unitable, table_size); // Initialize the unigram table, which will be later used in negative sampling
  // Initialize the classification weight vector
  for (a = 0; a < layer1_size; a++){
    classification_vecs[a] = rand() / (real)RAND_MAX;
  }
  
  start = clock(); // Record the starting time
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a); // Create the threads and start training
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  if (classes == 0) {
    // Save the word vectors
    if (dumpcv_file[0] != 0) {
      dumpVec(dumpcv_file, cv, context_vecs, layer1_size, binary);
    }
    dumpVec(output_file, wv, word_vecs, layer1_size, binary);
    dumpVec(rweight_file, rv, relation_weight, 1, binary);
  } else {
    dumpKMean(classes, output_file, wv, word_vecs, layer1_size);
  }
  free(classification_vecs);
}


int32_t main(int32_t argc, char **argv) {
  int32_t i;
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
    printf("\t-rweight filename\n");
    printf("\t\trelation weights output file\n");
    printf("\t-batch <int>\n");
    printf("\t\tbatch size during the training\n");
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
  if ((i = ArgPos((char *)"-batch", argc, argv)) > 0) batch_size = atoi(argv[i + 1]);
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
  if (rweight_file[0] == 0) { printf("must supply -rweight.\n\n"); return 0; }
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  free(expTable);
  return 0;
}
