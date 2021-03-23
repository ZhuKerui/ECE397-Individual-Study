from embeddings.model import Pair2Vec
from embeddings.util import Config, load_model
from embeddings.indexed_field import Field
from embeddings.matrix_data import create_vocab
import pyhocon
import torch

class Play:
    def __init__(self, model_file, model_config_file):
        pair2vec_config = Config(**pyhocon.ConfigFactory.parse_file(model_config_file))
        field = Field(batch_first=True)
        create_vocab(pair2vec_config, field)
        self.pair2vec = Pair2Vec(pair2vec_config, field.vocab, field.vocab)
        load_model(model_file, self.pair2vec)
        # freeze pair2vec
        for param in self.pair2vec.parameters():
            param.requires_grad = False
    
    def stoi(self, s):
        if isinstance(s, str):
            return self.pair2vec.arg_vocab.stoi[s]
        elif isinstance(s, list):
            return [self.pair2vec.arg_vocab.stoi[s_] for s_ in s]
        else:
            return None

    def get_prediction(self, subjects:list, objects:list):
        subjects = [subjects] if isinstance(subjects[0], str) else subjects
        objects = [objects] if isinstance(objects[0], str) else objects
        subjects_idx, objects_idx = torch.LongTensor([self.stoi(span) for span in subjects]), torch.LongTensor([self.stoi(span) for span in objects])
        subjects_idx, objects_idx = self.pair2vec.to_tensors((subjects_idx, objects_idx))
        embedded_subjects = self.pair2vec.represent_argument(subjects_idx)
        embedded_objects = self.pair2vec.represent_argument(objects_idx)
        predict_relation = self.pair2vec.predict_relations(embedded_subjects, embedded_objects)
        return predict_relation

    def get_relation(self, relations:list):
        relations = [relations] if isinstance(relations[0], str) else relations
        relations_idx = torch.LongTensor([self.stoi(span) for span in relations])
        relations_idx = (relations_idx, 1.0 - torch.eq(relations_idx, self.pair2vec.pad).float())
        relation_vector = self.pair2vec.represent_relations(relations_idx)
        return relation_vector

    def itos(self, arr):
        return [self.pair2vec.arg_vocab.itos[i] for i in arr]