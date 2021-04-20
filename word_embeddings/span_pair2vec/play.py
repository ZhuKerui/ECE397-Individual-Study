from span_pair2vec.model import Pair2Vec
from span_pair2vec.util import Config, load_model
from span_pair2vec.indexed_field import Field
from span_pair2vec.matrix_data import create_vocab
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append('..')
from my_util import ntopidx
import numpy as np
import pyhocon
import torch

class Play:
    def __init__(self, model_file, model_config_file, relation_file=''):
        pair2vec_config = Config(**pyhocon.ConfigFactory.parse_file(model_config_file))
        field = Field(batch_first=True)
        create_vocab(pair2vec_config, field)
        self.pair2vec = Pair2Vec(pair2vec_config, field.vocab, field.vocab)
        load_model(model_file, self.pair2vec)
        # freeze pair2vec
        for param in self.pair2vec.parameters():
            param.requires_grad = False
        
        self.rel_str = None
        self.relation_representation = None
        if relation_file:
            self.rel_str = open(relation_file).read().splitlines()
            self.relation_representation = self.get_relation([line.split() for line in self.rel_str])
    
    def stoi(self, s):
        if isinstance(s, str):
            return self.pair2vec.arg_vocab.stoi[s]
        elif isinstance(s, list):
            return [self.pair2vec.arg_vocab.stoi[s_] for s_ in s]
        else:
            return None

    def token_align(self, phrases:list, length:int) -> tuple:
        """Align the input phrases.

        Input the list of words and output the aligned phrase list and the words with the length of "length".

        Args:
            pharses: A list of strings , each string is a phrase seperated by spaces.
            length: The length of the aligned phrase.

        Returns:
            phrases list: A list of phrases with each phrase has the length of 6.
            words: A list of strings with each string is a phrase seperated by space.
        """
        tokens = [phrase.split() for phrase in phrases if len(phrase.split()) <= length]
        return [token + ['<pad>'] * (length - len(token)) for token in tokens], [' '.join(token) for token in tokens]

    def encode_keyword_phrase(self, keyword_phrases:list) -> np.ndarray:
        """Calculate the representation of keyword phrases.

        Input the list of keyword phrases and output the representation.

        Args:
            keyword_phrases: The list of keyword phrases, each item should be a list of the tokens in the keyword with the length aligned.

        Returns:
            numpy.ndarray with each row as the representation of the keyword phrase at the same row.
        """
        keyword_idx = torch.LongTensor([self.stoi(span) for span in keyword_phrases])
        keyword_idx = (keyword_idx, 1.0 - torch.eq(keyword_idx, self.pair2vec.pad).float())
        keyword_vector = self.pair2vec.represent_argument(keyword_idx)
        return keyword_vector

    def get_prediction(self, subjects:list, objects:list) -> np.ndarray:
        """Calculate the prediction of the relation of two phrases.

        Input the list of key phrases as subjects and objects and output the predicted relation representation.

        Args:
            subjects: The list of subject phrases, each item should be a list of the tokens in the subject with the length aligned.
            objects: The list of object phrases, just like the subjects. The length of the objects should be the same as the length of the subjects.

        Returns:
            numpy.ndarray with each row as the predicted relation representation of the pair of subject and object at the same row.
        """
        embedded_subjects = self.encode_keyword_phrase(subjects)
        embedded_objects = self.encode_keyword_phrase(objects)
        predict_relation = self.pair2vec.predict_relations(embedded_subjects, embedded_objects).numpy()
        return predict_relation

    def get_relation(self, relations:list) -> np.ndarray:
        """Calculate the relation representation of relation phrases.

        Input the list of relation phrases and output the relation representation.

        Args:
            relations: The list of relation phrases, each item should be a list of the tokens in the relation with the length aligned.

        Returns:
            numpy.ndarray with each row as the relation representation of the relation phrase at the same row.
        """
        relations_idx = torch.LongTensor([self.stoi(span) for span in relations])
        relations_idx = (relations_idx, 1.0 - torch.eq(relations_idx, self.pair2vec.pad).float())
        relation_vector = self.pair2vec.represent_relations(relations_idx).numpy()
        return relation_vector

    def itos(self, arr):
        return [self.pair2vec.arg_vocab.itos[i] for i in arr]

    def find_closest_relation(self, subjects:list, objects:list, num:int, return_score:bool=False):
        predicted_representation = self.get_prediction(subjects, objects)
        score = cosine_similarity(predicted_representation, self.relation_representation)[0]
        result = []
        idx = np.arange(len(score))
        for s in score:
            filtered_idx = ntopidx(s)
            if not return_score:
                result.append([self.rel_str[i] for i in filtered_idx])
            else:
                result.append(([self.rel_str[i] for i in filtered_idx], [s[i] for i in filtered_idx]))
        return result

    def cal_score(self, subjects:list, objects:list, relations:list) -> np.ndarray:
        """Calculate the score given the triples.

        Input the list of phrases as subjects, objects and relations and output the score.

        Args:
            subjects: The list of subject phrases, each item should be a list of the tokens in the subject with the length aligned.
            objects: The list of object phrases, each item should be a list of the tokens in the object with the length aligned.
            relations: The list of relation phrases, each item should be a list of the tokens in the relation with the length aligned.

        Returns:
            numpy.ndarray where the (i, j) element is the score between the ith subject-object pair and the jth relation.
        """
        relation_representation = self.get_relation(relations)
        predicted_representation = self.get_prediction(subjects, objects)
        score = cosine_similarity(predicted_representation, relation_representation)
        return score