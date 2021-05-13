# Import libraries and define global variables
import torch
from torch.nn import Module, Dropout, ReLU, Embedding, Sequential, Linear
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from typing import Tuple, List

from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm

import pandas as pd

def create_fields():
    path_field = data.Field(sequential=True, tokenize=lambda x: x.split(), lower=True, fix_length=10)
    entity_field = data.Field(sequential=False)
    return path_field, entity_field

class MyDataset(data.Dataset):
    def __init__(self, corpus_path:str, path_field:data.Field, entity_field:data.Field, test:bool=False, **kwargs):
        
        fields = [('id', None), ('path', path_field), ('subjs', entity_field), ('objs', entity_field)]
        corpus_data = pd.read_csv(corpus_path)

        if test:
            examples = [data.Example.fromlist([None, text, None, None], fields=fields) for text in tqdm(corpus_data['path'])]
        else:
            examples = [data.Example.fromlist([None, path, subj, obj], fields=fields) for path, subj, obj in tqdm(zip(corpus_data['path'], corpus_data('subj'), corpus_data['obj']))]
        super(MyDataset, self).__init__(examples=examples, fields=fields, **kwargs)

path_field, entity_field = create_fields()
train_data = MyDataset('train.tsv', path_field=path_field, entity_field=entity_field, test=False)
valid_data = MyDataset('valid.tsv', path_field=path_field, entity_field=entity_field, test=False)

path_field.build_vocab(train_data)
entity_field.build_vocab(train_data)
train_iter, val_iter = data.BucketIterator.splits((train_data, valid_data), batch_sizes=(8, 8), device=-1, sort_key=lambda x: len(x.path_field), sort_within_batch=True, repeat=False)








# model_name = 'bert-base-uncased'
# raw_data = pd.read_csv('drive/MyDrive/Colab Notebooks/CS412/data/train.csv')

# # Define classes
# class DataToDataset(Dataset):
#     def __init__(self, depPath:List[str], pair:List[Tuple[str, str]]):
#         self.depPath = depPath
#         self.pair = pair

#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, index):
#         return self.depPath[index], self.pair[index]

# class PairEncoder(Module):
#     def __init__(self, keyword_vocab:List[str], dropout:float=0.5, normalize_args:bool=False, d_args:int=300, d_rels:int=300):
#         super(PairEncoder, self).__init__()
#         self.dropout = Dropout(p=dropout)
#         self.nonlinearity  = ReLU()
#         self.normalize = normalize if normalize_args else (lambda x : x)
#         self.arg_emb = Embedding(len(keyword_vocab), d_args)
        
#         self.pairEncoder = Sequential(
#             self.dropout, 
#             Linear(3 * d_args, d_args), 
#             self.nonlinearity, 
#             self.dropout, 
#             Linear(d_args, d_args), 
#             self.nonlinearity, 
#             self.dropout, 
#             Linear(d_args, d_args), 
#             self.nonlinearity, 
#             self.dropout, 
#             Linear(d_args, d_rels)
#         )

#     def forward(self, subjects:torch.Tensor, objects:torch.Tensor):
#         subjects = self.normalize(subjects)
#         objects = self.normalize(objects)
#         return self.pairEncoder(torch.cat([subjects, objects, subjects * objects], dim=-1))

# class PathEncoder(Module):
#     def __init__(self, config):
#         super(PathEncoder, self).__init__()

