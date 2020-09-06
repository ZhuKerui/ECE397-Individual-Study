from whoosh.index import create_in, open_dir
from whoosh.fields import TEXT, ID, KEYWORD, STORED, Schema
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup
from whoosh.query import And, Or
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import io
import json
import logging
import requests

class MyEngine:
    def __init__(self, index_path:str='./', json_data_path:str=''):
        logging.debug('Start building index')
        # Initialize index storage path
        self.index_path = index_path
        # Create the index storage if not exist
        if not os.path.exists(self.index_path):
            # Create the index storage path
            os.mkdir(self.index_path)
            # Initialize schema
            self.schema = Schema(keyword=ID(stored=True), related_info=KEYWORD(stored=True,scorable=True))
            # Create the index object
            self.ix = create_in(self.index_path, self.schema)
            logging.debug('New index storage created.')
            if json_data_path:
                self.expand_vocab(json_data_path)
        # Load index from storage if exist
        else:
            self.ix = open_dir(self.index_path)
            self.schema = self.ix.schema
            logging.debug('Index loaded from existing storage.')


    def expand_vocab(self, json_data_path:str):
        # Load the dataset if it exists
        logging.debug('Start loading json data from ' + json_data_path)
        if os.path.exists(json_data_path):
            with io.open(json_data_path, 'r', encoding='utf-8') as load_file:
                # Load the json data
                dataset = json.load(load_file)
                # Create a writer
                writer = self.ix.writer()
                for keyword, suggested_words in dataset.items():
                    n_sw_suggested = [word for word in suggested_words if word not in stopwords.words()]
                    writer.add_document(keyword=str(keyword), related_info=str(' '.join(n_sw_suggested)))
                writer.commit()
            logging.debug('Index expanded with new value.')
        else:
            logging.debug('Dataset file not found')


    def get_suggested(self, keyword):
        results = {}
        params = {"client":"firefox", "hl":"en"}
        url = 'http://suggestqueries.google.com/complete/search'
        q_word = keyword.strip().lower()
        if q_word:
            params["q"] = q_word
            temp_r = requests.get(url, params=params)
            logging.debug('Feedback received')
            result = temp_r.json()
            # Make sure that the keyword itself is in the list
            keyword_list = q_word.split(' ')
            for suggest in result[1]:
                # Skip the results that change the query word
                if q_word not in suggest:
                    continue
                suggested_words = suggest.split(' ')
                for word in suggested_words:
                    if word not in keyword_list and word not in stopwords.words():
                        keyword_list.append(word)
            return keyword_list


    def search(self, keywords:str, local_search:bool=False, limit:int=5)->dict:
        logging.debug('Enter search')
        # Pre-process the keywords
        raw_keywords = keywords.split(';')
        keywords_list = []
        for keyword in raw_keywords:
            # Strip the keyword
            keyword = keyword.strip()
            # Avoid processing the empty string
            if keyword:
                keywords_list.append(keyword.lower())

        # Prepare the weights dict
        weights = {}
        # Prepare the scores and overlaps dict
        scores = {}
        overlaps = {}
        # Prepare the rank list
        rank = []
        # Search the queried keywords in the index storage
        with self.ix.searcher() as searcher:
            # logging.debug('local search:' + str(local_search))
            if not local_search:
                # Search from internet
                for keyword in keywords_list:
                    suggested_words = self.get_suggested(keyword)
                    # Fill the weights
                    for word in suggested_words:
                        if word in weights.keys():
                            weights[word] += 1
                        else:
                            weights[word] = 1
            else:
                # Search from local index storage
                # Generate the search parse
                first_parse_str = '"'+'" "'.join(keywords_list)+'"'
                first_query = QueryParser("keyword", schema=self.schema, group=OrGroup).parse(first_parse_str)
                first_results = searcher.search(first_query)

                for data in first_results:
                    suggested_words = data['related_info'].split(' ')
                    # Fill the weights
                    for word in suggested_words:
                        if word in weights.keys():
                            weights[word] += 1
                        else:
                            weights[word] = 1

            second_parse_str = ' '.join(weights.keys())
            second_query = QueryParser("related_info", schema=self.schema, group=OrGroup).parse(second_parse_str)
            second_results = searcher.search(second_query)
            
            for data in second_results:
                sub_keyword = data['keyword']
                scores[sub_keyword] = 0
                sub_suggested_words = data['related_info'].split(' ')
                suggested_overlap = [word for word in sub_suggested_words if word in weights.keys()]
                overlaps[sub_keyword] = suggested_overlap
                for word in suggested_overlap:
                    scores[sub_keyword] += weights[word]
                if not rank:
                    rank.append(sub_keyword)
                else:
                    i = 0
                    while i < len(rank):
                        if scores[sub_keyword] > scores[rank[i]]:
                            break
                        i += 1
                    rank.insert(i, sub_keyword)
                    
        ret = {}

        if len(rank) > limit:
            ret_len = limit
        else:
            ret_len = len(rank)
        
        for i in range(ret_len):
            ret[rank[i]] = overlaps[rank[i]]

        return ret

def json_gen(word_list_path, json_store_path):
    # if os.path.exists(word_list_path) and not os.path.exists(json_store_path):
    if os.path.exists(word_list_path):
        results = {}
        params = {"client":"firefox", "hl":"en"}
        url = 'http://suggestqueries.google.com/complete/search'
        with io.open(word_list_path, 'r', encoding='utf-8') as load_file:
            for line in load_file:
                q_word = line.strip().lower()
                if q_word:
                    params["q"] = q_word
                    temp_r = requests.get(url, params=params)
                    result = temp_r.json()
                    # Make sure that the keyword itself is in the list
                    keyword_list = q_word.split(' ')
                    for suggest in result[1]:
                        # Skip the results that change the query word
                        if q_word not in suggest:
                            continue
                        suggested_words = suggest.split(' ')
                        for word in suggested_words:
                            if word not in keyword_list:
                                keyword_list.append(word)
                    results[q_word] = keyword_list
        with io.open(json_store_path, 'w', encoding='utf-8') as dump_file:
            json.dump(results, dump_file)
