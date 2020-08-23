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

class MyEngine:
    def __init__(self, index_path, dataset_path):
        logging.debug('Start building index')
        # Initialize index storage path
        self.index_path = index_path or 'default_index'
        # Initialize schema
        self.schema = Schema(id=ID(stored=True), submitter=TEXT(stored=True), 
                             authors=TEXT(stored=True), title=TEXT(stored=True), 
                             comments=TEXT(stored=True), journal_ref=ID(stored=True),
                             doi=ID(stored=True), report_no=ID(stored=True),
                             categories=KEYWORD(stored=True,scorable=True), 
                             license=STORED,abstract=TEXT(stored=True), 
                             versions=STORED, update_date=ID(stored=True), authors_parsed=STORED)
        self.fields = self.schema.names()
        self.positioned_fields = ['submitter', 'authors', 'title', 'comments', 'abstract']
        self.n_positioned_fields = ['id', 'journal_ref', 'doi', 'report_no', 'categories', 'update_date']
        self.searchable_fields = self.n_positioned_fields + self.positioned_fields
        # Create the index storage if not exist
        if not os.path.exists(self.index_path):
            # Create the index storage path
            os.mkdir(self.index_path)
            # Create the index object
            self.ix = create_in(self.index_path, self.schema)
            # Load the dataset if it exists
            if os.path.exists(dataset_path):
                with io.open(dataset_path, 'r', encoding='utf-8') as load_file:
                    writer = self.ix.writer()
                    for line in load_file:
                        jsonObj = json.loads(line)
                        writer.add_document(id=str(jsonObj['id']), 
                                            submitter=str(jsonObj['submitter']), 
                                            authors=str(jsonObj['authors']), 
                                            title=str(jsonObj['title']), 
                                            comments=str(jsonObj['comments']), 
                                            journal_ref=str(jsonObj['journal-ref']),
                                            doi=str(jsonObj['doi']), 
                                            report_no=str(jsonObj['report-no']),
                                            categories=str(jsonObj['categories']), 
                                            license=str(jsonObj['license']),
                                            abstract=str(jsonObj['abstract']), 
                                            versions=json.dumps(jsonObj['versions']), 
                                            update_date=str(jsonObj['update_date']), 
                                            authors_parsed=json.dumps(jsonObj['authors_parsed']))
                    writer.commit()
                logging.debug('Dataset loaded to index.')
            else:
                logging.debug('Dataset file not found')

            logging.debug('New index storage created.')
        # Load index from storage if exist
        else:
            self.ix = open_dir(self.index_path)
            logging.debug('Index loaded from existing storage.')
        
    @classmethod
    def load_from_index(cls, index_path):
        return cls(index_path, '')

    def search(self, keywords:str, fields:str, data_fields:list, is_strict:bool, limit:int=5)->list:
        logging.debug('Enter search')
        # Pre-process the keywords
        raw_keywords = keywords.split(';')
        keywords_list = []
        for keyword in raw_keywords:
            # Strip the keyword
            keyword = keyword.strip()
            # Avoid processing the empty string
            if not keyword == '':
                tokened_keyword = word_tokenize(keyword)
                sw_removed = ' '.join([word for word in tokened_keyword if not word in stopwords.words()])
                keywords_list.append(sw_removed)
        
        # Pre-process the fields
        fields_list = fields.split(',')
        if '' in fields_list:
            search_fields = self.searchable_fields
        else:
            search_fields = fields_list
        
        # Create the parse according to the is_strict
        if is_strict:
            phrase_fields = [field for field in search_fields if field in self.positioned_fields]
            n_phrase_fields = search_fields
            phrase_kw = [word for word in keywords_list if ' ' in word]
            n_phrase_kw = [word for word in keywords_list if ' ' not in word]

            # If there are phrases in keywords but no positioned field for search, directly return empty list since these keywords can't be found
            if (phrase_kw and not phrase_fields) or not (phrase_kw or n_phrase_kw):
                return []

            if phrase_kw:
                phrase_kw_str = '"'+'" "'.join(phrase_kw)+'"'
                phrase_parse = MultifieldParser(phrase_fields, schema=self.schema).parse(phrase_kw_str)
            
            if n_phrase_kw:
                n_phrase_kw_str = ' '.join(n_phrase_kw)
                n_phrase_parse = MultifieldParser(n_phrase_fields, schema=self.schema).parse(n_phrase_kw_str)

            if phrase_kw and n_phrase_kw:
                search_parse = And([phrase_parse, n_phrase_parse])
            elif phrase_kw:
                search_parse = phrase_parse
            else:
                search_parse = n_phrase_parse

        else:
            search_parse = MultifieldParser(search_fields, schema=self.schema, group=OrGroup).parse(' '.join(keywords_list))

        # Filter the data fields
        ret_fields = [field for field in data_fields if field in self.fields]
        if not ret_fields:
            return []

        # Search for results and prepare the return value
        ret = []
        logging.debug('Start searching')
        with self.ix.searcher() as searcher:
            results = searcher.search(search_parse, limit = limit)
            for data in results:
                temp_dict = {}
                for field in ret_fields:
                    temp_dict[field] = data[field]
                ret.append(temp_dict)
                
        return ret