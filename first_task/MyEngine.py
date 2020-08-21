from whoosh.index import create_in, open_dir
from whoosh.fields import TEXT, Schema
import os
import io
import json

class MyEngine:
    def __init__(self, index_path, dataset_path):
        # Initialize index storage path
        self.index_path = index_path or 'default_index'
        # Initialize schema
        self.schema = Schema(id=TEXT(stored=True), submitter=TEXT(stored=True), 
                             authors=TEXT(stored=True), title=TEXT(stored=True), 
                             comments=TEXT(stored=True), journal_ref=TEXT(stored=True),
                             doi=TEXT(stored=True), report_no=TEXT(stored=True),
                             categories=TEXT(stored=True), license=TEXT(stored=True),
                             abstract=TEXT(stored=True))
        # Create the index storage if not exist
        if not os.path.exists(self.index_path):
            # Create the 
            os.mkdir(self.index_path)
            self.ix = create_in(self.index_path, self.schema)
            if os.path.exists(dataset_path):
                with io.open(dataset_path, 'r', encoding='utf-8') as load_file:
                    writer = self.ix.writer()
                    for line in load_file:
                        jsonObj = json.loads(line)
                        writer.add_document(title=str(jsonObj['title']), abstract=str(jsonObj['abstract']))
                    writer.commit()
                    print('Dataset loaded to index.')
            else:
                print('Dataset file not found')
            print('New index storage created.')
        # Load index from storage if exist
        else:
            self.ix = open_dir(self.index_path)
            print('Index loaded from existing storage.')
        
    @classmethod
    def load_from_index(cls, index_path):
        return cls(index_path, '')