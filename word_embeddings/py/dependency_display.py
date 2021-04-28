import sys
from spacy import displacy, load
nlp = load('en_core_web_sm')

doc = nlp(sys.argv[1])
# entity_types = ((ent.text, ent.label_) for ent in doc.ents)
# print(tabulate(entity_types, headers=['Entity', 'Entity Type']))
displacy.serve(doc, style='dep')