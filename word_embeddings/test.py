import json
import io
import spacy
import re

bad_deps = ['aux', 'auxpass', 'cc', 'neg', 'num', 'ROOT', 'pobj', 'punct', 'det', 'dep']
examples = []

nlp = spacy.load('en_core_web_sm')

t = 'Neural word embeddings are often considered opaque and uninterpretable, unlike sparse vector space representations in which each dimension corresponds to a particular known context, or LDA models where dimensions correspond to latent topics. While this is true to a large extent, we observe that S KIP G RAM does allow a non-trivial amount of introspection. Although we cannot assign a meaning to any particular dimension, we can indeed get a glimpse at the kind of information being captured by the model, by examining which contexts are “activated” by a target word.'

doc = nlp(t)

for sentence in doc.sents:
    for word in sentence:
        #casing should probably be a paramater
        source =  word.head.text.lower()
        target= word.text.lower()
        dep = word.dep_
        for child in word.children:
            source =  child.head.text.lower()
            target= child.text.lower()
            dep = child.dep_
            #If we see a prepositional dependency, we want to merge it
            #so ('scientist', prep, 'with') and ('with', pobj,'telescope) 
            #becomes ('scientist, 'prep_with', 'telescope)
            if dep == 'prep':
                for c2 in child.children:
                    if (c2.dep_ == 'pobj'):
                        examples.append((source,"prep_" + child.text.lower(),  c2.text))
            else:
                if not dep in bad_deps:
                    examples.append((source, dep,target))

for item in examples:
    print(item)