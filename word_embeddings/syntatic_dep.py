import json
import io
import spacy
import re
from spacy_conll import init_parser

nlp = spacy.load('en_core_web_sm')
last_end_line = 1076000

def json_generator(json_file):
    with io.open(json_file, 'r', encoding='utf-8') as load_file:
        for line in load_file:
            yield json.loads(line)


def extract_sent(json_file, store_file):
    with io.open(store_file, 'a', encoding='utf-8') as output:
        cnt = 0
        for jsonObj in json_generator(json_file):
            cnt += 1
            if cnt <= last_end_line:
                continue
            para = jsonObj['abstract'].strip().replace('\n', ' ')
            latex_str = re.search(r'\$.*?\$', para)
            while latex_str:
                para = para.replace(latex_str.group(), '')
                latex_str = re.search(r'\$.*?\$', para)
            doc = nlp(para)
            for sentence in doc.sents:
                output.write(str(sentence) + '\n')
            if cnt % 1000 == 0:
                print(cnt)
        print(cnt)


def conll_gen(text, store_file):
    nlp = init_parser("spacy", "en")
    # Parse a given string
    doc = nlp(text)
    with io.open(store_file, 'w', encoding='utf-8') as output:
        output.write(doc._.conll_str)
        

def analysis_data(input_txt, output_txt):
    with io.open(input_txt, 'r', encoding='utf-8') as input_file:
        context_dict = {}
        for line in input_file:
            context = line.split(' ')
            if context[1] in context_dict.keys():
                context_dict[context[1]].append(context[0])
            else:
                context_dict[context[1]] = [context[0]]
    with io.open(output_txt, 'w', encoding='utf-8') as output_file:
        for key, item in context_dict.items():
            output_file.write(str(key)+'\n')
            for word in item:
                output_file.write('    ' + str(word) + '\n')
            output_file.write('\n')
