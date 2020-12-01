from relation_similar_web.dep_generator import *

class Pair_Embed(Dep_Based_Embed_Generator):
    def extract_pair_context(self, id_:int, corpus:str, context_file:str, reformed_file:str=None, sent_split:bool=False, start_line:int=0, lines:int=0):
        try:
            self.keywords
        except NameError:
            print('Keywords are not loaded')
            return
            
        if lines <= 0:
            return
        global is_exit
        reformed_output_file = None
        if reformed_file is not None:
            reformed_output_file = io.open(reformed_file, 'w', encoding='utf-8')
        with io.open(context_file, 'w', encoding='utf-8') as context_output_file:
            with io.open(corpus, 'r', encoding='utf-8') as load_file:
                idx = -1
                str_buffer = []
                for idx, line in enumerate(load_file):
                    if is_exit:
                        break
                    if idx < start_line:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    if reformed_output_file is not None:
                        line = self._process_sent(line)
                        if not line:
                            continue
                        if not sent_split:
                            reformed_output_file.write(line + '\n')
                    doc = nlp(line)
                    for sentence in doc.sents:
                        if reformed_output_file is not None and sent_split:
                            reformed_output_file.write(sentence.text + '\n')
                        kw2ctx = {}
                        for word in sentence:
                            if word.text not in self.keywords:
                                continue
                            word_txt = word.text
                            if word_txt not in kw2ctx:
                                kw2ctx[word_txt] = set()
                            for child in word.children:
                                if child.dep_ == 'prep':
                                    relation = ''
                                    child_txt = ''
                                    for grand_child in child.children:
                                        if grand_child.dep_ == 'pobj':
                                            relation = 'prep_' + child.text.lower()
                                            child_txt = grand_child.text.lower()
                                    if not relation:
                                        continue
                                else:
                                    relation = child.dep_
                                    child_txt = child.text.lower()
                                kw2ctx[word_txt].add(relation + '_' + child_txt)

                            kw2ctx[word_txt].add(word.dep_ + 'I_' + word.head.text)
                        if len(kw2ctx) <= 1:
                            continue
                        kws = list(kw2ctx.keys())
                        for i in range(len(kws)-1):
                            for j in range(i+1, len(kws)):
                                pair_1 = kws[i] + '__' + kws[j]
                                pair_2 = kws[j] + '__' + kws[i]
                                for ctx in kw2ctx[kws[i]]:
                                    str_buffer.append(pair_1 + ' h_' + ctx + '\n')
                                    str_buffer.append(pair_2 + ' t_' + ctx + '\n')
                                for ctx in kw2ctx[kws[j]]:
                                    str_buffer.append(pair_1 + ' t_' + ctx + '\n')
                                    str_buffer.append(pair_2 + ' h_' + ctx + '\n')
                    cnt = idx - start_line
                    if cnt >= lines - 1:
                        break
                    if cnt % 100 == 0:
                        context_output_file.write(''.join(str_buffer))
                        str_buffer = []
                        print('Thread %d has processed %.2f' %(id_, float(cnt) * 100 / lines))

                context_output_file.write(''.join(str_buffer))
                if reformed_output_file is not None:
                    reformed_output_file.close()
                if is_exit:
                    print('Thread %d is terminated' % (id_))
                else:
                    print('Extract context accomplished with %d lines processed' % (1 + idx - start_line))
