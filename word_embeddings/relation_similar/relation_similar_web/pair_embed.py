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
                        for word in sentence:
                            if word.text not in self.keywords:
                                continue
                            word_txt = word.text.lower()
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
                                context_output_file.write(' '.join((word_txt, '_'.join((relation, child_txt)))) + '\n')

                            context_output_file.write(' '.join((word_txt, 'I_'.join((word.dep_, word.head.text.lower())))) + '\n')
                    cnt = idx - start_line
                    if cnt >= lines - 1:
                        break
                    if cnt % 100 == 0:
                        print('Thread %d has processed %.2f' %(id_, float(cnt) * 100 / lines))

                if reformed_output_file is not None:
                    reformed_output_file.close()
                if is_exit:
                    print('Thread %d is terminated' % (id_))
                else:
                    print('Extract context accomplished with %d lines processed' % (1 + idx - start_line))

    def extract_context_multithread(self, corpus, context_file:str, reformed_file:str=None, thread_num:int=1):
        global is_exit
        if self.keywords is None:
            print("You haven't load the keywords yet, please use build_word_tree(input_txt, dump_file) or load_word_tree(json_file) to load the keywords")
            return
        if thread_num <= 0:
            return
        line_count = -1
        with io.open(corpus, 'r', encoding='utf-8') as load_file:
            for line_count, line in enumerate(load_file):
                pass
            line_count += 1
        unit_lines = line_count / thread_num
        threads = []
        signal.signal(signal.SIGINT, multithread_kill)
        signal.signal(signal.SIGTERM, multithread_kill)
        is_exit = False
        for i in range(thread_num):
            # id:int, corpus:str, context_file:str, reformed_file:str=None, start_line:int=0, lines:int=0
            id_ = i
            temp_context_file = context_file + str(id_)
            temp_reformed_file = None
            if reformed_file is not None:
                temp_reformed_file = reformed_file + str(id_)
            start_line = unit_lines * id_
            if i < thread_num - 1:
                lines = unit_lines
            else:
                lines = line_count - unit_lines * i
            t = Process(target=self.extract_context, args=(id_, corpus, temp_context_file, temp_reformed_file, True, start_line, lines))
            t.setDaemon(True)
            threads.append(t)
        for i in range(thread_num):
            threads[i].start()
        while 1:
            alive = False
            for i in range(thread_num):
                alive = alive or threads[i].isAlive()
            if not alive:
                break
        tailor(context_file, context_file, '', thread_num, remove=True)
        if reformed_file is not None:
            tailor(reformed_file, reformed_file, '', thread_num, remove=True)