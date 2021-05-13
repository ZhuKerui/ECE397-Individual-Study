from tools.BasicUtils import my_read, my_write
from typing import Dict, List

# Co-occurrence list related functions
def gen_co_occur(occur_dict:Dict[str, set], sent_len:int, word2idx_dict:Dict[str, int]):
    co_occur_list = [set() for i in range(sent_len)]
    for key, set_ in occur_dict.items():
        idx = word2idx_dict[key]
        for line in set_:
            co_occur_list[line].add(idx)
    return co_occur_list

def co_occur_dump(co_occur_file:str, co_occur_list:List[set]):
    my_write(co_occur_file, [' '.join(map(str, set_)) for set_ in co_occur_list])

def co_occur_load(co_occur_file:str):
    return [list(map(int, line.split())) for line in my_read(co_occur_file)]