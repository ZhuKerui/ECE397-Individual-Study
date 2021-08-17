from typing import List

def processed_file_reader(file_in:str) -> List[List[str]]:
    sof = True
    ret = []
    for line in open(file_in, 'r'):
        line = line.strip()
        if sof:
            ret.append([line])
            sof = False
        elif line:
            ret[-1].append(line)
        else:
            sof = True
    return ret

def openie_my_map(read_list:list, func):
    return [func(item) for item in read_list]

def get_triple_with_kw(item:List[str], kw1:str, kw2:str=None):
    new_list = [item[0]]
    for triple in item[1:]:
        if kw1 not in triple:
            continue
        if kw2 is not None:
            if kw2 not in triple:
                continue
        new_list.append(triple)
    return new_list

def write_file(write_file:str, write_list:List[List[str]], discard_empty:bool=False):
    if discard_empty:
        write_items = ['\n'.join(item) for item in write_list if len(item) > 1]
    else:
        write_items = ['\n'.join(item) for item in write_list]
    with open(write_file, 'w') as f_out:
        f_out.write('\n\n'.join(write_items))