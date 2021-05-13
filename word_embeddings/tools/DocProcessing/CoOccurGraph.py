import networkx as nx
from typing import List
import math

# Graph related functions
def build_graph(co_occur_list:List[List[int]], keyword_list:List[str]):
    g = nx.Graph(c=0)
    g.add_nodes_from(range(len(keyword_list)), c=0)
    print('Reading Co-occurrence lines')
    for line_idx, line in enumerate(co_occur_list):
        kw_num = len(line)
        g.graph['c'] += kw_num * (kw_num - 1)
        for i in range(kw_num):
            u = line[i]
            g.nodes[u]['c'] += (kw_num - 1)
            for j in range(i+1, kw_num):
                v = line[j]
                if not g.has_edge(u, v):
                    g.add_edge(u, v, c=0)
                g.edges[u, v]['c'] += 1
        if line_idx % 5000 == 0:
            print('\r%d' % line_idx, end='')
    print('')
    print('Reading Done! NPMI analysis starts...')
    Z = float(g.graph['c'])
    for e, attr in g.edges.items():
        attr['npmi'] = -math.log((2 * Z * attr['c']) / (g.nodes[e[0]]['c'] * g.nodes[e[1]]['c'])) / math.log(2 * attr['c'] / Z)
    print('NPMI analysis Done')
    return g

def graph_dump(g:nx.Graph, gpickle_file:str):
    nx.write_gpickle(g, gpickle_file)

def graph_load(gpickle_file:str):
    return nx.read_gpickle(gpickle_file)

def get_subgraph(g:nx.Graph, npmi_threshold:float, min_count:int):
    return g.edge_subgraph([e[0] for e in g.edges.items() if e[1]['npmi'] > npmi_threshold and e[1]['c'] >= min_count])