import networkx as nx

g = nx.Graph()
g.add_nodes_from(range(5), c=0)
g.add_edge(1,2)
for item in g.nodes.items():
    print(item)
# for item in g.edges.items():
#     item[1]['cnt'] = 1
# for item in g.edges.items():
#     print(item)
g.edge