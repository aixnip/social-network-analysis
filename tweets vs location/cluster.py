"""
cluster.py
"""
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import k_means

def read_files():
    """
    this method assumes the files locates in the current folder...
    network.pkl

    returns
     data - a dict with node and it's following account ids
    """
    data = pickle.load(open('network.pkl', 'rb'))
    return data

def cal_edges(data):
    """
    this method computes pair-wise edge weight

    returns
     edges - a set containing tuples (node1, node2, weight)
    """
    user = list(data.keys())
    edges = set()
    for i in range(len(user)-1):
        for j in range(i, len(user)):
            w = len((set(data[user[i]]) & set(data[user[j]])))/len((set(data[user[i]]) | set(data[user[j]])))
            if w > 0.01: #tried 0.1, 0.05, 0.01, 0.005, 0.001
                edges.add((user[i], user[j], w))
    return edges

def create_graph(edges):
    """
    create graph out of the edges list

    Args:
     edges - list of edges tuples (node1, node2, weight)

    return
     graph - a networkx graph
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    return graph

def split_into_comm(graph, min_size=3, max_size=100):
    """
    this method recursively devides the input graph to communities with min and max sizes.

    Args:
     graph - networkx graph
     min_size - the min community size
     max_size - the max community size

    return:
     communities - a list of networkx graph
    
    """
    #detect existing communities
    communities = []
    large_comm = []
    for sg in nx.connected_component_subgraphs(graph, copy=True):
        if sg.order() > max_size:
            large_comm.append(sg.copy())
        elif sg.order() >= min_size:
            communities.append(sg)

    if len(large_comm) == 0:
        return communities
    
    #split into communities
    for sg in large_comm:
        i = 0
        sorted_betweenness = sorted(nx.edge_betweenness_centrality(sg).items(), key=lambda item: (-item[1], item[0]))
        while len(list(nx.connected_component_subgraphs(sg))) < 2 and i < len(sorted_betweenness):
            edge = sorted_betweenness[i][0]
            #remove only nodes that has more than 1 edge
            if len(list(nx.all_neighbors(sg, edge[0]))) > 2 and len(list(nx.all_neighbors(sg, edge[1]))) > 2: 
                sg.remove_edge(edge[0], edge[1])
            i += 1
        #recursively call this method to split into smaller communities.
        print('splitted into size %d , %d'%(tuple([ssg.order() for ssg in nx.connected_component_subgraphs(sg)])))
        for ssg in (nx.connected_component_subgraphs(sg)):
            communities.extend(split_into_comm(ssg, min_size, max_size))

    return communities

def main():
    data = read_files()
    edges = cal_edges(data)
    graph = create_graph(edges)
    print('graph has %d nodes and %d edges' % (graph.order(), graph.number_of_edges()))

    communities = split_into_comm(graph, max_size=min(100, int(len(data)/2)))
    print('\ndetected %d communities'%len(communities))
    community_sizes = [c.order() for c in communities]
    print(community_sizes)
    print('mean sizes %0.3f'%np.mean(community_sizes))
    nx.draw(graph,node_color=range(graph.order()),node_size=40,cmap=plt.cm.summer, linewidths=0.25, width=0.25, edge_color='#cccccc', font_size=10)
    plt.savefig('network.png')

    result = {'num_users':len(data),'community_sizes':community_sizes}
    pickle.dump(result, open('cluster_result.pkl', 'wb'))
    
if __name__ == '__main__':
    main()
