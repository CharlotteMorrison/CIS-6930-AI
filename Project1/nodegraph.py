import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


class NodeGraph:
    def __init__(self, ig):
        self.graph = nx.Graph()
        self.ig = ig

    def new_node(self, node, ig, fv, sv, result):
        # if leaf
        if ig <= 0:
            label = "Leaf: " + fv + "\nResult: " + result
        # not a leaf
        else:
            feature = "Feature: " + str([f for f in sv]) + "\n"
            info_gain = "IG: " + str(round(ig, 4))
            att = "Att: " + fv + "\n"
            label = att + feature + info_gain

        self.graph.add_node(node, label=label)

    def new_node_list(self, nodes):
        self.graph.add_nodes_from(nodes)

    def new_edge(self, edge, label):
        self.graph.add_edge(*edge, edge_label=label)

    def new_edge_list(self, edges):
        self.graph.add_edges_from(edges)

    def print_nodes(self):
        print("Nodes of the graph: ")
        print(self.graph.nodes())

    def print_edges(self):
        print("Edges of the graph: ")
        print(self.graph.edges())

    def draw_graph(self, dataset_num):
        labels = nx.get_node_attributes(self.graph, 'label')
        edge_labels = nx.get_edge_attributes(self.graph, 'edge_label')
        # convert nx graph to dot for pygraphviz
        pos = graphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, pos, labels=labels, node_size=500, node_shape="s", font_size=8, node_color='skyblue',
                linewidths=4, font_color="gray", width=2, edge_color="gray")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        # nx.draw(self.graph, labels=labels, node_size=300, k=0.15, node_shape="s", font_size=8, node_color='skyblue')
        plt.savefig("images/" + self.ig + "_mushroom_graph_" + str(dataset_num) + "_.png")
        plt.show()

        print(nx.info(self.graph))
