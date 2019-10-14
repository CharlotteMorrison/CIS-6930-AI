import networkx as nx
import matplotlib.pyplot as plt


class NodeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def new_node(self, node):
        self.graph.add_node(node)

    def new_node_list(self, nodes):
        self.graph.add_nodes_from(nodes)

    def new_edge(self, edge):
        self.graph.add_edge(edge)

    def new_edge_list(self, edges):
        self.graph.add_edges_from(edges)

    def print_nodes(self):
        print("Nodes of the graph: ")
        print(self.graph.nodes())

    def print_edges(self):
        print("Edges of the graph: ")
        print(self.graph.edges())

    def draw_graph(self, dataset_num):
        nx.draw(self.graph)
        plt.savefig("images/mushroom_graph_" + dataset_num + "_.png")
        plt.show()
