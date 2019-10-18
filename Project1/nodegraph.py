import networkx as nx
import matplotlib.pyplot as plt


class NodeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def new_node(self, node, ig):
        self.graph.add_node(node, info_gain=ig)

    def new_node_list(self, nodes):
        self.graph.add_nodes_from(nodes)

    def new_edge(self, edge):
        self.graph.add_edge(*edge)

    def new_edge_list(self, edges):
        self.graph.add_edges_from(edges)

    def print_nodes(self):
        print("Nodes of the graph: ")
        print(self.graph.nodes())

    def print_edges(self):
        print("Edges of the graph: ")
        print(self.graph.edges())

    def draw_graph(self, dataset_num):
        labels = nx.get_node_attributes(self.graph, 'info_gain')
        nx.draw(self.graph, labels=labels, node_size=1000)
        plt.savefig("images/mushroom_graph_" + str(dataset_num) + "_.png")
        plt.show()
        # other stuff for later
        print(nx.info(self.graph))
