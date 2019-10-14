class Node:
    def __init__(self):
        self.parent = None  # the parent node for the current node
        self.children = []
        self.split = None  # what information the node is split on.
        self.feature_values = None  # the vales the feature split is based on
        self.leaf = False  # is it a leaf?
        self.result = None  # saved for output
        self.info_gain = 0  # stores the information gain
        self.mode = None  # the data at this node
        self.level = 0  # the depth in the tree

    def __repr__(self):
        return str(self.leaf) + '   ' + str(self.level) + '  ' + str(self.split) + '  ' + self.result + '  ' + \
               str(self.info_gain) + ':   ' + str(self.parent)
