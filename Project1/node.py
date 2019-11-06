class Node:
    def __init__(self):
        self.id = 0
        self.children = []          # list of the children nodes
        self.attribute_name = None           # what information the node is split on.
        self.attribute_values = None  # the values the feature split is based on
        self.edge_value = None
        self.result = None          # saved for output
        self.info_gain = 0          # stores the information gain
        self.mode = None
        self.parent = None

    def is_leaf(self):
        return len(self.children) == 0

    def get_level(self):
        levels = [child.get_level() for child in self.children]
        level = max(levels) + 1 if levels else 1
        return level

    def __iter__(self):
        yield self
        for child in self.children:
            yield child

    def __repr__(self):
        return str(self.is_leaf()) + '   ' + str(self.attribute_name) + '  ' + str(self.attribute_values) + '  ' \
               + str(self.result) + '  ' + str(self.info_gain) + '  \n\t' + str(self.children)
