class Node:
    def __init__(self):
        self.children = []          # list of the children nodes
        self.split_value = None           # what information the node is split on.
        self.feature_values = None  # the values the feature split is based on
        self.result = None          # saved for output
        self.info_gain = 0          # stores the information gain
        self.parent = None
        self.leaf = 0

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
        return str(self.is_leaf()) + '   ' + str(self.split_value) + '  ' + str(self.feature_values) + '  ' \
               + str(self.result) + '  ' + str(self.info_gain) + '  \n\t' + str(self.children)
