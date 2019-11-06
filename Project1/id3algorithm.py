import numpy as np
import copy

import pandas as pd

from Project1.nodegraph import NodeGraph
from Project1.node import Node


class ID3Algorithm(object):

    def __init__(self, ig):
        self.graph = NodeGraph(ig)
        # which information gain to use
        self.ig = ig

    # find the entropy for the dataset takes in the feature values only
    @classmethod
    def get_entropy(cls, dataset, feature):
        # get the count of the number of each values from the target column (edible?)
        counts = dataset[feature].value_counts(ascending=False).tolist()
        entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(counts))])
        return entropy

    def get_information_gain(self, dataset, target_attribute, attribute):
        # get the information entropy for the whole dataset on the feature
        entropy = self.get_entropy(dataset, target_attribute)

        # get the values needed to compute the information gain
        counts = dataset[attribute].value_counts(ascending=False).tolist()
        values = dataset[attribute].value_counts(ascending=False).index.tolist()
        total = np.sum(counts)
        attribute_entropy = 0

        # compute the information gain for the attribute
        for i in range(len(counts)):
            dv = counts[i] / total
            temp_entropy = self.get_entropy(dataset.loc[dataset[attribute] == values[i]], target_attribute)
            attribute_entropy = attribute_entropy + dv * temp_entropy
        info_gain = entropy - attribute_entropy
        return info_gain

    def get_c45_gain(self, dataset, target_attribute, attribute):
        dataset_length = len(dataset)
        pre_split_impurity = self.get_entropy(dataset, target_attribute)
        post_split_impurity = 0
        # split the dataset on the value
        attributes = dataset[attribute].unique()
        splits = {att: pd.DataFrame for att in attributes}

        for key in splits.keys():
            splits[key] = dataset[:][dataset[attribute] == key]

        weighting = [len(split)/dataset_length for split in splits]
        for i in range(len(attributes)):
            post_split_impurity += weighting[i] * self.get_entropy(splits[attributes[i]], target_attribute)
        info_gain = pre_split_impurity - post_split_impurity

        return info_gain

    def run_id3_algorithm(self, examples, target_attribute, attributes, parent_examples, parent_node=None):
        dataset = copy.deepcopy(examples)
        attribute_list = copy.deepcopy(attributes)
        parent_dataset = copy.deepcopy(parent_examples)
        node = Node()
        check_target_values = dataset[target_attribute].nunique()
        dataset_length = len(dataset.index)

        # if the dataset has no values return the mode of the target attribute
        if dataset_length == 0:
            node.attribute_name = parent_node.attribute_name
            node.result = parent_examples[target_attribute].mode()[0]
            return node

        # if there is only one unique value in the target column- it is sorted, return the value that it is.
        elif check_target_values <= 1:
            node.attribute_name = parent_node.attribute_name
            node.result = dataset[target_attribute].mode()[0]
            return node

        # if the attributes are empty, then return the mode of the dataset
        elif len(attribute_list) == 0:
            node.attribute_name = parent_node.attribute_name
            node.result = dataset[target_attribute].mode()[0]
            return node

        # if none are true then add to the tree
        else:
            # get the best information gain attribute from id3
            item_values = []
            if self.ig == 'id3':
                item_values = [self.get_information_gain(parent_dataset, target_attribute, attribute)
                               for attribute in attribute_list]
            elif self.ig == 'c45':
                item_values = [self.get_c45_gain(parent_dataset, target_attribute, attribute)
                               for attribute in attribute_list]
            else:
                print('no ig...')
            best_value = attribute_list[np.argmax(item_values)]

            best_attribute_values = dataset[best_value].value_counts(ascending=False).index.tolist()

            # create the the node
            node.attribute_name = best_value
            node.attribute_values = best_attribute_values
            node.info_gain = max(item_values)
            node.children = []

            # remove the best value attribute from the list
            attribute_list.remove(best_value)

            self.graph.new_node(node, node.info_gain, node.attribute_name, node.attribute_values, node.result, node.edge_value)
            # build the decision tree
            for value in best_attribute_values:  # list of attributes from the best one...
                # set up children
                subtree = Node()
                subtree.parent = node

                attribute_dataset = copy.deepcopy(dataset.loc[dataset[best_value] == value])
                subtree = self.run_id3_algorithm(attribute_dataset, target_attribute, attribute_list, parent_examples, node)
                subtree.edge_value = value
                node.children.append(subtree)
                self.graph.new_node(subtree, subtree.info_gain, subtree.attribute_name, subtree.attribute_values,
                                    subtree.result, subtree.edge_value)
                self.graph.new_edge((node, subtree), value)

            return node

    def graph_it(self, index):
        self.graph.draw_graph(index)
