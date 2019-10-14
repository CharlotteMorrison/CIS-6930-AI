import numpy as np
import scipy.stats
import copy

from Project1.node import Node


class ID3Algorithm(object):

    def __init__(self):
        pass

    # find the entropy for the dataset takes in the feature values only
    @classmethod
    def get_entropy(cls, dataset, feature):
        # get the count of the number of each values from the target column (edible?)
        counts = dataset[feature].value_counts(ascending=False).tolist()

        # scipy implementation
        # entropy = scipy.stats.entropy(pk=counts, base=2.0)

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

    def run_id3_algorithm(self, examples, target_attribute, attributes, parent_examples, parent_node=None):
        dataset = copy.deepcopy(examples)
        attribute_list = attributes
        parent_dataset = copy.deepcopy(parent_examples)
        node = Node()
        check_target_values = dataset[target_attribute].count()
        dataset_length, _ = dataset.shape

        # if the dataset has no values return the mode of the target attribute
        if dataset_length == 0:
            node.parent = parent_node
            node.level = parent_node.level + 1
            node.leaf = True
            node.result = parent_examples[0].mode()
            node.parent.leaf = True
            return node

        # if there is only one value in the target column- it is sorted, return the value that it is.
        elif check_target_values <= 1:
            node.parent = parent_node
            node.level = parent_node.level + 1
            node.leaf = True
            node.result = 1
            node.parent.leaf = True
            return node

        # if the attributes are empty, then return the mode of the dataset
        #TODO fix this weirdness.
        elif len(attribute_list) == 0:
            # node.parent = parent_node
            # node.level = parent_node.level + 1
            # node.leaf = True
            # node.result = 1
            # node.parent.leaf = True
            return parent_node

        # if none are true then add to the tree
        else:
            # get the best information gain attribute
            item_values = [self.get_information_gain(parent_dataset, target_attribute, attribute)
                           for attribute in attribute_list]
            best_value = attribute_list[np.argmax(item_values)]

            best_attribute_values = dataset[best_value].value_counts(ascending=False).index.tolist()

            # create the root node and record pertinent data

            node.split = best_value
            node.feature_values = best_attribute_values
            node.info_gain = max(item_values)
            node.parent = parent_node
            node.mode = dataset.loc[:, best_value]

            if node.parent is not None:
                node.level = parent_node.level + 1

            # remove the best value attribute from the list
            attribute_list.remove(best_value)

            # build the decision tree
            for value in best_attribute_values:  # list of attributes from the best one...
                attribute_dataset = dataset.loc[dataset[best_value] == value]

                node.result = value

                subtree = self.run_id3_algorithm(attribute_dataset, target_attribute, attribute_list, parent_examples, node)
                if isinstance(subtree, Node):
                    node.children.append(subtree)
                    print(node)
                    node = subtree

            return node
