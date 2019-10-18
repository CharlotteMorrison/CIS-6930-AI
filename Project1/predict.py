def predict(tree, row):
    # Step 1: search the tree for value in the test set.  It is possible that the value of the feature will not have
    # been seen by the model, in this case, need to return a default.
    if tree.is_leaf():
        return tree.result
    else:
        att_name = row[tree.attribute_name]
        if att_name[0] not in tree.attribute_values:
            # in case the attribute stores a value that the model hasn't seen
            # add a value that will always show incorrect, but not misidentified.
            return 'x'
        else:
            for child in tree.children:
                if att_name[0] == child.edge_value:
                    return predict(child, row)
