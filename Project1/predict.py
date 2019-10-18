def predict(tree, row):
    # Step 1: search the tree for value in the test set.  It is possible that the value of the feature will not have
    # been seen by the model, in this case, need to return a default.
    for test_val in row:
        print(tree.split_value)
        att_val = row[tree.split_value]
        if att_val[0] not in tree.feature_values:
            # in case the attribute stores a value that the model hasn't seen
            # add a value that will always show incorrect, but not misidentified.
            print(test_val[0])
            return 'x'
        else:
            # Step 2: if we are at a leaf, get the value
            if not tree.children:
                return tree.result

            else:
                for child in tree.children:
                    if test_val[0] == child.edge_value:
                        return predict(child, row)
