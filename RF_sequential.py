'''
Preprocessing data:
  - Convert pixel to integer:
    - 0 - 50 : 1
    - 51 - 100: 2
    - 101 - 150: 3
    - 151 - 200: 4
    - 201 - 256: 5
'''

# -------------------------- import --------------------------------------
import numpy as np
import pandas as pd

# ---------------------------- data preprocessing ------------------------
def preprocessing(data):
  for c in data.columns:
    if c != 'label':
        column = [int(i/50) for i in data[c]]
        data[c] = column
  return data

# -------------------------- code for training ----------------------------
'''
Calculate entropy in a node
  - dataset (DataFrame): attr_1, attr_2,...., label
  - attr_to_calc: name of column want to calculate entropy
'''
def Entropy(dataset, attr_to_calc):

  entropy = 0
  parent_length = dataset.shape[0]
  # each unique value is a child node to calculate entropy
  uniq_values = np.unique(dataset[attr_to_calc])
  for v in uniq_values:
    # get all rows have value is v
    records = [[dataset[attr_to_calc][i], dataset['label'][i]] for i in range(parent_length) if dataset[attr_to_calc][i] == v]

    child_entropy = 0
    #get all labels of child node
    child_labels = [r[1] for r in records]
    labels, counts = np.unique(child_labels, return_counts=True)
    # if more than one label (else child_entropy = 0)
    if len(counts) > 1:
        child_length = len(child_labels)
        for c in counts:
          child_entropy = child_entropy -  (c/child_length)*np.log10(c/child_length)
        entropy = entropy + (child_length/parent_length)*child_entropy
  return entropy

# select random data from dataset to train a Tree
# input: DataFrame
# output: DataFrame

def Bootstrapping(dataset):
  index = range(0, dataset.shape[0])
  random_index = np.random.choice(index, dataset.shape[0])

  random_data = [dataset.iloc[i] for i in random_index]
  bootstrapping_dataset = pd.DataFrame(data = random_data, columns = dataset.columns)
  bootstrapping_dataset.reset_index(drop=True, inplace=True)
  return bootstrapping_dataset

# select random attributes from dataset to train a Tree
# input: DataFrame
# output: all columns name to build tree
def Attributes2BuildTree(dataset):
  columns = dataset.columns[:-1] # Can't choose column 'label' to build tree
  num_of_select = int(np.sqrt(len(columns)))

  attributes = np.random.choice(columns, num_of_select)
  return attributes


# Attribute to create a new node or split (attribute that has minimum entropy)
# input: DataFrame
# output: name of attribute will be use to split
def Attribute2Split(dataset):
  attributes = Attributes2BuildTree(dataset)
  attribute_split = '' # name of attribute to split

  min_entropy = 1
  for i in attributes:
    cur_entropy = Entropy(dataset, i)
    if min_entropy > cur_entropy:
      min_entropy = cur_entropy
      attribute_split = i
  return attribute_split


# recursion to create a tree from a dataset
def Tree(dataset):
  tree = []
  new_node = Attribute2Split(dataset)
  tree.append(new_node)

  child_values = np.unique(dataset[new_node]) # all distinct values --> all -child branches
  if len(child_values) < 2:
    return tree
  for value in child_values:
    child_dataset = dataset[dataset[new_node] == value]
    child_dataset.reset_index(drop=True, inplace=True)
    tree.append([value, Tree(child_dataset)])

  return tree

# create a random forest from a dataset
def RandomForest(dataset):
  forest = []
  for t in range(1,max_tree):
    tree_dataset = Bootstrapping(dataset)
    forest.append(Tree(tree_dataset))

  return forest


# ------------------------------- code for prediction ----------------------------------------
def tree_prediction(datapoint, tree):
  if len(tree) < 2: # this is leaf node
    return tree[0]
  value = datapoint[tree[0]].values[0]
  datapoint = datapoint.drop(columns = [tree[0]])
  tree.remove(tree[0])

  available_values = [branch[0] for branch in tree]
  available_values.sort()

  value_2_split = available_values[-1]
  for v in available_values:
    if (value == v) or (value < v):
        value_2_split = v
        break

  for b in tree:
    if b[0] == value_2_split:
      return tree_prediction(datapoint, b[1])


def prediction(datapoint, forest):
  tree_results = []
  for tree in forest:
    tree_results.append(tree_prediction(datapoint, tree))

  labels, counts = np.unique(tree_results, return_counts=True)

  max_voted = counts[0]
  position = 0
  for i in range(len(counts)):
    if counts[i] > max_voted:
      postion = i
      max_voted = counts[i]

  return labels[position]
