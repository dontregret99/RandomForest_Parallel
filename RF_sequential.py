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


# -------------------------- code for training ----------------------------
'''
Calculate entropy in a node
  - datanode (np.array): just get 2 column: attribute_to_split & label
  - datanode[0]: values
  - datanode[1]: labels
'''

def Entropy(datanode):

  entropy = 0
  parent_length = len(datanode[0])
  # each unique value is a child node to calculate entropy
  uniq_values = np.unique(datanode[0])

  for v in uniq_values:
    # get all rows have value is v
    records = [[datanode[0][i],datanode[1][i]] for i in range(len(datanode[0])) if datanode[0][i] == v]

    child_entropy = 0
    # get all labels of child node
    child_labels = [r[1] for r in records]
    labels, counts = np.unique(child_labels, return_counts=True)

    # if more than one label (else child_entropy = 0)
    if len(counts) > 1:
        child_length = len(child_labels)

        for c in counts:
            child_entropy = child_entropy -  (c/child_length)*np.log2(c/child_length)
        entropy = entropy + (child_length/parent_length)*child_entropy

  return entropy


# select random data from dataset to train a Tree
def Bootstrapping(dataset):
  index = range(0, len(dataset[0]))
  random_index = np.random.choice(index, len(dataset[0])-1)# the last column is label
  random_data = []

  for attribute in dataset:
    column = [attribute[i] for i in random_index]
    random_data.append(column)

  return random_data


# select random attributes from dataset to train a Tree
def Attributes2BuildTree(dataset):
  attribute_count = 0
  for i in dataset:
    attribute_count = attribute_count + 1
  num_of_select = int(np.sqrt(attribute_count))

  attributes = np.random.choice(range(0,attribute_count), num_of_select)
  return attributes


# Attribute to create a new node or split (attribute that has minimum entropy)
def Attribute2Split(dataset):
  index_of_attributes = Attributes2BuildTree(dataset)
  attribute_split = 0 # index of attribute to split

  min_entropy = 1
  for i in index_of_attributes:
    child_data = []
    child_data.append(dataset[i])
    child_data.append(dataset[-1])
    cur_entropy = Entropy(child_data)
    if min_entropy > cur_entropy:
      min_entropy = cur_entropy
      attribute_split = i

  return attribute_split


# recursion to create a tree from a dataset
def Tree(dataset):
  tree = []
  attributes = Attributes2BuildTree(dataset)
  new_node = Attribute2Split(dataset, attributes)
  tree.append(new_node)

  for value in AllValueOfCurrentAttribute(dataset, new_node[0]):
    child_dataset = [d for d in dataset if d[new_node[1]] == value]
    tree.append(Tree(child_dataset))

  return tree


# create a random forest from a dataset
def RandomForest(dataset):
  forest = []
  for t range(1,max_tree):
    tree_dataset = Bootstrapping(dataset)
    forest.append(Tree(tree_dataset))

  return forest
