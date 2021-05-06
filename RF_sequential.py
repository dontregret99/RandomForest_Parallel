'''
Preprocessing data:
  - Convert pixel to integer:
    - 0 - 50 : 1
    - 51 - 100: 2
    - 101 - 150: 3
    - 151 - 200: 4
    - 201 - 256: 5
'''
# -------------------------- code for training ----------------------------

def Entropy(values_of_attribute, labels):

  return entropy


# select random data from dataset to train a Tree
def Bootstrapping(dataset, num_of_point_to_get):

  return random_data


# select random attributes from dataset to train a Tree
def Attributes2BuildTree(dataset):

  return attributes


# Attribute to create a new node or split (attribute that has minimum entropy)
def Attribute2Split(dataset, attributes):

  return attribute_i


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
