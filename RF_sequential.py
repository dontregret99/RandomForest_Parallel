import random
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.datasets import fashion_mnist
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

def entropy(labels, count = None):
  num_labels = len(labels)
  
  if num_labels <= 1:
    return 0
  values, counts = np.unique(labels, return_counts=True)
  probs = counts / num_labels
  n_classes = np.count_nonzero(probs)
  if n_classes <=1:
    return 0
    
  entropy = 0
  for i in probs:
    entropy += -i*np.log2(i)

  return entropy

    
def information_gain(left_child, right_child, count = None):
    parent = left_child + right_child
    IG_p = entropy(parent, count)
    IG_l = entropy(left_child, count)
    IG_r = entropy(right_child, count)
    return IG_p - len(left_child) / len(parent) * IG_l - len(right_child) / len(parent) * IG_r

def draw_bootstrap(X_train, y_train):
    bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
    idx_dict = Counter(bootstrap_indices)
    full_set = set(list(range(len(X_train))))
    bootstrap_set = set(list(idx_dict.keys()))
    oob_set = full_set - bootstrap_set
    oob_indices = list(oob_set)
    X_bootstrap = X_train.iloc[bootstrap_indices].values
    y_bootstrap = y_train[bootstrap_indices]
    X_oob = X_train.iloc[oob_indices].values
    y_oob = y_train[oob_indices]
    return X_bootstrap, y_bootstrap, X_oob, y_oob


def oob_score(tree, X_test, y_test):
    mis_label = 0
    for i in range(len(X_test)):
        pred = predict_tree(tree, X_test[i])
        if pred != y_test[i]:
            mis_label += 1
    return mis_label / len(X_test)

def find_split_point(X_bootstrap, y_bootstrap, max_features):
    feature_ls = list()
    num_features = len(X_bootstrap[0])

    while len(feature_ls) <= max_features:
      feature_idx = random.sample(range(num_features), 1)
      if feature_idx not in feature_ls:
          feature_ls.extend(feature_idx)

    best_info_gain = -999
    node = None

    for feature_idx in feature_ls:
      for split_point in X_bootstrap[:,feature_idx]:
          left_child = {'X_bootstrap': [], 'y_bootstrap': []}
          right_child = {'X_bootstrap': [], 'y_bootstrap': []}

          for i, value in enumerate(X_bootstrap[:,feature_idx]):
              if value <= split_point:
                  left_child['X_bootstrap'].append(X_bootstrap[i])
                  left_child['y_bootstrap'].append(y_bootstrap[i])
              else:
                  right_child['X_bootstrap'].append(X_bootstrap[i])
                  right_child['y_bootstrap'].append(y_bootstrap[i])

          split_info_gain = information_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
          if split_info_gain > best_info_gain:
              best_info_gain = split_info_gain
              left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
              right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
              node = {'information_gain': split_info_gain,
                      'left_child': left_child,
                      'right_child': right_child,
                      'split_point': split_point,
                      'feature_idx': feature_idx}

    return node

def terminal_node(node):
    y_bootstrap = node['y_bootstrap']
    pred = max(y_bootstrap, key = y_bootstrap.count)
    return pred


def split_node(node, max_features, min_samples_split, max_depth, depth):
    left_child = node['left_child']
    right_child = node['right_child']

    del(node['left_child'])
    del(node['right_child'])

    if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
        empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
        node['left_split'] = terminal_node(empty_child)
        node['right_split'] = terminal_node(empty_child)
        return

    if depth >= max_depth:
        node['left_split'] = terminal_node(left_child)
        node['right_split'] = terminal_node(right_child)
        return node

    if len(left_child['X_bootstrap']) <= min_samples_split:
        node['left_split'] = node['right_split'] = terminal_node(left_child)
    else:
        node['left_split'] = find_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_features)
        split_node(node['left_split'], max_depth, min_samples_split, max_depth, depth + 1)
    if len(right_child['X_bootstrap']) <= min_samples_split:
        node['right_split'] = node['left_split'] = terminal_node(right_child)
    else:
        node['right_split'] = find_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_features)
        split_node(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)

def build_tree(X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
    root_node = find_split_point(X_bootstrap, y_bootstrap, max_features)
    split_node(root_node, max_features, min_samples_split, max_depth, 1)
    return root_node

def random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
    tree_ls = list()
    oob_ls = list()
    for i in range(n_estimators):
        X_bootstrap, y_bootstrap, X_oob, y_oob = draw_bootstrap(X_train, y_train)
        tree = build_tree(X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split)
        tree_ls.append(tree)
        oob_error = oob_score(tree, X_oob, y_oob)
        oob_ls.append(oob_error)
    print("OOB estimate: {:.2f}".format(np.mean(oob_ls)))
    return tree_ls

def predict_tree(tree, X_test):
    feature_idx = tree['feature_idx']

    if X_test[feature_idx] <= tree['split_point']:
        if type(tree['left_split']) == dict:
            return predict_tree(tree['left_split'], X_test)
        else:
            value = tree['left_split']
            return value
    else:
        if type(tree['right_split']) == dict:
            return predict_tree(tree['right_split'], X_test)
        else:
            return tree['right_split']

def predict_rf(tree_ls, X_test):
    pred_ls = list()
    for i in range(len(X_test)):
        ensemble_preds = [predict_tree(tree, X_test.values[i]) for tree in tree_ls]
        final_pred = max(ensemble_preds, key = ensemble_preds.count)
        pred_ls.append(final_pred)
    return np.array(pred_ls)

if __name__ == '__main__':
    # Download dataset and preprocessing data (div by 255, and reshape to 1-D array), 
    # finnaly convert to DataFrame
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = (x_train / 255.0).reshape(-1, 784)
    x_test = (x_test / 255.0).reshape(-1, 784)
    train_df = pd.DataFrame(x_train)
    train_df['label'] = y_train
    test_df = pd.DataFrame(x_test)
    test_df['label'] = y_test

    print(train_df.head())
    print(test_df.head())

    train_df_demo = train_df.sample(n=1000, random_state=1999)
    X_train = train_df_demo.drop(['label'], axis=1)
    y_train = train_df_demo['label'].values

    print(X_train.info())

    #Hyper params
    n_estimators = 10
    max_features = 10
    max_depth = 5
    min_samples_split = 2
    #Model training
    model = random_forest(X_train, y_train, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split)

    #save
    joblib.dump(model, "my_random_forest.joblib")
    # load
    loaded_model = joblib.load("my_random_forest.joblib")

    #Create test dataframe
    X_test = test_df.drop(['label'], axis=1)
    y_test = test_df['label'].values

    #Predict on test data
    preds = predict_rf(loaded_model, X_test)

    #Caculate accuracy
    acc = sum(preds == y_test) / len(y_test)
    print("Testing accuracy: {}".format(np.round(acc,3))) #RESULT 0.643

    # what a tree look like (the first 3 nodes)
    tree_1 = model[0]
    for k in tree_1.keys():
        if type(tree_1[k]) == dict:
            print('{}: '.format(k))
            for kk in tree_1[k].keys():
                if type(tree_1[k][kk]) == dict:
                    print('\t{}: '.format(kk))
                    for kkk in tree_1[k][kk].keys():
                        print('\t\t{}: {}'.format(kkk, tree_1[k][kk][kkk]))
                    
                else:
                    print('\t{}: {}'.format(kk, tree_1[k][kk]))
        else:
            print('{}: {}'.format(k, tree_1[k]))


    #Define sklearn model with the same hyper params above
    sklearn_model = RandomForestClassifier(n_estimators = n_estimators, 
                                       max_features=max_features, 
                                       max_depth=max_depth, 
                                       min_samples_split =min_samples_split, 
                                       bootstrap=True, 
                                       oob_score=True)

    #Train model
    sklearn_model.fit(X_train, y_train)
    #Predict on test data
    sklearn_preds = sklearn_model.predict(X_test)

    #save
    joblib.dump(model, "sklearn_fr.joblib")
    # load
    loaded_model = joblib.load("sklearn_fr.joblib")

    #Calculate accuracy
    acc = sum(sklearn_preds == y_test) / len(y_test)
    print("Sklearn accuracy: {}".format(np.round(acc,3))) #RESULT: 0.741
