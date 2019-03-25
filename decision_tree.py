# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import sys
from collections import defaultdict

from sklearn import tree
from sklearn import metrics 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

def partition(x):
    # x is vector of one attribute with many values
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    # where KEY is the unique value and VALUE is the array of indices where x==VALUE
    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    
    dict = defaultdict(list) 
    z = np.unique(x) # returns a list of all unique values in x column vector
    for v in z:
        dict[v] = np.where(x==v)[0]
    return dict
    raise Exception('Function not yet implemented!')

import math
def entropy(y):
    
    # for the choosen attribute
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    #neglect # for a particular attribute, we are calculating entropy of y given z and entropy of z
    value,count = np.unique(y,return_counts=True) # returns unique value and its count
    H=0
    for c in count:
        P = (c/y.shape[0])
        H = H + P*math.log(P)
    return H
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    # x is the column of one attribute with many values
    # does it return max mutual info after comparing using all values is the attribute
    #choose particular value which gives max I
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.
    #for arr,count in np.unique(y[np.where(x==vx)],return_counts=True):
    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    Hy = entropy(y)
    #Hygivenx = {}
    Ixy = defaultdict(int)   # without default dict it is giving key error 2
    for vx,cx in zip(np.unique(x,return_counts=True)[0],np.unique(x,return_counts=True)[1]):
        Hygivenx = -(cx/x.shape[0])*entropy(y[np.where(x==vx)[0]]) - (1-cx/x.shape[0])*entropy(y[np.where(x!=vx)[0]])
        Ixy[vx] = Hy-Hygivenx
    
    return Ixy

    # INSERT YOUR CODE HERE
    # passing x as key value pair
    """
    vx,cx = np.unique(x[0],return_counts=True)
    num = cx[vx==x[1]]
    Hygivenx = (num/x[0].shape[0])*entropy(y[np.where(x[0]==x[1])]) + (1-num/x[0].shape[0])*entropy(y[np.where(x[0]!=x[1])])
    Hy = entropy(y)
    Ixy = Hy-Hygivenx
    return Ixy
    """
    raise Exception('Function not yet implemented!')

# input is : the training data and maximum depth of the tree to be learned.
def id3(x, y, attribute_value_pairs, depth, max_depth):
              
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),                                       // assumming x1,x2 is the column number in x
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """ 
    

    if(y.size==0):
        return 'null'
        # If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
    u,count = np.unique(y,return_counts=True)
    if(u.shape[0]==1):
        return u[0]
    
    if(attribute_value_pairs == [] or depth == max_depth):
        return u[count==count.max()][0]
    
    # select the best attribute value pair using information gain
    # returns the max gain and pair used for getting maximum mutual information
    maxgain = -sys.maxsize -1
    for kvpair in attribute_value_pairs:
        index = kvpair[0]
        X = x[:,index]                                          # X is (432,)  array([])
        val = kvpair[1]
        MI = mutual_information(X, y)[val]
        if( MI > maxgain ):
            maxgain = MI
            maxpair = kvpair
           
    attribute_value_pairs.remove(maxpair);                  # remove the maxpair from attribute_value_pairs
    
    """
    v = 
    col = maxpair[0]
    vect = x[:,maxpair[0]]
    pvec = partition(x[:,maxpair[0]])
    """
    indexes = partition(x[:,maxpair[0]])[maxpair[1]]  #indexes that belong to pair with max gain
    # we have 0 at indexes , so we dont need 0 after x[indexes:]
    mask = np.ones(x.shape[0],bool)
    mask[indexes] = False
                          
    # a leaf node is created and labelled with the most common class of the examples in the parent node's set. 
    """
    if(len(indexes)==0):
        root = {(maxpair[0],maxpair[1],True):u[count==count.max()][0],
                    (maxpair[0],maxpair[1],False):id3(x,y,attribute_value_pairs, depth+1, max_depth)}
    if(len(indexes) == x.shape[0]):
        root = {(maxpair[0],maxpair[1],True):id3(x, y, attribute_value_pairs, depth+1, max_depth),
                    (maxpair[0],maxpair[1],False):u[count==count.max()][0]}
        """
    #else:
    a = id3(x[indexes,:], y[indexes], attribute_value_pairs, depth+1, max_depth)
    if(a=='null'):
        a = u[count==count.max()][0]
    b = id3(x[mask,:],y[mask],attribute_value_pairs, depth+1, max_depth)
    if(b=='null'):
        b = u[count==count.max()][0]
    root = {(maxpair[0],maxpair[1],True):a,(maxpair[0],maxpair[1],False):b}

    return root

    #partition(x[:,maxpair[0]])[maxpair[1]]        # indexes that belong to pair with max gain
    #x[partition(x[:,maxpair[0]])[maxpair[1]],:]    # data_x that belong to pair with max gain
    #y[partition(x[:,maxpair[0]])[maxpair[1]],:]    # data_y that belong to pair with max gain
    #x[mask]                                         # data_x that doesnot belong to pair with max gain
    #y[mask]                                         # data_y that doesnot belong to pair with max gain
    

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.
    Returns the predicted label of x according to tree
    """
    
    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # tree[attr,value,t/f]
    #label = list(tree.values())[0]
    #if(label==0 or label==1): 
    #    return label 
    lst = list(tree.keys())[0]
    i = lst[0]
    v = lst[1]
    if(x[i] == v):
        label = tree.get((i,v,True))
        if(label==0 or label == 1):
            return label
        return predict_example( x,label )
    else: 
        label = tree.get((i,v,False))
        if(label==0 or label==1):
            return label
        return predict_example( x,label ) 
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    sum=0
    n = y_true.shape[0]
    for i in range(n):
        sum = sum + (y_true[i] != y_pred[i])
    return sum/n
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """ 
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

# used to generate.png images of the trees learned by both scikit-learn and your code.
            
def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid

 
import matplotlib.pyplot as plt
def plotlearningcurves(path_trn,path_tst):
    trn_err = {}
    tst_err = {}
    for i in range(1,11): #iterating over depths = 1 to 10
        
        M = np.genfromtxt(path_trn, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]
        attribute_value_pairs = findattribute_value_pairs(Xtrn)
        decision_treei = id3(Xtrn, ytrn,attribute_value_pairs,0,i)

        # Load the test data
        M = np.genfromtxt(path_tst, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]
        # Compute the test error
        y_pred_tst = [predict_example(x, decision_treei) for x in Xtst]
        tst_err[i] = compute_error(ytst, y_pred_tst)
        # Compute the train error
        y_pred_trn = [predict_example(x, decision_treei) for x in Xtrn]
        trn_err[i] = compute_error(ytrn, y_pred_trn)
    #tst_err , trn_err , d
    plt.figure()
    plt.title(path_trn[2:9])
    plt.plot(tst_err.keys(), tst_err.values(), marker='v', linewidth=3, markersize=12)
    plt.plot(trn_err.keys(), trn_err.values(), marker='o', linewidth=3, markersize=12)
    plt.xlabel('values of depth',fontsize=16)
    plt.ylabel('training error/test error',fontsize=16)
    plt.xticks(list(trn_err.keys()), fontsize=12)
    plt.legend(['Training Error','Test Error'],fontsize=16)

from sklearn.metrics import confusion_matrix
def confusionmatrix(path_trn,path_tst):
    for d in range(1,6,2):
        M = np.genfromtxt(path_trn, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]
        attribute_value_pairs = findattribute_value_pairs(Xtrn)
        decision_tree = id3(Xtrn, ytrn,attribute_value_pairs,0,d)
        print('\n')
        print(f'TREE FOR depth {d}')
        pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, f'./C. tree for  at depth {d}')
        # Load the test data
        M = np.genfromtxt(path_tst, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = M[:, 0]
        Xtst = M[:, 1:]
        # Compute the test error
        y_pred_tst = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred_tst)
        print(f'test error at depth {d}: {tst_err}')
        print(f'Confusion Matrix for depth {d}')
        print(confusion_matrix(ytst,y_pred_tst))
        
        

def findattribute_value_pairs(Xtrn):
       # Learn a decision tree of depth 3
    attribute_value_pairs = []                              # make aatribute paits instead of hard coding
    """
    attribute_value_pairs = [(0,1),(0,2),(0,3),                                
                             (1,1),(1,2),(1,3),
                             (2,1),(2,2),
                             (3,1),(3,2),(3,3),
                             (4,1),(4,2),(4,3),(4,4),
                             (5,1),(5,2)]
    """
    #make a list of attribute value pairs from Xtrn
    for i in range(Xtrn.shape[1]):
        for j in np.unique(Xtrn[:,i]):
            attribute_value_pairs.append((i,j))
    return attribute_value_pairs


def sklearnd135(path_trn,path_tst):
    for d in range(1,6,2):
        clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=d)
        M = np.genfromtxt(path_trn, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytrn = M[:, 0]
        Xtrn = M[:, 1:]
        clf = clf.fit(Xtrn, ytrn)
            # Load the test data
            
        Mt = np.genfromtxt(path_tst, missing_values=0, skip_header=0, delimiter=',', dtype=int)
        ytst = Mt[:, 0]
        Xtst = Mt[:, 1:]
        y_pred = clf.predict(Xtst) 
        
        
        print(f'Mean Absolute SkLearn Error at maxdepth {d} :', metrics.mean_absolute_error(ytst, y_pred))  
        
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,  
                        filled=True, rounded=True,
                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        graph.write_png(f'sklearntree {path_trn[2:8]} depth {d}.png')
        Image(graph.create_png())

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    attribute_value_pairs = []                              # make aatribute paits instead of hard coding
    """
    attribute_value_pairs = [(0,1),(0,2),(0,3),                                
                             (1,1),(1,2),(1,3),
                             (2,1),(2,2),
                             (3,1),(3,2),(3,3),
                             (4,1),(4,2),(4,3),(4,4),
                             (5,1),(5,2)]
    """
    #make a list of attribute value pairs from Xtrn
    for i in range(Xtrn.shape[1]):
        for j in np.unique(Xtrn[:,i]):
            attribute_value_pairs.append((i,j))
    #print(attribute_value_pairs)  
    max_depth = 5    
    decision_tree = id3(Xtrn, ytrn,attribute_value_pairs,0,max_depth)
    # Pretty print it to console
    print('\n')
    print(f'TREE FOR depth {max_depth}')
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    
    tst_err = compute_error(ytst, y_pred)
   
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    
    """  PART C"""
    print("c. Confusion Matrix and decision tree for depth 1,3,5 for monks-1")
    confusionmatrix('./monks-1.train','./monks-1.test')

    """ PART D """ 
    """
    d. (scikit-learn, 20 points) For monks-1, use scikit-learn's DecisionTreeClassifier3 to learn a decision
    tree using criterion='entropy' for depth = 1; 3; 5. You may use scikit-learn's confusion matrix()
    function4.
    """
    print(f'd. sklearn on monks-1 on depth: 1,3,5')
    sklearnd135('./monks-1.train','./monks-1.test')
    
    """ PART E"""
    """
    e. Repeat steps (c) and (d) with your \own" data set and report the confusion
    matrices. You can use other data sets in the UCI repository.
    """
    print("e. Confusion Matrix and decision tree for depth 1,3,5 for SPECT")
    confusionmatrix('./other_data/SPECT.train.txt','./other_data/SPECT.test.txt')
    print(f'e. sklearn on SPECT on depth: 1,3,5')
    sklearnd135('./other_data/SPECT.train.txt','./other_data/SPECT.test.txt')
    
    """  PART B"""
    print("\n")
    print("b. LEARNING CURVES: ")
    
    plotlearningcurves('./monks-1.train','./monks-1.test')
    plotlearningcurves('./monks-2.train','./monks-2.test')   
    plotlearningcurves('./monks-3.train','./monks-3.test')
    
    
        
   