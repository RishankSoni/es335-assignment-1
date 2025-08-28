"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

# importing all the necessary libraries
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *
import pandas as pd
np.random.seed(42)


@dataclass

#defining the structure of a node of the decision tree
class Node:
    def __init__(self, features=None, threshold=None, left=None, right=None, value=None,gain=0):
        
        self.features = features  # Features to split on
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value for leaf nodes
        self.gain = gain  # Information gain for the split
        
    
    def check_leaf(self):
        return self.value is not None

# defining the structure of the decision tree
class DecisionTree:


    criterion: Literal["information_gain", "gini_index"]  # criterion that will be used for splitting the discrete attributes
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion #storing the criterion that will be used to split discrete attributes
        self.max_depth = max_depth # storing the max_depth of tree which if not specifically mentioned will be 5 
        self.trained_features = [] # To store column names after encoding

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

      
        # building the tree by using recursion by the algorithm explained in the class
        def build_tree(X, y, depth=0):
            if len(y.unique()) == 1 or depth >= self.max_depth: # if all attributes of y are same or we have achieved max_depth defined we make the node as a leaf
                if check_ifreal(y): # if y is real store its mean value as value of leaf
                    return Node(value=y.mean())
                else:
                    return Node(value=y.mode()[0]) # if y is discrete store its mode value as value of leaf
            best_attr = opt_split_attribute(X, y, self.criterion, X.columns)
            
            if best_attr is None: # if there is no attribute left to split upon we make the node as a leaf
                if check_ifreal(y): # if y is real store its mean value as value of leaf
                    return Node(value=y.mean())
                else:
                    return Node(value=y.mode()[0]) # if y is discrete store its mode value as value of leaf
            
            if check_ifreal(X[best_attr]): # if we find a best atrribute split the data on the basis of optimal value
                opt_val= best_threshold(X[best_attr],y,self.criterion) # for real attributes finding the optimal threshold and then splitting
            else:
                opt_val = X[best_attr].mode()[0] # for discrete attributes finding the optimal threshold is found by the mode
            left_indices,right_indices=split_data(X, y, best_attr, opt_val)
            
            if left_indices.empty or right_indices.empty: # if after split any of the data is empty we make the node as a leaf
                if check_ifreal(y):
                    return Node(value=y.mean())
                else:
                    return Node(value=y.mode()[0])
            
            left_node = build_tree(X.loc[left_indices], y.loc[left_indices], depth + 1)
            right_node = build_tree(X.loc[right_indices], y.loc[right_indices], depth + 1)

            
            best_gain = information_gain(y, X[best_attr], self.criterion)
            
            return Node(features=best_attr, threshold=opt_val, left=left_node, right=right_node, value=None,gain=best_gain)
        
        #converting discrete features to one hot encoding
        X=one_hot_encoding(X)
        self.trained_features = X.columns.tolist()
        self.root=build_tree(X,y,0)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        def traverse_tree(node, x):
            if(node.check_leaf()):
                return node.value
            if(check_ifreal(x[node.features])==False):
                if x[node.features] == node.threshold:
                    return traverse_tree(node.left, x)
                else:
                    return traverse_tree(node.right, x)
            else:
                if x[node.features] <=node.threshold:
                    return traverse_tree(node.left, x)
                else:
                    return traverse_tree(node.right, x)
                
        X_encoded = one_hot_encoding(X)
        for col in self.trained_features:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[self.trained_features]
        
            

        return pd.Series([traverse_tree(self.root, x) for _, x in X_encoded.iterrows()])  



    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        def plot_tree(node,depth=0):
            if node.check_leaf():
                print("\t"*depth+"Leaf: Value = "+str(node.value))
            else:
                print("\t"*depth+"?( "+str(node.features)+" <= "+str(node.threshold)+" ) Gain: "+str(node.gain))
                plot_tree(node.left,depth+1)
                plot_tree(node.right,depth+1)
        if(self.root is not None):
            plot_tree(self.root)
        else:
            print("The tree has not been trained yet.")
     
