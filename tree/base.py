"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *
import pandas as pd
np.random.seed(42)


@dataclass

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
class DecisionTree:


    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 
        def build_tree(X, y, depth=0):
            if len(y.unique()) == 1 or depth >= self.max_depth:
                if check_ifreal(y):
                    return Node(value=y.mean()[0])
                else:
                    return Node(value=y.mode()[0])
            best_attr = opt_split_attribute(X, y, self.criterion, X.columns)
            
            if best_attr is None:
                if check_ifreal(y):
                    return Node(value=y.mean()[0])
                else:
                    return Node(value=y.mode()[0])
            
            if check_ifreal(y):
                opt_val= best_threshold(X[best_attr],y,self.criterion)
            else:
                opt_val = X[best_attr].mode()[0]
            # left_indices = X[best_attr] <= opt_val
            # right_indices = X[best_attr] > opt_val
            left_indices,right_indices=split_data(X, y, best_attr, opt_val)
            left_node = build_tree(opt_val,X[left_indices], y[left_indices], depth + 1)
            right_node = build_tree(opt_val,X[right_indices], y[right_indices], depth + 1)

            
            best_gain = information_gain(y, X[best_attr], self.criterion)
            ##
            return Node(features=best_attr, threshold=opt_val, left=left_node, right=right_node, gain=information_gain(y, X[best_attr], self.criterion), value=None,gain=best_gain)
            

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

        return pd.Series([traverse_tree(self.root, x) for _, x in X.iterrows()])  



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
     
