"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    # Making discrete features as different columns with binary values with on hot encoding
    for col in X.columns:
        if not check_ifreal(X[col], int_threshold=10):
            dummies = pd.get_dummies(X[col], prefix=f"is_{col}")
            X = pd.concat([X, dummies], axis=1)
            X.drop(col, axis=1, inplace=True)
    return X


def check_ifreal(y: pd.Series, int_threshold = 5) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if pd.api.types.is_float_dtype(y):
        return True
    if pd.api.types.is_categorical_dtype(y):
        return False
    if pd.api.types.is_integer_dtype(y):
        return len(np.unique(y)) >= int_threshold
    return False




def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """

    probabilities = Y.value_counts(normalize=True)
    entropy_value = 0.0

    for prob in probabilities:
        if prob > 0:
            entropy_value -= prob * np.log2(prob)
    return entropy_value


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    probabilities = Y.value_counts(normalize=True)
    gini_value = 0.0

    for prob in probabilities:
        if prob > 0:
            gini_value += prob * prob
    gini_value = 1 - gini_value
    return gini_value


def Mean_squared_error(y: pd.Series):
    means = y.mean()
    ms = 0.0
    for i in y:
        ms += (means - i) ** 2
    ms /= len(y)
    return ms


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if check_ifreal(attr):
        threshold = best_threshold(attr, Y, criterion)
        left_index,right_index = split_data(attr.to_frame(), Y, attr.name, threshold)

    else :
        left_index,right_index = attr[attr == attr.mode()[0]].index, attr[attr != attr.mode()[0]].index
    left_weight = len(left_index) / len(Y)
    right_weight = len(right_index) / len(Y)
    if len(left_index) == 0 or len(right_index) == 0:
        return 0.0
    if check_ifreal(Y) == False:
        if criterion == "information_gain":
   
            return entropy(Y) - (left_weight * entropy(Y[left_index]) + right_weight * entropy(Y[right_index]))
        elif criterion == "gini_index":
    
            return gini_index(Y) - (left_weight * gini_index(Y[left_index]) + right_weight * gini_index(Y[right_index]))
    else:
        return Mean_squared_error(Y) - (left_weight * Mean_squared_error(Y[left_index]) + right_weight * Mean_squared_error(Y[right_index]))


def best_threshold(X: pd.Series, y: pd.Series, criterion):
    X_sorted = X.sort_values()
    y_sorted = y.loc[X_sorted.index]
    best_gain = -np.inf
    best_thres = None
    for i in range(1, len(X_sorted)):
        if X_sorted.iloc[i] == X_sorted.iloc[i - 1]:
            continue
        threshold = (X_sorted.iloc[i] + X_sorted.iloc[i - 1]) / 2

        left = y_sorted[X_sorted <= threshold]
        right = y_sorted[X_sorted > threshold]

        if len(left) == 0 or len(right) == 0:
            continue

        gain = Mean_squared_error(y) - (
            (len(left) / len(y) * Mean_squared_error(left))
            + (len(right) / len(y) * Mean_squared_error(right))
        )
        if gain > best_gain and gain > 0:
            best_gain = gain
            best_thres = threshold
    return best_thres


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    max_gain = -float("inf")
    best_attr = None
    for feature in features:
        if check_ifreal(X[feature]):
            # If the feature is real-valued, calculate information gain or gini index
            gain = information_gain(y, X[feature], criterion)
            if gain > max_gain and gain > 0:  # Ensure gain is positive
                max_gain = gain
                best_attr = feature
        else:
            gain = information_gain(y, X[feature], criterion)
            if gain > max_gain and gain > 0:
                max_gain = gain
                best_attr = feature
    if best_attr is not None:
        return best_attr

    return None  # If no suitable attribute is found

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    pass


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if check_ifreal(X[attribute]):
        X_left = X[X[attribute] <= value]
        X_right = X[X[attribute] > value]
    else:
        X_left = X[X[attribute] == value]
        X_right = X[X[attribute] != value]

    return X_left.index, X_right.index
