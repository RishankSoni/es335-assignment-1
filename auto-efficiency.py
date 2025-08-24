import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree

from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn
cleaned_data = data.replace('?', np.nan).dropna()
cleaned_data['horsepower'] = cleaned_data['horsepower'].astype(float)
cleaned_data = cleaned_data.drop(columns=['car name', 'origin', 'model year'])
y = cleaned_data['mpg']
X = cleaned_data.drop(columns=['mpg'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Training the decision tree model
model = DecisionTree(criterion='entropy', max_depth=5)


model.fit(X_train, y_train)
# Making predictions
y_pred = model.predict(X_test)
# Evaluating the model
print("Root Mean Squared Error:", rmse(y_pred, y_test))
print("Mean Absolute Error:", mae(y_pred, y_test))


# Comparing with sklearn
sk_learn_model = DecisionTreeRegressor(max_depth=5, random_state=42)
sk_learn_model.fit(X_train, y_train)
y_pred_sklearn = sk_learn_model.predict(X_test)
print("Root Mean Squared Error (sklearn):", rmse(pd.Series(y_pred_sklearn), y_test))
print("Mean Absolute Error (sklearn):", mae(pd.Series(y_pred_sklearn), y_test))
