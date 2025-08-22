from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    assert y.size > 0, "Input series cannot be empty"
    return (y_hat == y).mean()
    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    assert y.size > 0, "Input series cannot be empty"
    TP = ((y_hat == cls) & (y == cls)).sum()
    FP = ((y_hat == cls) & (y != cls)).sum()
    if (TP + FP) == 0:
        return 0.0  # avoid division by zero
    return TP / (TP + FP)



def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    assert y.size > 0, "Input series cannot be empty"
    TP = ((y_hat == cls) & (y == cls)).sum()
    FN = ((y_hat != cls) & (y == cls)).sum()
    if (TP + FN) == 0:
        return 0.0
    return TP / (TP + FN)

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    assert y.size > 0, "Input series cannot be empty"
    return ((y_hat - y) ** 2).mean() ** 0.5




def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    assert y.size > 0, "Input series cannot be empty"
    return (y_hat - y).abs().mean()