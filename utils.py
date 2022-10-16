import numpy as np
from numpy import ndarray
from typing import List, Tuple, Callable


Array_Function = Callable[[ndarray], ndarray]

Chain = List[Array_Function]

def deriv(func: Callable[[ndarray], ndarray], x: ndarray, h: float = 1e-5) -> ndarray:
    """Compute the derivative of a function at a point.

    Args:
        func (Callable[[ndarray], ndarray]): The function to compute the derivative of.
        x (ndarray): The point to compute the derivative at.
        h (float, optional): The step size. Defaults to 1e-5.

    Returns:
        ndarray: The derivative of the function at the point.
    """
    return (func(x + h) - func(x - h)) / (2 * h)


def chain_deriv(chain: Chain, x: ndarray, h: float = 1e-5) -> ndarray:
    """Compute the derivative of a chain of functions at a point.

    Args:
        chain (Chain): The chain of functions to compute the derivative of.
        x (ndarray): The point to compute the derivative at.
        h (float, optional): The step size. Defaults to 1e-5.

    Returns:
        ndarray: The derivative of the chain of functions at the point.
    """
    return deriv(lambda x: chain[-1](x), x, h) * chain[-2](x)

def mae(y_true: ndarray, y_pred: ndarray) -> float:
    """Compute the mean absolute error.

    Args:
        y_true (ndarray): The true values.
        y_pred (ndarray): The predicted values.

    Returns:
        float: The mean absolute error.
    """
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: ndarray, y_pred: ndarray) -> float:
    """Compute the root mean squared error.

    Args:
        y_true (ndarray): The true values.
        y_pred (ndarray): The predicted values.

    Returns:
        float: The root mean squared error.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def forward_linear_regression(X: ndarray, w: ndarray, b: float) -> ndarray:
    """Compute the forward pass of a linear regression model.

    Args:
        X (ndarray): The input data.
        w (ndarray): The weights.
        b (float): The bias.

    Returns:
        ndarray: The predictions.
    """
    return X @ w + b

def backward_linear_regression(X: ndarray, w: ndarray, b: float, y_true: ndarray, y_pred: ndarray) -> Tuple[ndarray, ndarray, float]:
    """Compute the backward pass of a linear
    regression model.

    Args:
        X (ndarray): The input data.
        w (ndarray): The weights.
        b (float): The bias.
        y_true (ndarray): The true values.
        y_pred (ndarray): The predicted values.

    Returns:
        Tuple[ndarray, ndarray, float]: The gradients of the weights, the gradients of the bias, and the loss.
    """
    m = len(y_true)
    loss = rmse(y_true, y_pred)
    dw = (1 / m) * (X.T @ (y_pred - y_true))
    db = (1 / m) * np.sum(y_pred - y_true)
    return dw, db, loss


def train_linear_regression(X: ndarray, y_true: ndarray, lr: float = 0.01, epochs: int = 1000) -> Tuple[ndarray, float]:
    """Train a linear regression model.

    Args:
        X (ndarray): The input data.
        y_true (ndarray): The true values.
        lr (float, optional): The learning rate. Defaults to 0.01.
        epochs (int, optional): The number of epochs. Defaults to 1000.

    Returns:
        Tuple[ndarray, float]: The weights and the bias.
    """
    w = np.random.randn(X.shape[1])
    b = np.random.randn()
    for _ in range(epochs):
        y_pred = forward_linear_regression(X, w, b)
        dw, db, loss = backward_linear_regression(X, w, b, y_true, y_pred)
        w -= lr * dw
        b -= lr * db
    return w, b, loss