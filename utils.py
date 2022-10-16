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
    for c in reversed(chain):
        x = deriv(c, x, h)
    return x

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

def forward_linear_regression(X: ndarray, w: ndarray, b: float, chain:Chain) -> ndarray:
    """Compute the forward pass of a linear regression model.

    Args:
        X (ndarray): The input data.
        w (ndarray): The weights.
        b (float): The bias.

    Returns:
        ndarray: The predictions.
    """
    if len(X.shape) == 0:
        X = X.reshape(-1, 1)
    if len(w.shape) == 0:
        w = w.reshape(-1, 1)
    if len(b.shape) == 0:
        b = b.reshape(-1, 1)
    y_pred = X @ w + b
    # apply the function chain
    for c in chain:
        y_pred = c(y_pred)

    return y_pred

def backward_linear_regression(X: ndarray, w: ndarray, b: float, y_true: ndarray, y_pred: ndarray, chain: Chain) -> Tuple[ndarray, ndarray, float]:
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
    if len(X.shape) == 0:
        X = X.reshape(-1, 1)
    if len(w.shape) == 0:
        w = w.reshape(-1, 1)
    if len(b.shape) == 0:
        b = b.reshape(-1, 1)
    m = len(y_true)
    loss = rmse(y_true, y_pred)
    dw = (1 / m) * (X.T @ (y_pred - y_true))
    db = (1 / m) * np.sum(y_pred - y_true)
    # calculate the derivative of the function chain
    dw = chain_deriv(chain, dw)
    db = chain_deriv(chain, db)

    return dw, db, loss


def train_linear_regression(X: ndarray, y_true: ndarray, w:ndarray, b:ndarray, chain:Chain, lr: float = 0.01, epochs: int = 1000) -> Tuple[ndarray, float]:
    """Train a linear regression model.

    Args:
        X (ndarray): The input data.
        y_true (ndarray): The true values.
        lr (float, optional): The learning rate. Defaults to 0.01.
        epochs (int, optional): The number of epochs. Defaults to 1000.

    Returns:
        Tuple[ndarray, float]: The weights and the bias.
    """
    for _ in range(epochs):
        y_pred = forward_linear_regression(X, w, b, chain)
        dw, db, loss = backward_linear_regression(X, w, b, y_true, y_pred, chain)
        # extract dw and db from the chain
        if dw.shape == (1, 1):
            dw = dw[0, 0]
        else:
            dw = dw[0]
        w -= lr * dw
        b -= lr * db
    return w, b, loss, y_pred

def sigmoid(x: ndarray) -> ndarray:
    """Compute the sigmoid function.

    Args:
        x (ndarray): The input.

    Returns:
        ndarray: The output.
    """
    return 1 / (1 + np.exp(-x))

def relu(x: ndarray) -> ndarray:
    """Compute the ReLU function.

    Args:
        x (ndarray): The input.

    Returns:
        ndarray: The output.
    """
    return np.maximum(0, x)

def leacky_relu(x: ndarray) -> ndarray:
    """Compute the leaky ReLU function.

    Args:
        x (ndarray): The input.

    Returns:
        ndarray: The output.
    """
    return np.maximum(0.01 * x, x)