import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import json


def _regress(ax, X, y, y_pred, title, color="steelblue"):
    ax.scatter(X, y, c=color, edgecolor="white", s=70)
    ax.plot(X, y_pred, color="black", lw=2)
    ax.set_title(title)
    return


def plot_reg(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred):
    # Indices for sorting during plotting
    idx_train = np.argsort(X_train, axis=0).ravel()
    idx_test = np.argsort(X_test, axis=0).ravel()

    fig, ax = plt.subplots(
        1, 2, figsize=(10, 5), constrained_layout=True, sharex=True, sharey=True
    )
    _regress(
        ax=ax[0],
        X=X_train[idx_train, :].ravel(),
        y=y_train[idx_train],
        y_pred=y_train_pred[idx_train],
        title="Train",
    )
    _regress(
        ax=ax[1],
        X=X_test[idx_test, :].ravel(),
        y=y_test[idx_test],
        y_pred=y_test_pred[idx_test],
        title="Test",
        color="limegreen",
    )
    fig.supxlabel("Lower status of the population [LSTAT]")
    fig.supylabel("Median value of homes in $1000s [MEDV]")

    return fig, ax


def _residual(ax, y_pred, y, title, color="steelblue"):
    ax.scatter(y_pred, y_pred - y, c=color, marker="o", edgecolor="white", s=70)
    ax.hlines(y=0, xmin=-10, xmax=50, color="black", lw=2)
    ax.axes.set_aspect("equal")
    ax.set_title(title)


def plot_res(y_train, y_test, y_train_pred, y_test_pred):
    fig, ax = plt.subplots(
        1, 2, figsize=(9, 5), constrained_layout=True, sharex=True, sharey=True
    )
    _residual(ax=ax[0], y_pred=y_train_pred, y=y_train, title="Train")
    _residual(ax=ax[1], y_pred=y_test_pred, y=y_test, title="Test", color="limegreen")
    ax[0].set_xlim([-5, 35])
    fig.supxlabel("Predicted values")
    fig.supylabel("Residual")


def isLinearModel(model):
    if (
        isinstance(model, LinearRegression)
        or isinstance(model, Ridge)
        or isinstance(model, Lasso)
        or isinstance(model, ElasticNet)
    ):
        return True
    else:
        return False


def store_results(
    results, model_name, model, y_train, y_test, y_train_pred, y_test_pred, params={}
):
    MSE_train = mean_squared_error(y_train, y_train_pred)
    MSE_test = mean_squared_error(y_test, y_test_pred)

    R2_train = r2_score(y_train, y_train_pred)
    R2_test = r2_score(y_test, y_test_pred)

    coef = None
    intercept = None

    if isLinearModel(model):
        coef = np.linalg.norm(model.coef_)
        intercept = model.intercept_

    data = {
        "Model": model_name,
        "Coef": coef,
        "Intercept": intercept,
        "MSE Train": MSE_train,
        "MSE Test": MSE_test,
        "R2 Train": R2_train,
        "R2 Test": R2_test,
        "Params": json.dumps(params),
    }
    results.append(data)
    return
