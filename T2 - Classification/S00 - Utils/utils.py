import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier

# Setup
resolution = 0.02
markers = ("s", "^", "o", "v", "x")
linestyles = (":", "--", "-.")
colors = ("red", "blue", "lightgreen", "gray", "cyan")


def _check_array(X_train, X_test, y_train, y_test) -> None:
    if (X_test is not None) & (y_test is not None):
        if (
            (type(X_train) != np.ndarray)
            | (type(X_test) != np.ndarray)
            | (type(y_train) != np.ndarray)
            | (type(y_test) != np.ndarray)
        ):
            raise Exception("Plotting Error:X and Y needs to be NumPy arrays.")
        if (len(X_train.shape) != 2) | (len(X_test.shape) != 2):
            raise Exception("Plotting Error: X needs to be a 2D NumPy array.")
        if (X_train.shape[1] != 2) | (X_test.shape[1] != 2):
            raise Exception("Plotting Error: X needs to have 2 columns.")
    else:
        if (type(X_train) != np.ndarray) | (type(y_train) != np.ndarray):
            raise Exception("Plotting Error:X and Y needs to be NumPy arrays.")
        if len(X_train.shape) != 2:
            raise Exception("Plotting Error: X needs to be a 2D NumPy array.")
        if X_train.shape[1] != 2:
            raise Exception("Plotting Error: X needs to have 2 columns.")


def _check_clf(clf) -> None:
    if isinstance(clf, Perceptron):
        fitType = "perceptron"
    # elif isinstance(clf, LogisticRegression):
    #     fitType = "logistic"
    # elif isinstance(clf, SVC):
    #     fitType = "svm"
    # elif isinstance(clf, DecisionTreeClassifier):
    #     fitType = "tree"
    # elif isinstance(clf, RandomForestClassifier):
    #     fitType = "forest"
    # elif isinstance(clf, KNeighborsClassifier):
    #     fitType = "knn"
    # elif isinstance(clf, GaussianNB):
    #     fitType = "bayes"
    # elif isinstance(clf, AdaBoostClassifier):
    #     fitType = "ada"
    # elif isinstance(clf, GradientBoostingClassifier):
    #     fitType = "gb"
    # else:
    #     raise Exception("Unknown classifier: " + type(clf))
    else:
        fitType = "other"
    return fitType


def _get_fig(X_test):
    if X_test is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        return fig, ax1, ax2
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        return fig, ax1, None


def _get_grid_predict(X, clf):
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    return xx1, xx2, Z


def _plot1(clf, X, y, ax, xx1, xx2, title, xlabel, ylabel):
    for idx, (W, w0) in enumerate(zip(clf.coef_, clf.intercept_)):
        hm = -W[0] / W[1]
        hx = np.linspace(xx1.min(), xx1.max())
        hy = hm * hx - (w0) / W[1]
        ax.plot(hx, hy, linestyle=linestyles[idx])
    # Plot data
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.6,
            edgecolor="black",
            marker=markers[idx],
            label=cl,
        )
    ax.set_title(title)
    ax.set_xlim([xx1.min(), xx1.max()])
    ax.set_ylim([xx2.min(), xx2.max()])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def _plot2(X, y, ax, xx1, xx2, Z, title, xlabel, ylabel):
    # Contour plot
    ax.contourf(xx1, xx2, Z, alpha=0.4, cmap="Set3_r")
    # Plot data
    for idx, cl in enumerate(np.unique(y)):
        px = X[y == cl, 0]
        py = X[y == cl, 1]
        ax.scatter(
            x=px,
            y=py,
            alpha=0.6,
            edgecolor="black",
            marker=markers[idx],
            label=cl,
        )
    ax.set_title(title)
    ax.set_xlim([xx1.min(), xx1.max()])
    ax.set_ylim([xx2.min(), xx2.max()])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


# Main function
def plot_ds(X_train, X_test, y_train, y_test, clf, filename=""):
    # Check input
    _check_array(X_train, X_test, y_train, y_test)

    # Check classifier
    fitType = _check_clf(clf)

    # Get grid and prediction on grid
    xx1, xx2, Z = _get_grid_predict(X_train, clf)

    # Figure 1
    if fitType == "perceptron":
        # Get figures
        fig1, ax1, ax2 = _get_fig(X_test)
        _plot1(
            clf,
            X_train,
            y_train,
            ax1,
            xx1,
            xx2,
            "Training Data",
            "Petal Length (Scaled)",
            "Petal Width (Scaled)",
        )
        if X_test is not None:
            _plot1(
                clf,
                X_test,
                y_test,
                ax2,
                xx1,
                xx2,
                "Test Data",
                "Petal Length (Scaled)",
                "Petal Width (Scaled)",
            )

    # Figure 2
    fig2, ax3, ax4 = _get_fig(X_test)
    _plot2(
        X_train,
        y_train,
        ax3,
        xx1,
        xx2,
        Z,
        "Training Data",
        "Petal Length (Scaled)",
        "Petal Width (Scaled)",
    )
    if X_test is not None:
        _plot2(
            X_test,
            y_test,
            ax4,
            xx1,
            xx2,
            Z,
            "Test Data",
            "Petal Length (Scaled)",
            "Petal Width (Scaled)",
        )

    # Save figures
    if filename:
        if fitType == "perceptron":
            fig1.savefig("./" + filename + "_1" + ".png", dpi=300)
        fig2.savefig("./" + filename + "_2" + ".png", dpi=300)
