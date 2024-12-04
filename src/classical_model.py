import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from stats_analysis import stratified_subsample
import time
from scipy.interpolate import interp1d
warnings.filterwarnings("ignore")

def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into train and test sets.

    Parameters:
    - X: array-like, feature matrix
    - y: array-like, target variable
    - test_size: float, optional (default=0.2), proportion of the dataset to include in the test split
    - random_state: int or None, optional (default=None), seed for random number generator for reproducibility

    Returns:
    - X_train: array-like, feature matrix for training
    - X_test: array-like, feature matrix for testing
    - y_train: array-like, target variable for training
    - y_test: array-like, target variable for testing
    """
    X = np.array(X)
    y = np.array(y)
    # check in lengths are equal
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    # number of samples in test set
    num_test_samples = int(len(X) * test_size)
    # random seed
    if random_state is not None:
        np.random.seed(random_state)
    # shuffle indices
    indices = np.random.permutation(len(X))
    # split indices and data
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def calculate_accuracy(y_true, y_pred):
    """
    Calculate binary classification accuracy.

    Parameters:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted labels

    Returns:
    - accuracy: float, binary classification accuracy
    """
    correct_predictions = (y_true == y_pred).sum()
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


def plot_iteration_accuracy(y_true, y_pred):
    """
    Plots the iteration accuracy curve.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    None
    """
    if y_pred.ndim == 1:
        # single model iteration
        predictions = calculate_accuracy(y_true, y_pred)
    else:
        #multi-prediction 
        predictions = []
        for pred in y_pred:
            predictions.append(calculate_accuracy(y_true, pred))
    plt.figure(figsize=(16, 8))
    plt.plot(range(0, 2500), predictions, 'g-', alpha=0.6)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Iteration", fontsize=25)
    plt.ylabel("Classification Accuracy", fontsize=20)
    plt.savefig("./plots/iteration_accuracy.pdf", dpi=1200)
    plt.close()


def print_summary(fit):
    pass


def logreg_model(X, y):
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr

    
def confusion_plot(y, y_pred, cmap = 'Blues', name = None):
    cm = confusion_matrix(y, y_pred)
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap=cmap, fmt='g', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig("./plots/confusion_matrix_"+str(name)+".pdf", dpi=1200)
    plt.close()

def time_vs_size_plot(lr_times, svm_times, sizes, qsvm_times=None, qlr_times=None, q_sizes=None):
    plt.figure(figsize=(12, 8))
    plt.semilogy(sizes, lr_times, 'g-', alpha=0.6, label="Logistic Regression")
    plt.semilogy(sizes, svm_times, 'r-', alpha=0.6, label="SVM")
    if qsvm_times is not None and qlr_times is not None:
        l = len(q_sizes)
        # Interpolate quantum times to match classical sizes
        qlr_interp = interp1d(q_sizes, qlr_times, kind='linear', fill_value="extrapolate")
        qsvm_interp = interp1d(q_sizes, qsvm_times, kind='linear', fill_value="extrapolate")
        # Interpolated times for size 200000
        for s in np.linspace(5000, 200000, 10):
            q_sizes.append(s)
            qlr_times.append(qlr_interp(s))
            qsvm_times.append(qsvm_interp(s))
        plt.semilogy(q_sizes[:l], qlr_times[:l], 'b^-', alpha=0.6, label="Quantum Logistic Regression")
        plt.semilogy(q_sizes[:l], qsvm_times[:l], 'C1x-', alpha=0.6, label="Quantum SVM")
        plt.semilogy(q_sizes[l:], qlr_times[l:], 'b--', alpha=0.6)
        plt.semilogy(q_sizes[l:], qsvm_times[l:], 'C1--', alpha=0.6)
        plt.annotate(r"$QLR$", (q_sizes[5], qlr_times[5]), textcoords="offset points", xytext=(-20, 10), fontsize=15, color='blue')
        plt.annotate(r"$QSVM$", (q_sizes[8], qsvm_times[8]), textcoords="offset points", xytext=(-20, -15), fontsize=15, color='orange')
    mid = len(sizes) // 2
    plt.annotate(r"$O(knd)$", (sizes[mid], lr_times[mid]), textcoords="offset points", xytext=(-20, 10), fontsize=15, color='green')
    plt.annotate(r"$O(n^3)$", (sizes[mid], svm_times[mid]), textcoords="offset points", xytext=(-20, -15), fontsize=15, color='red')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Subsample Size N", fontsize=25)
    plt.ylabel("Log Time (seconds)", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("./plots/time_vs_size_classical.pdf", dpi=1200)
    plt.close()

def svm_model(X, y, kernel = 'rbf', C = 50):
    clf = SVC(kernel = kernel, C = C)
    clf.fit(X, y)
    return clf
def decision_boundary_plot(data, f1='C', f2='A'):
    """
    Plots the decision boundary for an SVM model with RBF kernel.
    
    Parameters:
    data (DataFrame): Input data containing features and labels.
    f1 (str): Feature 1 for the x-axis.
    f2 (str): Feature 2 for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[data['morphological_type'] > 0][f1], 
                data[data['morphological_type'] > 0][f2], 
                color='blue', label='Spiral')
    plt.scatter(data[data['morphological_type'] == 0][f1], 
                data[data['morphological_type'] == 0][f2], 
                color='red', label='Elliptical', alpha=0.6)
    xx, yy = np.meshgrid(
        np.linspace(data[f1].min() - 1, data[f1].max() + 1, 500),
        np.linspace(data[f2].min() - 1, data[f2].max() + 1, 500)
    )
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[f1, f2])
    m = svm_model(data[[f1, f2]], data['morphological_type'])
    Z = m.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title('SVM Decision Boundary with RBF Kernel')
    plt.legend()
    plt.savefig("./plots/decision_boundary_svm.pdf", dpi=1200)
    plt.close()


if __name__ in "__main__":
    # load processed data
    X = pd.read_csv("../data/data_features.csv")
    #X = pd.read_csv("../data/engineered_features.csv")
    y = pd.read_csv("../data/data_labels.csv")
    y = y['morphological_type']
    df = pd.concat([X, y], axis=1)
    subsampled_df = stratified_subsample(df, "morphological_type", total_samples=10000, random_state=42)
    y = subsampled_df["morphological_type"]
    X = subsampled_df.drop(columns = ["morphological_type"])
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    svm = svm_model(X_train, y_train)
    train_accuracy = svm.score(X_train, y_train)
    print("SVM classification accuracy on training set: ", train_accuracy)
    # predict
    y_pred = svm.predict(X_test)
    # calculate accuracy
    accuracy = calculate_accuracy(y_test, y_pred)
    print("SVM classification accuracy on test set: ", accuracy)
    #decision_boundary_plot(subsampled_df)
    confusion_plot(y_train, svm.predict(X_train), name = "svm", cmap = 'Greens')
    print(classification_report(y_train, svm.predict(X_train)))

    # train model
    lr = logreg_model(X_train, y_train)
    train_accuracy = lr.score(X_train, y_train)
    print("Log Reg classification accuracy on training set: ", train_accuracy)
    # predict   
    y_pred = lr.predict(X_test)
    # calculate accuracy
    accuracy = calculate_accuracy(y_test, y_pred)
    print("Log Reg classification accuracy on test set: ", accuracy)
    # confusion matrix
    confusion_plot(y_train, lr.predict(X_train), name = "logistic_regression")
    print(classification_report(y_train, lr.predict(X_train)))


    # various subsample sizes
    sizes = [1000,5000,10000,15000,20000,30000,50000,80000,100000,120000,150000,180000,200000]
    lr_times, svm_times = [], []
    # Combine X and y into a single DataFrame
    #X = pd.read_csv("../data/engineered_features.csv")
    X = pd.read_csv("../data/data_features.csv")
    y = pd.read_csv("../data/data_labels.csv")
    y = y['morphological_type']
    df = pd.concat([X, y], axis=1)
    #
    for size in sizes:
        subsampled_df = stratified_subsample(df, "morphological_type", size, random_state=42)
        y = subsampled_df["morphological_type"]
        X = subsampled_df.drop(columns = ["morphological_type"])
        # train/test split
        X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)
        # train model
        t1 = time.time()
        lr = logreg_model(X_train, y_train)
        t2 = time.time()
        lr_times.append(t2-t1)
        t3 = time.time()
        svm = svm_model(X_train, y_train, 'linear', 1.0)
        t4 = time.time()
        svm_times.append(t4-t3)
    time_vs_size_plot(lr_times=lr_times, svm_times=svm_times, sizes=sizes)


    ##### just to compare size efficacy with quantum models
    X = pd.read_csv("../data/data_features.csv")
    #X_new = pd.read_csv("./data/subsampled_engineered_features.csv")
    y = pd.read_csv("../data/data_labels.csv")
    y = y['morphological_type']
    df = pd.concat([X, y], axis=1)
    subsampled_df = stratified_subsample(df, "morphological_type", total_samples=200, random_state=42)
    y = subsampled_df["morphological_type"]
    X = subsampled_df.drop(columns = ["morphological_type"])
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    svm = svm_model(X_train, y_train)
    train_accuracy = svm.score(X_train, y_train)
    print("SVM classification accuracy on training set: ", train_accuracy)
    # predict
    y_pred = svm.predict(X_test)
    # calculate accuracy
    accuracy = calculate_accuracy(y_test, y_pred)
    print("SVM classification accuracy on test set: ", accuracy)
    #confusion_plot(y_train, svm.predict(X_train), name = "svm")
    print(classification_report(y_train, svm.predict(X_train)))

    # train model
    lr = logreg_model(X_train, y_train)
    train_accuracy = lr.score(X_train, y_train)
    print("Log Reg classification accuracy on training set: ", train_accuracy)
    # predict   
    y_pred = lr.predict(X_test)
    # calculate accuracy
    accuracy = calculate_accuracy(y_test, y_pred)
    print("Log Reg classification accuracy on test set: ", accuracy)
    # confusion matrix
    #confusion_plot(y_train, lr.predict(X_train), name = "logistic_regression")
    print(classification_report(y_train, lr.predict(X_train)))
     

 #
