import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from stats_analysis import stratified_subsample
import time
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

def svm_model(X, y, kernel = 'linear'):
    if kernel == 'linear':
        svm = LinearSVC(dual=False)
    else:
        svm = SVC(kernel=kernel, n_jobs=-1)
    svm.fit(X, y)
    return svm
    
def confusion_plot(y, y_pred, name = None):
    cm = confusion_matrix(y, y_pred)
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig("./plots/confusion_matrix_"+str(name)+".pdf", dpi=1200)
    plt.close()

def time_vs_size_plot(lr_times, svm_times, sizes):
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, lr_times, 'g-', alpha=0.6, label="Logistic Regression")
    plt.plot(sizes, svm_times, 'r--', alpha=0.6, label="SVM")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Subsample Size N", fontsize=25)
    plt.ylabel("Time (seconds)", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("./plots/time_vs_size_classical.pdf", dpi=1200)
    plt.close()


if __name__ in "__main__":
    # load processed data
    X = pd.read_csv("../data/subsampled_data_features.csv")
    #X_new = pd.read_csv("./data/subsampled_engineered_features.csv")
    y = pd.read_csv("../data/subsampled_data_labels.csv")
    y = y['morphological_type']
    # train/test split
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)
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
    confusion_plot(y_test, y_pred, name = "logistic_regression")
    print(classification_report(y_test, y_pred))

    # for svm now
    svm = svm_model(X_train, y_train)
    train_accuracy = svm.score(X_train, y_train)
    print("SVM classification accuracy on training set: ", train_accuracy)
    # predict
    y_pred = svm.predict(X_test)
    # calculate accuracy
    accuracy = calculate_accuracy(y_test, y_pred)
    print("SVM classification accuracy on test set: ", accuracy)
    # confusion matrix
    confusion_plot(y_test, y_pred, name = "svm")
    print(classification_report(y_test, y_pred))

    # various subsample sizes
    sizes = [1000,2000,5000,10000,15000,20000,30000,50000,80000,100000,120000,150000,180000,200000]
    lr_times, svm_times = [], []
    # Combine X and y into a single DataFrame
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
        svm = svm_model(X_train, y_train)
        t4 = time.time()
        svm_times.append(t4-t3)
    time_vs_size_plot(lr_times=lr_times, svm_times=svm_times, sizes=sizes)
     

    
