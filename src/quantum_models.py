import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from qiskit_machine_learning.algorithms.classifiers import VQC, QSVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import Sampler, StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA, ADAM
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Session
from qiskit.visualization import plot_histogram
import pandas as pd
from stats_analysis import stratified_subsample
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from qiskit import transpile
from qiskit_aer import Aer, AerSimulator
from scipy.special import expit  # Sigmoid function
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from qiskit_machine_learning.datasets import ad_hoc_data
import time
from quantum_logistic_regression import train_quantum_logistic_regression, sigmoid, quantum_linear_model, quantum_logistic_regression

def set_up_backend():
    service = QiskitRuntimeService(
    channel="ibm_quantum", 
    token="6207c6bd837ec1f058247abf4bfa048ee64f005d88632388a189c359e46d19f6f8e43b187739698f67eb288ddedf329a38530286553a156bcf9fc9a0ee764747"
    )
    backends = service.backends()
    # find the backend with the fewest pending jobs
    min_pending_jobs = float('inf')
    best_backend = None
    for backend in backends:
        try:
            pending_jobs = backend.status().pending_jobs
            print(f"{backend.name}: Pending jobs = {pending_jobs}")
            if pending_jobs < min_pending_jobs:
                min_pending_jobs = pending_jobs
                best_backend = backend
        except Exception as e:
            print(f"Could not retrieve status for {backend.name}: {e}")
    # set the best backend
    if best_backend:
        print(f"Selected backend: {best_backend.name} with {min_pending_jobs} pending jobs")
        return best_backend
    else:
        print("No suitable backend found. Terminating.")
        exit()
        

def load_data():
    # load processed data
    X = pd.read_csv("../data/data_features.csv")
    #X = pd.read_csv("../data/subsampled_engineered_features.csv")
    n_features = X.shape[1]
    y = pd.read_csv("../data/data_labels.csv")
    y = y['morphological_type']
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=1, entanglement='circular')
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    return quantum_kernel, X, y


def run_multple_sessions_qsvm(quantum_kernel, X, y, backend):
    sizes = [1000,5000,10000,20000]
    times = []
    # Combine X and y into a single DataFrame
    df = pd.concat([X, y], axis=1)
    #
    session = Session(backend=backend)
    try:
        for size in sizes:
            subsampled_df = stratified_subsample(df, "morphological_type", size, random_state=42)
            y = subsampled_df["morphological_type"]
            X = subsampled_df.drop(columns = ["morphological_type"])
            #scaler = StandardScaler()
            #X = scaler.fit_transform(X) 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            qsvc = QSVC(quantum_kernel=quantum_kernel)
            t1 = time.time()
            qsvc.fit(X_train, y_train)
            t2 = time.time()
            times.append(t2-t1)
            print(f"QSVC train accuracy on quantum computer: {accuracy_score(y_train, qsvc.predict(X_train))}")
            y_pred = qsvc.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"QSVC test accuracy on quantum computer: {accuracy}")
    finally:
        session.close()
        print("Session closed")
    time_vs_size_plot(times, sizes)

def run_single_session_qsvm(quantum_kernel, X, y, size, backend):
    df = pd.concat([X, y], axis=1)
    subsampled_df = stratified_subsample(df, "morphological_type", size, random_state=42)
    y = subsampled_df["morphological_type"]
    X = subsampled_df.drop(columns = ["morphological_type"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    session = Session(backend=backend)
    try:
        qsvc = QSVC(quantum_kernel=quantum_kernel)
        t1 = time.time()
        circuits = quantum_kernel.evaluate(X_train)
        t2 = time.time()
        print(f"Time to evaluate quantum kernel: {t2-t1}")
        qsvc.fit(X_train, y_train)
        total_quantum_execution_time = session.usage()
        print(f"Actual quantum execution time for all jobs: {total_quantum_execution_time} seconds")
        print(f"QSVC train accuracy on quantum computer: {accuracy_score(y_train, qsvc.predict(X_train))}")
        y_pred = qsvc.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"QSVC test accuracy on quantum computer: {accuracy}")
    finally:
        session.close()
        print("Session closed")
    return y_test, y_pred

def local_qsvm(quantum_kernel, X, y, size = 100, optimize_params = False):
    df = pd.concat([X, y], axis=1)
    subsampled_df = stratified_subsample(df=df, 
                                         class_label="morphological_type", 
                                         total_samples=size, 
                                         random_state=42
                                         )
    y = subsampled_df["morphological_type"]
    X = subsampled_df.drop(columns = ["morphological_type"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    if optimize_params:
        # Perform GridSearch for hyperparameter tuning
        param_grid = {"C": [0.01, 1, 10], "gamma": [0.01, 0.1, 1]}
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=qsvc, param_grid=param_grid, cv=cv, scoring="accuracy")
        print("Optimizing QSVC parameters with GridSearch...")
        t1 = time.time()
        grid_search.fit(X_train, y_train)
        t2 = time.time()
        print(f"Parameter optimization completed in {t2 - t1:.2f} seconds.")
        print("Best parameters found:", grid_search.best_params_)
        qsvc = grid_search.best_estimator_
    else:
        # Train QSVC
        print("Training QSVC locally...")
        t1 = time.time()
        qsvc.fit(X_train, y_train)
        t2 = time.time()
        print(f"Total time for QSVC training: {t2 - t1:.2f} seconds.")
    
    # Evaluate on the training set
    y_train_pred = qsvc.predict(X_train)
    train_accuracy = accuracy_score(y_train,y_train_pred)
    print(f"Local QSVC Training Accuracy: {train_accuracy:.4f}")
    
    # Predict on the test set
    y_pred_qsvc = qsvc.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_qsvc)
    print(f"Local QSVC Test Accuracy: {test_accuracy:.4f}")
    
    return y_train, y_train_pred, y_test, y_pred_qsvc
    

def local_qlr_old(X, y, size = 100):
    df = pd.concat([X, y], axis=1)
    subsampled_df = stratified_subsample(df, "morphological_type", size, random_state=42)
    y = subsampled_df["morphological_type"]
    X = subsampled_df.drop(columns = ["morphological_type"])
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Getting optimal params for QLR")
    optimal_params = train_quantum_logistic_regression(X_train, y_train, num_params=X_train.shape[1])
    print("Training QLR locally")
    train_predictions = [sigmoid(quantum_linear_model(optimal_params, x)) for x in X_train]
    y_pred = [1 if train_predictions[i] >= 0.5 else 0 for i in range(len(train_predictions))]
    # Print predictions and accuracy
    print(f"QLR train accuracy: {accuracy_score(y_train, y_pred):.4f}")
    test_predictions = [sigmoid(quantum_linear_model(optimal_params, x)) for x in X_test]
    y_pred = [1 if test_predictions[i] >= 0.5 else 0 for i in range(len(test_predictions))]
    print(f"QLR test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    return y_test, y_pred

def local_qlr( X, y, size = 100):
    df = pd.concat([X, y], axis=1)
    subsampled_df = stratified_subsample(df=df, 
                                         class_label="morphological_type", 
                                         total_samples=size, 
                                         random_state=42
                                         )
    y = subsampled_df["morphological_type"]
    X = subsampled_df.drop(columns = ["morphological_type"])
    y_train, y_train_pred, y_test, y_pred = quantum_logistic_regression(X, y)
    return y_train, y_train_pred, y_test, y_pred


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

def time_vs_size_plot(times, sizes, name = None):
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, times, 'g-', alpha=0.6)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Subsample Size N", fontsize=25)
    plt.ylabel("Time (seconds)", fontsize=20)
    plt.savefig(f"./plots/time_vs_size_{name}.pdf", dpi=1200)
    plt.close()

if __name__ == "__main__":
    quantum_kernel, X, y = load_data()

    y_train, y_train_pred, y_test, y_pred = local_qsvm(quantum_kernel, X, y, size = 200, optimize_params=False)
    y_train_pred = pd.Series(y_train_pred)
    y_pred = pd.Series(y_pred)
    y_train = pd.concat([y_train] * 10, ignore_index=True)
    y_train_pred = pd.concat([y_train_pred] * 10, ignore_index=True)
    confusion_plot(y_train, y_train_pred, cmap = 'Greens', name = "qsvm_local")
    print(classification_report(y_train, y_train_pred))


    y_train, y_train_pred, y_test, y_pred = local_qlr(X, y, size = 200)
    y_train_pred = pd.Series(y_train_pred)
    y_pred = pd.Series(y_pred)
    y_train = pd.concat([y_train] * 10, ignore_index=True)
    y_train_pred = pd.concat([y_train_pred] * 10, ignore_index=True)
    confusion_plot(y_train, y_train_pred, name = "qlr_local")
    print(classification_report(y_train, y_train_pred))

    backend = set_up_backend()
    y_test, y_pred = run_single_session_qsvm(quantum_kernel, X, y, 200, backend)
    confusion_plot(y_test, y_pred, name = "qsvm")
    print(classification_report(y_test, y_pred))
