import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
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
from scipy.special import expit 
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import time

# quantum linear regressor
def quantum_linear_model(params, x):
    """
    Creates a quantum circuit that models a linear function.
    :param params: Parameters for the quantum circuit.
    :param x: Input data.
    :return: Quantum circuit output as an approximation of linear function.
    """
    # quantum circuit
    num_qubits = len(x)
    qc = QuantumCircuit(num_qubits)
    # encode the data into the quantum circuit (feature encoding)
    for i in range(num_qubits):
        qc.rx(x[i], i)
    # apply parameterized rotations
    idx = 0
    for i in range(num_qubits):
        qc.ry(params[idx], i)
        idx += 1
    # apply entanglement
    for i in range(num_qubits - 1):
        qc.cz(i, i + 1)
    # sim
    backend = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, backend)
    result = backend.run(qc).result()
    statevector = result.get_statevector(qc)
    # return the real part of the expected output as the linear model's prediction
    return np.real(statevector[0])

def sigmoid(z):
    return expit(z)

def logistic_loss(params, X, y):
    """
    Computes the logistic loss (cross-entropy) for quantum logistic regression.
    :param params: Parameters for the quantum circuit.
    :param X: Input data.
    :param y: Labels.
    :return: Cross-entropy loss.
    """
    predictions = []
    for x in X:
        z = quantum_linear_model(params, x)
        predictions.append(sigmoid(z))
    # calculate logistic loss (cross-entropy)
    loss = log_loss(y, predictions)
    return loss

# training quantum logistic regression using grad descent
def train_quantum_logistic_regression(X, y, num_params):
    """
    Train the Quantum Logistic Regression model using gradient-based optimization.
    :param X: Input features.
    :param y: Labels.
    :param num_params: Number of parameters for the quantum model.
    :return: Optimized parameters.
    """
    # initialize random parameters for the quantum circuit
    initial_params = np.random.randn(num_params)
    # optimize the parameters to minimize the logistic loss function
    result = minimize(logistic_loss, initial_params, args=(X, y), method='BFGS', options={'maxiter': 200})
    return result.x  # optimized parameters

# Additional model: train QLR with quantum kernel
def quantum_logistic_regression(X, y, test_size=0.2):
    # doing the daata preprocessing here
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    num_qubits = X_train.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    print("Creating train and test quantum kernels")
    # compute quantum kernel matrix
    kernel_train = quantum_kernel.evaluate(X_train)
    kernel_test = quantum_kernel.evaluate(X_test, X_train)
    print("Training Quantum Logistic Regression")
    # train classical logistic regression on the quantum kernel
    logistic_model = LogisticRegression(class_weight='balanced')
    logistic_model.fit(kernel_train, y_train)
    # train accuracy
    y_train_pred = logistic_model.predict(kernel_train)
    print("Quantum Logistic Regression Train Accuracy:", accuracy_score(y_train, y_train_pred))
    # predict and evaluate
    y_pred = logistic_model.predict(kernel_test)
    print("Quantum Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    
    return y_train, y_train_pred, y_test, y_pred


