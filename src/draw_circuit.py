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
from quantum_models import load_data
from stats_analysis import stratified_subsample


# Quantum linear regressor
def quantum_linear_model(params, x, save_circuit=False):
    """
    Creates a quantum circuit that models a linear function.
    :param params: Parameters for the quantum circuit.
    :param x: Input data.
    :param save_circuit: Whether to save the circuit visualization.
    :return: Quantum circuit output as an approximation of linear function.
    """
    num_qubits = len(x)
    qc = QuantumCircuit(num_qubits)
    
    # Encode the data into the quantum circuit (feature encoding)
    for i in range(num_qubits):
        qc.rx(x[i], i)
    
    # Apply parameterized rotations
    idx = 0
    for i in range(num_qubits):
        qc.ry(params[idx], i)
        idx += 1
    
    # Apply entanglement
    for i in range(num_qubits - 1):
        qc.cz(i, i + 1)
    
    qc.measure_all()
    
    # Draw and save the circuit
    if save_circuit:
        fig, ax = plt.subplots(figsize=(12, 6))
        qc.draw(output='mpl', ax=ax)
        plt.savefig("./plots/quantum_circuit_with_sigmoid.pdf", bbox_inches='tight')
        plt.close(fig)
    
    # Simulate the circuit
    backend = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, backend)
    result = backend.run(qc).result()
    statevector = result.get_statevector(qc)
    
    # Return the real part of the expected output as the linear model's prediction
    return np.real(statevector[0])

# Sigmoid function
def sigmoid(z):
    return expit(z)

# Loss function (Cross-Entropy Loss)
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
    loss = log_loss(y, predictions)
    return loss

# Training quantum logistic regression using gradient descent
def train_quantum_logistic_regression(X, y, num_params):
    """
    Train the Quantum Logistic Regression model using gradient-based optimization.
    :param X: Input features.
    :param y: Labels.
    :param num_params: Number of parameters for the quantum model.
    :return: Optimized parameters.
    """
    initial_params = np.random.randn(num_params)
    result = minimize(logistic_loss, initial_params, args=(X, y), method='BFGS', options={'maxiter': 200})
    return result.x  # optimized parameters

# Example usage
if __name__ == "__main__":
    # Example data
    X = np.random.rand(10, 6)  # 10 samples, 2 features
    y = np.random.randint(0, 2, size=10)  # Binary labels
    
    # Train the model and save the circuit visualization
    num_params = X.shape[1]
    params = train_quantum_logistic_regression(X, y, num_params)
    quantum_linear_model(params, X[0], save_circuit=True)