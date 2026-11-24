# SVM kernel matrix pre-calculation for use with qiskit.
# We use state vectors and do not require access to a quantum
# simulator or backend-computer.
# Our goal is to make this as simple as possible such that users
# can have an easy first interaction with quantum algorithms.
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm


def main():
    print("""
    We use quantum computing to calculate the kernel matrix for a support
    vector machine (SVM). The classification task uses classical tools and
    the pre-calculated matrix.
    """)

    # Generate training and test set, all with equal parameters

    par_n = 0.05
    par_f = 0.7
    par_s = 41

    X_train, y_train = make_circles(200, par_n, par_f, par_s)

    X_test, y_test = make_circles(100, par_n, par_f, par_s + 10)

    K_train = calculate_kernel_matrix(X_train)
    classifier = SVC(kernel="precomputed")
    classifier.fit(K_train, y_train)

    K_test = calculate_cross_kernel_matrix(X_test, X_train)
    pred_test = classifier.predict(K_test)
    print(
        "Accuracy of the classifier on unkown test set:",
        100.0 * accuracy_score(y_test, pred_test),
        "percent",
    )


def calculate_kernel_matrix(X):

    n = len(X)
    K = np.zeros((n, n))

    for i in tqdm(range(n), desc="Kernel matrix (quantum circuits)"):
        for j in range(i, n):
            qc = kernel_circuit(X[i], X[j])

            state = Statevector.from_instruction(qc)

            K[i, j] = state.probabilities()[0]
            K[j, i] = K[i, j]

    return K


def calculate_cross_kernel_matrix(X1, X2):

    K = np.zeros((len(X1), len(X2)))

    for i in tqdm(range(len(X1)), desc="Cross kernel (quantum circuits)"):
        for j in range(len(X2)):
            qc = kernel_circuit(X1[i], X2[j])
            state = Statevector.from_instruction(qc)
            K[i, j] = state.probabilities()[0]
    return K


def kernel_circuit(x_i, x_j):

    qc = encode(x_i)
    qc.compose(encode(x_j).inverse(), inplace=True)

    return qc


def encode(x):

    qc = QuantumCircuit(2)

    # Dense encoding of two features to one qubit
    qc.ry(x[0], 0)
    qc.rz(x[1], 0)

    # To achieve better classification results,
    # encode to a second qubit and entangle

    qc.ry(x[1], 1)
    qc.rz(x[0], 1)

    # Entanglement
    qc.cx(0, 1)

    return qc


if __name__ == "__main__":
    main()
