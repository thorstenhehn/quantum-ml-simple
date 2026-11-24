# Overview

A set of minimal and beginner-friendly implementations
of quantum machine learning algorithms based on qiskit.
We use statevectors and require no backend access to a real QPU or a simulator.

- Quantum algorithm to pre-calculate an SVM kernel (Support Vector Machine) 

# Installation

Python and qiskit are required to run the scripts.  Follow IBM's
installation guide on
https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit.

# Algorithms and Usage

## Quantum algorithm to pre-calculate an SVM kernel (Support Vector Machine)

This is our first algorithm in the field of Quantum Machine Learning.
We calcalate the kernel matrix for a Support Vector Machine using a
quantum algorithm. To this end, we first encode the features on qubits
(using dense angle encoding, entanglement, and re-encoding). Our
circuit is designed such that the measured values represent the
entries of the kernel matrix we are interested in (up to a square).

Subsequently we use SVM tools from sklearn to show that the
pre-calculated matrix is suitable for the classification task.

Note: this script requires tqdm (for displaying a progress bar) and sklearn.