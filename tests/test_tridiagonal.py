"""Test script for TridiagonalToeplitz matrix implementation."""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from QLS.matrices.tridiagonal_toeplitz import TridiagonalToeplitz

def print_matrix_info(matrix_obj):
    """Print information about the matrix."""
    print("\nMatrix Properties:")
    print("-" * 50)
    print("Matrix shape:", matrix_obj.matrix.shape)
    print("\nActual Matrix:")
    print(matrix_obj.matrix)
    
    # Get eigenvalue bounds
    lambda_min, lambda_max = matrix_obj.eigs_bounds()
    print("\nEigenvalue bounds:")
    print(f"Min eigenvalue (absolute): {lambda_min:.4f}")
    print(f"Max eigenvalue (absolute): {lambda_max:.4f}")
    
    # Get condition number bounds
    kappa_min, kappa_max = matrix_obj.condition_bounds()
    print(f"\nCondition number: {kappa_min:.4f}")

def main():
    # Initialize parameters
    num_qubits = 2  # This will create a 4x4 matrix
    main_diagonal = 2.0
    off_diagonal = 1.0
    evolution_time = 1.0
    
    # Create the matrix object
    tridi_matrix = TridiagonalToeplitz(
        num_state_qubits=num_qubits,
        main_diag=main_diagonal,
        off_diag=off_diagonal,
        evolution_time=evolution_time
    )
    
    # Print matrix information
    print_matrix_info(tridi_matrix)
    
    # Create quantum circuit for the matrix
    qc = tridi_matrix.power(1)  # Get circuit for single power
    print("\nQuantum Circuit Details:")
    print("-" * 50)
    print(f"Number of qubits: {qc.num_qubits}")
    print(f"Circuit depth: {qc.depth()}")
    print("\nCircuit:")
    print(qc)
    
    # Create a simple state preparation
    qc_test = QuantumCircuit(num_qubits)
    qc_test.h(0)  # Put first qubit in superposition
    qc_test.compose(qc, inplace=True)  # Add our matrix evolution
    qc_test.measure_all()  # Add measurements
    
    # Simulate the circuit using AerSimulator
    simulator = AerSimulator()
    compiled_circuit = transpile(qc_test, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print("\nMeasurement Results:")
    print("-" * 50)
    print(counts)

if __name__ == "__main__":
    main() 