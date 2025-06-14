# Copyright IBM 2021, 2022. Licensed under Apache License 2.0.

"""The matrix functional of the vector solution to linear systems."""

from typing import Union, List
import numpy as np
from scipy.sparse import diags

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from observables.linear_system_observable import LinearSystemObservable


class MatrixFunctional(LinearSystemObservable):
    """Class for computing the matrix functional of a linear system solution.
    
    Evaluates x^T A x for a tridiagonal Toeplitz symmetric matrix A and solution vector x.
    """

    def __init__(self, main_diag: float, off_diag: float) -> None:
        """Initialize the matrix functional observable.
        
        Args:
            main_diag: Main diagonal value of the tridiagonal Toeplitz matrix.
            off_diag: Off-diagonal value of the tridiagonal Toeplitz matrix.
        """
        self._main_diag = main_diag
        self._off_diag = off_diag

    def observable(self, num_qubits: int) -> List[SparsePauliOp]:
        """Define the observable operators for the matrix functional.
        
        Args:
            num_qubits: Number of qubits for the observable.

        Returns:
            List of SparsePauliOp objects representing the observable components.
        """
        I_op = SparsePauliOp("I" * num_qubits, coeffs=[1.0])
        Z_op = SparsePauliOp("Z" * num_qubits, coeffs=[1.0])
        zero_op = (I_op + Z_op) * 0.5  # (I + Z)/2
        one_op = (I_op - Z_op) * 0.5   # (I - Z)/2
        
        observables = [I_op]  # First element for norm
        for i in range(num_qubits):
            j = num_qubits - i - 1
            prefix = SparsePauliOp("I" * j) if j > 0 else SparsePauliOp("I")
            suffix = SparsePauliOp("I" * i) if i > 0 else SparsePauliOp("I")
            
            zero_term = prefix @ zero_op @ suffix
            one_term = prefix @ one_op @ suffix
            observables.extend([zero_term, one_term])
        
        return observables

    def observable_circuit(self, num_qubits: int) -> List[QuantumCircuit]:
        """Create circuits to measure the matrix functional observable.
        
        Args:
            num_qubits: Number of qubits for the circuits.

        Returns:
            List of QuantumCircuits for measuring the observable components.
        """
        circuits = [QuantumCircuit(num_qubits, name="norm")]  # For norm measurement
        for i in range(num_qubits):
            qc = QuantumCircuit(num_qubits, name=f"bit_{i}")
            for j in range(i):
                qc.cx(i, j)
            qc.h(i)
            circuits.extend([qc.copy(), qc.copy()])  # Two circuits per bit position
        return circuits

    def post_processing(
        self,
        solution: Union[float, List[float]],
        num_qubits: int,
        scaling: float = 1.0
    ) -> float:
        """Evaluate the matrix functional from measurement results.
        
        Args:
            solution: List of probabilities from circuit execution.
            num_qubits: Number of qubits used.
            scaling: Scaling factor for the solution (default: 1.0).

        Returns:
            Real value of the matrix functional x^T A x.

        Raises:
            ValueError: If solution format is invalid or length doesn't match expected.
        """
        if not isinstance(solution, list):
            raise ValueError("Solution must be a list of probabilities.")
        expected_len = 1 + 2 * num_qubits  # 1 for norm + 2 per qubit
        if len(solution) != expected_len:
            raise ValueError(f"Solution length must be {expected_len}, got {len(solution)}.")
        
        main_val = solution[0] / (scaling ** 2)
        off_val = sum((solution[i] - solution[i + 1]) / (scaling ** 2)
                     for i in range(1, len(solution), 2))
        
        return float(np.real(self._main_diag * main_val + self._off_diag * off_val))

    def evaluate_classically(
        self,
        solution: Union[np.ndarray, QuantumCircuit]
    ) -> float:
        """Classically compute the matrix functional x^T A x.
        
        Args:
            solution: Solution vector as numpy array or QuantumCircuit preparing it.

        Returns:
            Real value of the matrix functional.

        Raises:
            ValueError: If solution type is invalid.
        """
        if isinstance(solution, QuantumCircuit):
            solution = Statevector(solution).data
        elif not isinstance(solution, np.ndarray):
            raise ValueError("Solution must be a numpy array or QuantumCircuit.")
        
        matrix = diags(
            [self._off_diag, self._main_diag, self._off_diag],
            [-1, 0, 1],
            shape=(len(solution), len(solution))
        ).toarray()
        
        return float(np.real(np.dot(solution.conj().T, np.dot(matrix, solution))))

# Example usage
# if __name__ == "__main__":
#     observable = MatrixFunctional(1.0, -1/3)
#     vector = np.array([1.0, -2.1, 3.2, -4.3])
#     norm_vector = vector / np.linalg.norm(vector)
#     num_qubits = int(np.log2(len(vector)))
    
#     qc = QuantumCircuit(num_qubits)
#     qc.initialize(norm_vector, range(num_qubits))
#     exact = observable.evaluate_classically(norm_vector)
#     print(f"Classical matrix functional: {exact:.6f}")