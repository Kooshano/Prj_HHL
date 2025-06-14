from typing import Union, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

class AbsoluteAverage:
    """Observable for the absolute average of a linear system solution.
    
    Computes |1/N * Σx_i| for a vector x = (x_1, ..., x_N), where N = 2^num_qubits.
    """

    def observable(self, num_qubits: int) -> SparsePauliOp:
        """Creates the observable operator as a SparsePauliOp.
        
        Args:
            num_qubits: Number of qubits for the observable.

        Returns:
            SparsePauliOp representing (I + Z)/2 tensored across all qubits, scaled by 1/2^n.
        """
        pauli_string = "Z" * num_qubits
        coeff = 1.0 / (2 ** num_qubits)
        return SparsePauliOp(pauli_string, coeffs=[coeff])

    def observable_circuit(self, num_qubits: int) -> QuantumCircuit:
        """Constructs the circuit for measuring the absolute average observable.
        
        Args:
            num_qubits: Number of qubits for the circuit.

        Returns:
            QuantumCircuit with Hadamard gates on all qubits.
        """
        qc = QuantumCircuit(num_qubits, name="abs_avg_observable")
        qc.h(range(num_qubits))
        return qc

    def post_processing(
        self,
        solution: Union[float, List[float]],
        num_qubits: int,
        scaling: float = 1.0
    ) -> float:
        """Computes the absolute average from measurement results.
        
        Args:
            solution: Probability or list containing a single probability from circuit execution.
            num_qubits: Number of qubits used.
            scaling: Scaling factor for the solution (default: 1.0).

        Returns:
            Absolute average value.

        Raises:
            ValueError: If solution is not a single float or a list with one float.
        """
        if isinstance(solution, list):
            if len(solution) != 1:
                raise ValueError("Solution must be a single value or a list with one value")
            solution = solution[0]
        elif not isinstance(solution, (int, float)):
            raise ValueError("Solution must be a numeric value")

        if solution < 0:
            raise ValueError("Solution probability cannot be negative")

        return float(np.abs(np.sqrt(solution / (2 ** num_qubits)) / scaling))

    def evaluate_classically(
        self,
        solution: Union[np.ndarray, QuantumCircuit]
    ) -> float:
        """Classically evaluates the absolute average.
        
        Args:
            solution: Solution as numpy array or QuantumCircuit preparing the state.

        Returns:
            Absolute average value.

        Raises:
            ValueError: If solution is neither a numpy array nor a QuantumCircuit.
        """
        if isinstance(solution, QuantumCircuit):
            solution = Statevector(solution).data
        elif not isinstance(solution, np.ndarray):
            raise ValueError("Solution must be a numpy array or QuantumCircuit")
        
        return float(np.abs(np.mean(solution)))

# Example usage
if __name__ == "__main__":
    # Initialize observable
    observable = AbsoluteAverage()
    
    # Test vector
    vector = np.array([1.0, -2.1, 3.2, -4.3])
    norm_vector = vector / np.linalg.norm(vector)
    num_qubits = int(np.log2(len(vector)))
    
    # Create state preparation circuit
    qc = QuantumCircuit(num_qubits)
    qc.initialize(norm_vector, range(num_qubits))
    qc.compose(observable.observable_circuit(num_qubits), inplace=True)
    
    # Get classical result for comparison
    exact = observable.evaluate_classically(norm_vector)
    print(f"Classical absolute average: {exact:.6f}")