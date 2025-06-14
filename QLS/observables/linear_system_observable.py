# Copyright IBM 2021, 2022. Licensed under Apache License 2.0.

"""An abstract class for linear system observables in Qiskit."""

from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp

class LinearSystemObservable(ABC):
    """Abstract base class for linear system observables in Qiskit."""

    @abstractmethod
    def observable(self, num_qubits: int) -> Union[SparsePauliOp, List[SparsePauliOp]]:
        """Define the observable operator.

        Args:
            num_qubits: Number of qubits for the observable.

        Returns:
            The observable as a SparsePauliOp or list of SparsePauliOps.
        """
        raise NotImplementedError

    @abstractmethod
    def observable_circuit(
        self, num_qubits: int
    ) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """Create the circuit implementing the observable.

        Args:
            num_qubits: Number of qubits for the observable.

        Returns:
            QuantumCircuit or list of QuantumCircuits implementing the observable.
        """
        raise NotImplementedError

    @abstractmethod
    def post_processing(
        self, 
        solution: Union[float, List[float]], 
        num_qubits: int, 
        scaling: float = 1.0
    ) -> float:
        """Evaluate the observable on the linear system solution.

        Args:
            solution: Probability or list of probabilities from circuit execution.
            num_qubits: Number of qubits the observable was applied to.
            scaling: Scaling factor for the solution (default: 1.0).

        Returns:
            The evaluated observable value.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_classically(
        self, 
        solution: Union[np.ndarray, QuantumCircuit]
    ) -> float:
        """Calculate the observable's analytical value from the solution.

        Args:
            solution: Solution vector as numpy array or circuit preparing it.

        Returns:
            The analytical value of the observable.
        """
        raise NotImplementedError