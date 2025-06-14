"""Abstract classes for linear system solvers and observables."""

from abc import ABC, abstractmethod
from typing import Union, Optional, List, Callable
import numpy as np

from qiskit import QuantumCircuit
from qiskit_algorithms import AlgorithmResult
from qiskit.quantum_info import SparsePauliOp


class LinearSystemObservable(ABC):
    """Abstract base class for linear system observables in Qiskit."""

    @abstractmethod
    def observable(self, num_qubits: int) -> Union[SparsePauliOp, List[SparsePauliOp]]:
        """Define the observable operator.

        Args:
            num_qubits: Number of qubits for the observable.

        Returns:
            Observable as a SparsePauliOp or list of SparsePauliOps.
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
            Evaluated observable value.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_classically(
        self,
        solution: Union[np.ndarray, QuantumCircuit]
    ) -> float:
        """Calculate the observable's analytical value from the solution.

        Args:
            solution: Solution as a numpy array or QuantumCircuit preparing it.

        Returns:
            Analytical observable value.
        """
        raise NotImplementedError


class LinearSolverResult(AlgorithmResult):
    """Base class for linear system solver results."""

    def __init__(self) -> None:
        """Initialize the result object with default None values."""
        super().__init__()
        self._state: Optional[Union[QuantumCircuit, np.ndarray]] = None
        self._observable: Optional[Union[float, List[float]]] = None
        self._euclidean_norm: Optional[float] = None
        self._circuit_results: Optional[
            Union[complex, List[complex], List[Union[complex, List[complex]]]]
        ] = None

    @property
    def observable(self) -> Optional[Union[float, List[float]]]:
        """Get the calculated observable(s)."""
        return self._observable

    @observable.setter
    def observable(self, value: Union[float, List[float]]) -> None:
        """Set the observable value(s)."""
        self._observable = value

    @property
    def state(self) -> Optional[Union[QuantumCircuit, np.ndarray]]:
        """Get the solution state (circuit or vector)."""
        return self._state

    @state.setter
    def state(self, value: Union[QuantumCircuit, np.ndarray]) -> None:
        """Set the solution state."""
        self._state = value

    @property
    def euclidean_norm(self) -> Optional[float]:
        """Get the Euclidean norm of the solution."""
        return self._euclidean_norm

    @euclidean_norm.setter
    def euclidean_norm(self, value: float) -> None:
        """Set the Euclidean norm."""
        self._euclidean_norm = value

    @property
    def circuit_results(
        self
    ) -> Optional[Union[complex, List[complex], List[Union[complex, List[complex]]]]]:
        """Get the raw circuit execution results."""
        return self._circuit_results

    @circuit_results.setter
    def circuit_results(
        self,
        value: Union[complex, List[complex], List[Union[complex, List[complex]]]]
    ) -> None:
        """Set the circuit execution results."""
        self._circuit_results = value


class LinearSolver(ABC):
    """Abstract base class for linear system solvers in Qiskit."""

    @abstractmethod
    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit],
        vector: Union[np.ndarray, QuantumCircuit],
        observable: Optional[Union[LinearSystemObservable, List[LinearSystemObservable]]] = None,
        observable_circuit: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]], int, float], float]
        ] = None
    ) -> LinearSolverResult:
        """Solve the linear system and compute observable(s).

        Args:
            matrix: Matrix A in Ax = b (numpy array or circuit preparing it).
            vector: Vector b in Ax = b (numpy array or circuit preparing it).
            observable: Optional observable(s) to evaluate on the solution.
            observable_circuit: Optional circuit(s) to extract information from the solution.
            post_processing: Optional function to process observable results.

        Returns:
            LinearSolverResult containing the solution and computed values.
        """
        raise NotImplementedError