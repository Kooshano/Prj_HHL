"""The Numpy LinearSolver algorithm (classical)."""

from typing import List, Union, Optional, Callable
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from QLS.linear_solver import LinearSolverResult, LinearSolver
from QLS.observables.linear_system_observable import LinearSystemObservable

# pylint: disable=too-few-public-methods


class NumPyLinearSolver(LinearSolver):
    """The Numpy Linear Solver algorithm (classical).

    This linear system solver computes the exact value of the given observable(s) or the full
    solution vector if no observable is specified.

    Examples:

        .. jupyter-execute::

            import numpy as np
            from quantum_linear_solvers.linear_solvers import NumPyLinearSolver
            from quantum_linear_solvers.linear_solvers.matrices import TridiagonalToeplitz
            from quantum_linear_solvers.linear_solvers.observables import MatrixFunctional

            matrix = TridiagonalToeplitz(2, 1, 1 / 3, trotter_steps=2)
            right_hand_side = [1.0, -2.1, 3.2, -4.3]
            observable = MatrixFunctional(1, 1 / 2)
            rhs = right_hand_side / np.linalg.norm(right_hand_side)

            np_solver = NumPyLinearSolver()
            solution = np_solver.solve(matrix, rhs, observable)
            result = solution.observable
    """

    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit],
        vector: Union[np.ndarray, QuantumCircuit],
        observable: Optional[
            Union[
                LinearSystemObservable,
                List[LinearSystemObservable],
            ]
        ] = None,
        observable_circuit: Optional[
            Union[QuantumCircuit, List[QuantumCircuit]]
        ] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]], int, float], float]
        ] = None,
    ) -> LinearSolverResult:
        """Solve classically the linear system and compute the observable(s)

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            observable: Optional information to be extracted from the solution.
                Default is the probability of success of the algorithm.
            observable_circuit: Optional circuit to be applied to the solution to extract
                information. Default is ``None``.
            post_processing: Optional function to compute the value of the observable.
                Default is the raw value of measuring the observable.

        Returns:
            The result of the linear system.
        """
        # Check if either matrix or vector are QuantumCircuits and get the array from them
        if isinstance(vector, QuantumCircuit):
            vector = Statevector(vector).data
        if isinstance(matrix, QuantumCircuit):
            if hasattr(matrix, "matrix"):
                matrix = matrix.matrix
            else:
                matrix = Operator(matrix).data

        solution_vector = np.linalg.solve(matrix, vector)
        solution = LinearSolverResult()
        solution.state = solution_vector
        if observable is not None:
            if isinstance(observable, list):
                solution.observable = []
                for obs in observable:
                    solution.observable.append(
                        obs.evaluate_classically(solution_vector)
                    )
            else:
                solution.observable = observable.evaluate_classically(solution_vector)
        solution.euclidean_norm = float(np.linalg.norm(solution_vector))
        return solution
