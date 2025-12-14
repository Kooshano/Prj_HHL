"""The HHL algorithm."""

from typing import Optional, Union, List, Callable, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.providers import Backend
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit.primitives import StatevectorEstimator
try:
    from qiskit.primitives import BackendEstimator
except ImportError:
    # In Qiskit 2.2+, BackendEstimator doesn't exist, use qiskit_aer.primitives.Estimator instead
    BackendEstimator = None
from QLS.linear_solver import LinearSolver, LinearSolverResult
from QLS.matrices.numpy_matrix import NumPyMatrix
from QLS.observables.linear_system_observable import LinearSystemObservable
from qiskit.circuit.library import StatePreparation

class HHL(LinearSolver):
    r"""Systems of linear equations arise naturally in many real-life applications in a wide range
    of areas, such as in the solution of Partial Differential Equations, the calibration of
    financial models, fluid simulation or numerical field calculation. The problem can be defined
    as, given a matrix :math:`A\in\mathbb{C}^{N\times N}` and a vector
    :math:`\vec{b}\in\mathbb{C}^{N}`, find :math:`\vec{x}\in\mathbb{C}^{N}` satisfying
    :math:`A\vec{x}=\vec{b}`.

    A system of linear equations is called :math:`s`-sparse if :math:`A` has at most :math:`s`
    non-zero entries per row or column. Solving an :math:`s`-sparse system of size :math:`N` with
    a classical computer requires :math:`\mathcal{ O }(Ns\kappa\log(1/\epsilon))` running time
    using the conjugate gradient method. Here :math:`\kappa` denotes the condition number of the
    system and :math:`\epsilon` the accuracy of the approximation.

    The HHL is a quantum algorithm to estimate a function of the solution with running time
    complexity of :math:`\mathcal{ O }(\log(N)s^{2}\kappa^{2}/\epsilon)` when
    :math:`A` is a Hermitian matrix under the assumptions of efficient oracles for loading the
    data, Hamiltonian simulation and computing a function of the solution. This is an exponential
    speed up in the size of the system, however one crucial remark to keep in mind is that the
    classical algorithm returns the full solution, while the HHL can only approximate functions of
    the solution vector.

    Examples:

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from quantum_linear_solvers.linear_solvers.hhl import HHL
            from quantum_linear_solvers.linear_solvers.matrices import TridiagonalToeplitz
            from quantum_linear_solvers.linear_solvers.observables import MatrixFunctional

            matrix = TridiagonalToeplitz(2, 1, 1 / 3, trotter_steps=2)
            right_hand_side = [1.0, -2.1, 3.2, -4.3]
            observable = MatrixFunctional(1, 1 / 2)
            rhs = right_hand_side / np.linalg.norm(right_hand_side)

            # Initial state circuit
            num_qubits = matrix.num_state_qubits
            qc = QuantumCircuit(num_qubits)
            qc.isometry(rhs, list(range(num_qubits)), None)

            hhl = HHL()
            solution = hhl.solve(matrix, qc, observable)
            approx_result = solution.observable

    References:

        [1]: Harrow, A. W., Hassidim, A., Lloyd, S. (2009).
        Quantum algorithm for linear systems of equations.
        `Phys. Rev. Lett. 103, 15 (2009), 1–15. <https://doi.org/10.1103/PhysRevLett.103.150502>`_

        [2]: Carrera Vazquez, A., Hiptmair, R., & Woerner, S. (2020).
        Enhancing the Quantum Linear Systems Algorithm using Richardson Extrapolation.
        `arXiv:2009.04484 <http://arxiv.org/abs/2009.04484>`_

    """

    def __init__(
        self,
        epsilon: float = 1e-2,
        expectation=None,  # Removed ExpectationBase since it’s not defined
        quantum_instance: Optional[Backend] = None,  # Updated to Backend only
    ) -> None:
        r"""
        Args:
            epsilon: Error tolerance of the approximation to the solution, i.e. if :math:`x` is the
                exact solution and :math:`\tilde{x}` the one calculated by the algorithm, then
                :math:`||x - \tilde{x}|| \le epsilon`.
            expectation: Deprecated optional argument previously used for expectation converters.
                Ignored in favor of built-in Estimator functionality.
            quantum_instance: A Qiskit Backend. If None, a Statevector calculation is done using StatevectorEstimator.
        """
        super().__init__()

        self._epsilon = epsilon
        # Tolerance for the different parts of the algorithm as per [1]
        self._epsilon_r = epsilon / 3  # conditioned rotation
        self._epsilon_s = epsilon / 3  # state preparation
        self._epsilon_a = epsilon / 6  # hamiltonian simulation

        self._scaling = None  # scaling of the solution

        # Store the backend for statevector simulation
        self._quantum_instance = quantum_instance
        # Create BackendEstimator if backend is provided, otherwise use StatevectorEstimator
        if quantum_instance is not None and BackendEstimator is not None:
            self._sampler = BackendEstimator(backend=quantum_instance)
        else:
            from qiskit.primitives import StatevectorEstimator
            self._sampler = StatevectorEstimator()  # For statevector simulation without backend

        self._expectation = None  # Set to None as Estimator handles expectations

        # For now the default reciprocal implementation is exact
        self._exact_reciprocal = True
        # Set the default scaling to 1
        self.scaling = 1

    @property
    def quantum_instance(self) -> Optional[Backend]:
        """Get the quantum instance.

        Returns:
            The quantum instance used to run this algorithm.
        """
        # Return the stored quantum instance
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Optional[Backend]
    ) -> None:
        """Set quantum instance.

        Args:
            quantum_instance: A Qiskit Backend used to run this algorithm.
                If None, a Statevector calculation is done using StatevectorEstimator elsewhere.
        """
        self._quantum_instance = quantum_instance
        # Create BackendEstimator if backend is provided, otherwise use StatevectorEstimator
        if quantum_instance is not None and BackendEstimator is not None:
            self._sampler = BackendEstimator(backend=quantum_instance)
        else:
            from qiskit.primitives import StatevectorEstimator
            self._sampler = StatevectorEstimator()  # For statevector simulation without backend
    @property
    def scaling(self) -> float:
        """The scaling of the solution vector."""
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: float) -> None:
        """Set the new scaling of the solution vector."""
        self._scaling = scaling

    @property
    def expectation(self):
        """The expectation value algorithm used to construct the expectation measurement from
        the observable. Deprecated: Expectation values are now handled by the Estimator primitive."""
        return self._expectation

    @expectation.setter
    def expectation(self, expectation) -> None:
        """Set the expectation value algorithm. Deprecated: Use the Estimator primitive instead."""
        self._expectation = expectation
    def _get_delta(self, n_l: int, lambda_min: float, lambda_max: float) -> float:
        """Calculates the scaling factor to represent exactly lambda_min on nl binary digits.

        Args:
            n_l: The number of qubits to represent the eigenvalues.
            lambda_min: the smallest eigenvalue.
            lambda_max: the largest eigenvalue.

        Returns:
            The value of the scaling factor.
        """
        formatstr = "#0" + str(n_l + 2) + "b"
        lambda_min_tilde = np.abs(lambda_min * (2**n_l - 1) / lambda_max)
        # floating point precision can cause problems
        if np.abs(lambda_min_tilde - 1) < 1e-7:
            lambda_min_tilde = 1
        binstr = format(int(lambda_min_tilde), formatstr)[2::]
        lamb_min_rep = 0
        for i, char in enumerate(binstr):
            lamb_min_rep += int(char) / (2 ** (i + 1))
        return lamb_min_rep

        # ------------------------------------------------------------------
    # Helper: make a matrix Hermitian by block‑embedding
    # ------------------------------------------------------------------
    def _embed_to_hermitian(
        self,
        A: np.ndarray,
        b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        If A is already Hermitian, return it unchanged.

        Otherwise build the Hermitian embedding
                H = [[ 0,  A ],
                     [A†,  0 ]]
        and extend the RHS vector with zeroes:

                b' = [b, 0].

        Returns
        -------
        (H, b', was_embedded)
        """
        if np.allclose(A, A.conj().T):
            return A, b, False   # already Hermitian

        # Build the Hermitian block matrix.
        zero = np.zeros_like(A)
        H = np.block([[zero, A], [A.conj().T, zero]])

        # Extend right‑hand side:  [b, 0]
        b_ext = np.concatenate([b, np.zeros_like(b)])

        return H, b_ext, True
    
    def _calculate_norm(self, qc: QuantumCircuit) -> float:
        """Calculates the value of the euclidean norm of the solution.

        Args:
            qc: The quantum circuit preparing the solution x to the system.

        Returns:
            The value of the euclidean norm of the solution.
        """
        # Calculate the number of qubits
        nb = qc.qregs[0].size  # Solution qubits
        nl = qc.qregs[1].size  # Eigenvalue qubits
        na = qc.num_ancillas   # Ancilla qubits

        # Import required classes
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from qiskit.primitives import StatevectorEstimator

        # Create the Operators Zero and One as Pauli operators
        zero_op = SparsePauliOp.from_list([("I", 0.5), ("Z", 0.5)])  # (I + Z) / 2
        one_op = SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)])  # (I - Z) / 2

        # Construct the norm observable: one_op on flag qubit, zero_op on nl+na qubits, I on nb qubits
        zero_tensor = zero_op
        for _ in range(nl + na - 1):
            zero_tensor = zero_tensor.tensor(zero_op)
        observable = one_op.tensor(zero_tensor).tensor(SparsePauliOp("I" * nb))

        # Compute the expectation value
        # Use backend if available and save_statevector is supported, otherwise use Statevector simulation
        if self._quantum_instance is not None and hasattr(self._quantum_instance, 'run'):
            # For AerSimulator (Backend), try to use save_statevector() instruction
            from qiskit import transpile
            try:
                # Try Qiskit 2.x API first (save_statevector with label)
                qc_with_save = qc.copy()
                if hasattr(qc_with_save, 'save_statevector'):
                    # Qiskit 2.x: save_statevector is a method
                    qc_with_save.save_statevector(label='statevector')
                    qc_transpiled = transpile(qc_with_save, backend=self._quantum_instance)
                    # Run on backend
                    result = self._quantum_instance.run(qc_transpiled).result()
                    state = result.get_statevector(qc_with_save)
                else:
                    # save_statevector not available, use Statevector directly
                    state = Statevector.from_instruction(qc)
            except (AttributeError, TypeError, ValueError) as e:
                # If save_statevector doesn't work, fall back to Statevector simulation
                state = Statevector.from_instruction(qc)
        else:
            # Use Statevector simulation directly (when quantum_instance is None or StatevectorEstimator)
            state = Statevector.from_instruction(qc)
        # Compute expectation value manually
        norm_2 = state.expectation_value(observable).real

        return np.real(np.sqrt(norm_2) / self.scaling)

    def _calculate_observable(
        self,
        solution: QuantumCircuit,
        ls_observable: Optional[LinearSystemObservable] = None,
        observable_circuit: Optional[QuantumCircuit] = None,
        post_processing: Optional[Callable[[Union[float, List[float]], int, float], float]] = None,
    ) -> Tuple[float, Union[complex, List[complex]]]:
        """Calculates the value of the observable(s) given.

        Args:
            solution: The quantum circuit preparing the solution x to the system.
            ls_observable: Information to be extracted from the solution.
            observable_circuit: Circuit to be applied to the solution to extract information.
            post_processing: Function to compute the value of the observable.

        Returns:
            The value of the observable(s) and the circuit results before post-processing as a tuple.
        """
        # Get the number of qubits
        nb = solution.qregs[0].size
        nl = solution.qregs[1].size
        na = solution.num_ancillas

        # Import required classes
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from qiskit.primitives import StatevectorEstimator

        # Default observable is identity on solution qubits
        observable = SparsePauliOp("I" * nb)

        # If observable is given, construct post_processing and observable_circuit
        if ls_observable is not None:
            observable_circuit = ls_observable.observable_circuit(nb)
            post_processing = ls_observable.post_processing
            if isinstance(ls_observable, LinearSystemObservable):
                observable = ls_observable.observable(nb)

        # Create the Operators Zero and One
        zero_op = SparsePauliOp.from_list([("I", 0.5), ("Z", 0.5)])  # (I + Z) / 2
        one_op = SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)])  # (I - Z) / 2

        is_list = True
        if not isinstance(observable_circuit, list):
            is_list = False
            observable_circuit = [observable_circuit]
            observable = [observable]

        circuits = []
        observables = []
        for circ, obs in zip(observable_circuit, observable):
            circuit = QuantumCircuit(solution.num_qubits)
            circuit.append(solution, circuit.qubits)
            circuit.append(circ, range(nb))

            # Construct observable: one_op on flag, zero_op on nl+na, obs on nb
            zero_tensor = zero_op
            for _ in range(nl + na - 1):
                zero_tensor = zero_tensor.tensor(zero_op)
            ob = one_op.tensor(zero_tensor).tensor(obs)
            circuits.append(circuit)
            observables.append(ob)

        # Compute expectation values using backend if available, otherwise Statevector simulation
        expectation_results = []
        for circ in circuits:
            if self._quantum_instance is not None and hasattr(self._quantum_instance, 'run'):
                # For AerSimulator (Backend), try to use save_statevector() instruction
                from qiskit import transpile
                try:
                    # Try Qiskit 2.x API first (save_statevector with label)
                    circ_with_save = circ.copy()
                    if hasattr(circ_with_save, 'save_statevector'):
                        # Qiskit 2.x: save_statevector is a method
                        circ_with_save.save_statevector(label='statevector')
                        circ_transpiled = transpile(circ_with_save, backend=self._quantum_instance)
                        # Run on backend
                        result = self._quantum_instance.run(circ_transpiled).result()
                        state = result.get_statevector(circ_with_save)
                    else:
                        # save_statevector not available, use Statevector directly
                        state = Statevector.from_instruction(circ)
                except (AttributeError, TypeError, ValueError) as e:
                    # If save_statevector doesn't work, fall back to Statevector simulation
                    state = Statevector.from_instruction(circ)
            else:
                # Use Statevector simulation directly (when quantum_instance is None or StatevectorEstimator)
                state = Statevector.from_instruction(circ)
            exp_value = state.expectation_value(observables[circuits.index(circ)]).real
            expectation_results.append(exp_value)
        expectation_results = expectation_results if is_list else expectation_results[0]

        # Apply post_processing (default to identity if None)
        if post_processing is None:
            def post_processing(x, _, __): return x
        result = post_processing(expectation_results, nb, self.scaling)

        return result, expectation_results
    def construct_circuit(
        self,
        matrix: Union[List, np.ndarray, QuantumCircuit],
        vector: Union[List, np.ndarray, QuantumCircuit],
        neg_vals: Optional[bool] = True,
    ) -> QuantumCircuit:
        """Construct the HHL circuit.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            neg_vals: States whether the matrix has negative eigenvalues. If False the
                computation becomes cheaper.

        Returns:
            The HHL circuit.

        Raises:
            ValueError: If the input is not in the correct format.
            ValueError: If the type of the input matrix is not supported.
        """
        # State preparation circuit
        if isinstance(matrix, list):
            matrix = np.array(matrix)
        if isinstance(vector, list):
            vector = np.array(vector)

        if isinstance(matrix, np.ndarray) and isinstance(vector, np.ndarray):
            matrix, vector, _ = self._embed_to_hermitian(matrix, vector)   # Embed to Hermitian if needed
        if isinstance(vector, QuantumCircuit):
            nb = vector.num_qubits
            vector_circuit = vector
        elif isinstance(vector, (list, np.ndarray)):
            if isinstance(vector, list):
                vector = np.array(vector)
            nb = int(np.log2(len(vector)))
            vector_circuit = QuantumCircuit(nb)
            # Replace isometry with StatePreparation
            state_prep = StatePreparation(vector / np.linalg.norm(vector))
            vector_circuit.append(state_prep, list(range(nb)))

        # If state preparation is probabilistic the number of qubit flags should increase
        nf = 1

        # Hamiltonian simulation circuit - default is Trotterization
        if isinstance(matrix, QuantumCircuit):
            matrix_circuit = matrix
        elif isinstance(matrix, (list, np.ndarray)):
            if isinstance(matrix, list):
                matrix = np.array(matrix)

            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")
            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Input matrix must be hermitian!")
            if matrix.shape[0] != 2**vector_circuit.num_qubits:
                raise ValueError(
                    "Input vector dimension does not match input "
                    "matrix dimension! Vector dimension: "
                    + str(vector_circuit.num_qubits)
                    + ". Matrix dimension: "
                    + str(matrix.shape[0])
                )
            matrix_circuit = NumPyMatrix(matrix, evolution_time=2 * np.pi)
        else:
            raise ValueError(f"Invalid type for matrix: {type(matrix)}.")

        # Set the tolerance for the matrix approximation
        if hasattr(matrix_circuit, "tolerance"):
            matrix_circuit.tolerance = self._epsilon_a

        # Check if the matrix can calculate the condition number and store the upper bound
        if (
            hasattr(matrix_circuit, "condition_bounds")
            and matrix_circuit.condition_bounds() is not None
        ):
            kappa = matrix_circuit.condition_bounds()[1]
        else:
            kappa = 1
        # Update the number of qubits required to represent the eigenvalues
        nl = max(nb + 1, int(np.ceil(np.log2(kappa + 1)))) + neg_vals

        # Check if the matrix can calculate bounds for the eigenvalues
        if (
            hasattr(matrix_circuit, "eigs_bounds")
            and matrix_circuit.eigs_bounds() is not None
        ):
            lambda_min, lambda_max = matrix_circuit.eigs_bounds()
            delta = self._get_delta(nl - neg_vals, lambda_min, lambda_max)
            matrix_circuit.evolution_time = (
                2 * np.pi * delta / lambda_min / (2**neg_vals)
            )
            self.scaling = lambda_min
        else:
            delta = 1 / (2**nl)
            print("The solution will be calculated up to a scaling factor.")

        if self._exact_reciprocal:
            reciprocal_circuit = ExactReciprocal(nl, delta, neg_vals=neg_vals)
            na = matrix_circuit.num_ancillas
        else:
            num_values = 2**nl
            constant = delta
            a = int(round(num_values ** (2 / 3)))
            r = 2 * constant / a + np.sqrt(np.abs(1 - (2 * constant / a) ** 2))
            degree = min(
                nb,
                int(
                    np.log(
                        1
                        + (
                            16.23
                            * np.sqrt(np.log(r) ** 2 + (np.pi / 2) ** 2)
                            * kappa
                            * (2 * kappa - self._epsilon_r)
                        )
                        / self._epsilon_r
                    )
                ),
            )
            num_intervals = int(np.ceil(np.log((num_values - 1) / a) / np.log(5)))
            breakpoints = []
            for i in range(0, num_intervals):
                breakpoints.append(a * (5**i))
                if i == num_intervals - 1:
                    breakpoints.append(num_values - 1)
            reciprocal_circuit = PiecewiseChebyshev(
                lambda x: np.arcsin(constant / x), degree, breakpoints, nl
            )
            na = max(matrix_circuit.num_ancillas, reciprocal_circuit.num_ancillas)

        # Initialise the quantum registers
        qb = QuantumRegister(nb)  # right hand side and solution
        ql = QuantumRegister(nl)  # eigenvalue evaluation qubits
        if na > 0:
            qa = AncillaRegister(na)  # ancilla qubits
        qf = QuantumRegister(nf)  # flag qubits

        if na > 0:
            qc = QuantumCircuit(qb, ql, qa, qf)
        else:
            qc = QuantumCircuit(qb, ql, qf)

        # State preparation
        qc.append(vector_circuit, qb[:])
        # QPE
        phase_estimation = PhaseEstimation(nl, matrix_circuit)
        if na > 0:
            qc.append(
                phase_estimation, ql[:] + qb[:] + qa[: matrix_circuit.num_ancillas]
            )
        else:
            qc.append(phase_estimation, ql[:] + qb[:])
        # Conditioned rotation
        if self._exact_reciprocal:
            qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])
        else:
            qc.append(
                reciprocal_circuit.to_instruction(),
                ql[:] + [qf[0]] + qa[: reciprocal_circuit.num_ancillas],
            )
        # QPE inverse
        if na > 0:
            qc.append(
                phase_estimation.inverse(),
                ql[:] + qb[:] + qa[: matrix_circuit.num_ancillas],
            )
        else:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:])
        return qc

    def solve(
        self,
        matrix: Union[List, np.ndarray, QuantumCircuit],
        vector: Union[List, np.ndarray, QuantumCircuit],
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
        """Tries to solve the given linear system of equations.

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.
            observable: Optional information to be extracted from the solution.
                Default is the probability of success of the algorithm.
            observable_circuit: Optional circuit to be applied to the solution to extract
                information. Default is `None`.
            post_processing: Optional function to compute the value of the observable.
                Default is the raw value of measuring the observable.

        Raises:
            ValueError: If an invalid combination of observable, observable_circuit and
                post_processing is passed.

        Returns:
            The result object containing information about the solution vector of the linear
            system.
        """
        # verify input
        if observable is not None:
            if observable_circuit is not None or post_processing is not None:
                raise ValueError(
                    "If observable is passed, observable_circuit and post_processing cannot be set."
                )

        solution = LinearSolverResult()
        solution.state = self.construct_circuit(matrix, vector)
        solution.euclidean_norm = self._calculate_norm(solution.state)

        if isinstance(observable, List):
            observable_all, circuit_results_all = [], []
            for obs in observable:
                obs_i, circ_results_i = self._calculate_observable(
                    solution.state, obs, observable_circuit, post_processing
                )
                observable_all.append(obs_i)
                circuit_results_all.append(circ_results_i)
            solution.observable = observable_all
            solution.circuit_results = circuit_results_all
        elif observable is not None or observable_circuit is not None:
            solution.observable, solution.circuit_results = self._calculate_observable(
                solution.state, observable, observable_circuit, post_processing
            )

        return solution