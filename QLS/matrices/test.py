from qiskit import QuantumCircuit
from tridiagonal_toeplitz import TridiagonalToeplitz


matrix = TridiagonalToeplitz(2, 1.0, -0.5)
circ = matrix.build()
print(circ)