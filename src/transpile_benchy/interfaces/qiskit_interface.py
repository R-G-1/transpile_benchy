"""QiskitInterface class.

This module contains the QiskitInterface class, which is a subclass of
SubmoduleInterface. It is intended to be used for submodules that are
written in Qiskit, and have a set of functions which return
QuantumCircuits.
"""
from typing import Callable, List, Type

from qiskit import QuantumCircuit

from transpile_benchy.interfaces.abc_interface import SubmoduleInterface


class QuantumCircuitFactory(SubmoduleInterface):
    """Subclass of SubmoduleInterface for generating quantum circuits.

    This class generates quantum circuits of a given type (e.g., QFT or QuantumVolume)
    for a specified set of qubit counts.

    Example usage:

    num_qubits = [8, 12, 16, 20, 24, 28, 32, 36]

    qiskit_functions_qft = QuantumCircuitFactory(QFT, num_qubits)

    qiskit_functions_qv = QuantumCircuitFactory(QuantumVolume, num_qubits)
    """

    def __init__(self, function_type: Type[Callable], num_qubits: List[int]) -> None:
        """Initialize QuantumCircuitFactory."""
        self.function_type = function_type
        self.num_qubits = num_qubits
        super().__init__()

    def _get_all_circuits(self) -> List[str]:
        """Return a list of all possible circuit names."""
        return [f"{self.function_type.__name__}_{n}" for n in self.num_qubits]

    def _load_circuit(self, circuit_str: str) -> QuantumCircuit:
        """Create a quantum circuit given the circuit name."""
        num_qubits = int(circuit_str.split("_")[-1])
        circuit = self.function_type(num_qubits)
        circuit.name = circuit_str
        return circuit
