"""Non-exhaustive list of circuits defined by Qiskit.

Each function returns a QuantumCircuit object, only paramerized by the
number of qubits.
"""
import networkx as nx
import numpy as np
import openfermion as of
from qiskit import QuantumCircuit
from qiskit.algorithms import AmplificationProblem, Grover, Shor
from qiskit.circuit.library import (
    EfficientSU2,
    HiddenLinearFunction,
    QAOAAnsatz,
    QuantumVolume,
)
from qiskit.circuit.library.arithmetic.adders.cdkm_ripple_carry_adder import (
    CDKMRippleCarryAdder,
)
from qiskit.circuit.library.arithmetic.multipliers import RGQFTMultiplier
from qiskit.circuit.library.basis_change import QFT
from supermarq.benchmarks.hamiltonian_simulation import HamiltonianSimulation
from supermarq.converters import cirq_to_qiskit

depth = 2  # arbitary idk what to set this to


# VQE
def vqe_linear(q):
    """Return a VQE circuit with linear entanglement."""
    # set np random seed
    # np.random.seed(42)
    # apply the ansatz depth times
    vqe_circuit_linear = EfficientSU2(
        num_qubits=q, entanglement="linear", reps=depth * 2
    )
    for param in vqe_circuit_linear.parameters:
        vqe_circuit_linear.assign_parameters({param: np.random.rand()}, inplace=1)
    return vqe_circuit_linear


def vqe_full(q):
    """Return a VQE circuit with full entanglement."""
    # set np random seed
    # np.random.seed(42)
    vqe_circuit_full = EfficientSU2(num_qubits=q, entanglement="full")
    for param in vqe_circuit_full.parameters:
        vqe_circuit_full.assign_parameters({param: np.random.rand()}, inplace=1)
    return vqe_circuit_full


# Quantum Volume
def qv(q):
    """Return a Quantum Volume circuit."""
    return QuantumVolume(num_qubits=q, depth=q)


# QFT
def qft(q):
    """Return a QFT circuit."""
    return QFT(q)


# QAOA
def qaoa(q):
    """Return a QAOA circuit."""
    # set np random seed
    # np.random.seed(42)
    qc_mix = QuantumCircuit(q)
    for i in range(0, q):
        qc_mix.rx(np.random.rand(), i)
    # create a random Graph
    G = nx.gnp_random_graph(q, 0.5)  # , seed=42)
    qc_p = QuantumCircuit(q)
    for pair in list(G.edges()):  # pairs of nodes
        qc_p.rzz(2 * np.random.rand(), pair[0], pair[1])
        qc_p.barrier()
    qaoa_qc = QAOAAnsatz(
        cost_operator=qc_p, reps=depth, initial_state=None, mixer_operator=qc_mix
    )
    return qaoa_qc


# Adder
def adder(q):
    """Return a ripple carry adder circuit."""
    if q % 2 != 0:
        raise ValueError("q must be even")
    add_qc = QuantumCircuit(q).compose(
        CDKMRippleCarryAdder(num_state_qubits=int((q - 1) / 2)), inplace=False
    )
    return add_qc


# Multiplier
def mul(q):
    """Return a rgqft multiplier circuit."""
    if q % 4 != 0:
        raise ValueError("q must be divisible by 4")
    mul_qc = QuantumCircuit(q).compose(
        RGQFTMultiplier(num_state_qubits=int(q / 4)), inplace=False
    )
    return mul_qc


# GHZ
def ghz(q):
    """Return a GHZ circuit."""
    ghz_qc = QuantumCircuit(q)
    ghz_qc.h(0)
    for i in range(1, q):
        ghz_qc.cx(0, i)
    return ghz_qc


# Hidden Linear Function
def hlf(q):
    """Return a Hidden Linear Function circuit."""
    # set np random seed
    # np.random.seed(42)
    # create a random symmetric adjacency matrix
    adj_m = np.random.randint(2, size=(q, q))
    adj_m = adj_m + adj_m.T
    adj_m = np.where(adj_m == 2, 1, adj_m)
    hlf_qc = HiddenLinearFunction(adjacency_matrix=adj_m)
    return hlf_qc


# Grover
def grover(q):
    """Return a Grover circuit."""
    q = int(q / 2)  # Grover's take so long because of the MCMT, do a smaller circuit
    # set numpy seed
    np.random.seed(42)
    # integer iteration
    oracle = QuantumCircuit(q)
    # mark a random state
    oracle.cz(0, np.random.randint(1, q))
    problem = AmplificationProblem(oracle)
    g = Grover(
        iterations=int(depth / 2)
    )  # takes too long to find SWAPs if too many iters
    grover_qc = g.construct_circuit(problem)
    return grover_qc


# Shor
def shor(q):
    """Return a shor circuit."""
    s = Shor(QuantumCircuit(q))
    factor = np.random.randint(4, 5000) * 2 + 1  # rand odd number
    return s.construct_circuit(factor)


# Hubbard
def hub(q):
    """Return a Fermi-Hubbard circuit.

    function from OpenFermion
    https://github.com/quantumlib/OpenFermion
    """
    # define the Hamiltonian
    # Parameters.
    nsites = int(q / 2)
    U = 2.0
    J = -1.0

    hubbard = of.fermi_hubbard(1, nsites, tunneling=-J, coulomb=U, periodic=False)
    qasm = of.trotterize_exp_qubop_to_qasm(hamiltonian=hubbard)
    return QuantumCircuit.from_qasm_file(qasm)


def tfim(q):
    """Return a Transverse Ising Models (TFIM) circuit.

    from supermarq https://github.com/Infleqtion/client-superstaq
    """
    return cirq_to_qiskit(HamiltonianSimulation(q, 1 / depth, 0.5).circuit())


# List of all available circuits
available_circuits = [
    vqe_full,
    vqe_linear,
    qv,
    qft,
    qaoa,
    adder,
    mul,
    ghz,
    hlf,
    grover,
    hub,
    tfim,
    shor,
]
