"""Correctness check for caching SparsePauliOp across save_expectation_value
calls (FUTURE_WORK.md #7.1's hot-path optimization), applied to the
NH3/N2 P=2 slowdown mitigation in src/api/interface.py.

Why this test exists: FUTURE_WORK.md #7.1 explicitly flagged this as
untested risk -- "unclear whether SparsePauliOp is safe to reuse across
multiple save_expectation_value circuit appends (Aer may mutate the op
internally)." Getting this wrong would silently produce WRONG energies,
not a crash, so this has to be verified directly against the real
Aer build (0.17.2, CUDA-enabled) before the caching fix in
interface.py's _evaluate_distributed_statevector / _expectation_on_gpu
can be trusted. Same verification pattern as the original GPU-native
expectation fix's own correctness check (docs/GPU_EXPECTATION_FIX.md,
"Correctness verification" section): build a small circuit + Hamiltonian,
compute the expectation value several times with a FRESH SparsePauliOp
each time vs. ONE REUSED SparsePauliOp across all of them, and diff to
machine epsilon.

Requires GPU (AerSimulator device='GPU') -- run via:
    make test-sparse-pauli-reuse
or directly inside the container:
    docker run --rm --gpus all vqe-mpi-gpu python3 tests/test_sparse_pauli_op_reuse.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


def _expectation(sim, bound_circuit, op, label):
    circuit = bound_circuit.copy()
    circuit.save_expectation_value(op, list(range(circuit.num_qubits)), label=label)
    result = sim.run(circuit).result()
    return float(result.data(0)[label].real)


@pytest.mark.gpu
def test_fresh_vs_reused_sparse_pauli_op():
    """Representative 4-qubit ansatz + a fake H2-shaped Hamiltonian, same
    structure as GPU_EXPECTATION_FIX.md's own local correctness check.
    Evaluates the SAME bound circuit's expectation value 5 times: once
    with a fresh SparsePauliOp built from scratch each call (the OLD,
    pre-caching behavior), once with a single SparsePauliOp built once
    and reused across all 5 calls (the NEW, cached behavior). If reuse is
    unsafe, Aer's internal handling of the op across repeated
    save_expectation_value appends would show up here as a numeric
    divergence -- not necessarily on the first call, so this repeats the
    call multiple times rather than checking only once.
    """
    sim = AerSimulator(method="statevector", device="GPU")

    # .decompose() matches production: src/api/problems.py:build_ansatz()
    # always decomposes EfficientSU2 before use -- an un-decomposed
    # "blueprint circuit" is rejected by Aer's compiler with "unknown
    # instruction: EfficientSU2" (caught by this test's own first run).
    ansatz = EfficientSU2(4, reps=1, entanglement="full").decompose()
    theta = {p: 0.137 * i for i, p in enumerate(sorted(ansatz.parameters, key=lambda x: x.name))}
    bound = ansatz.assign_parameters(theta)

    pauli_terms = [
        ("IIII", -0.81054),
        ("IIIZ", 0.17218),
        ("IIZI", -0.22575),
        ("IZII", 0.17218),
        ("ZIII", -0.22575),
        ("IIZZ", 0.12091),
        ("ZZII", 0.16893),
        ("XXXX", 0.04523),
    ]

    fresh_results = []
    for i in range(5):
        op = SparsePauliOp.from_list(pauli_terms)
        fresh_results.append(_expectation(sim, bound, op, f"fresh_{i}"))

    reused_op = SparsePauliOp.from_list(pauli_terms)
    reused_results = []
    for i in range(5):
        reused_results.append(_expectation(sim, bound, reused_op, f"reused_{i}"))

    print(f"Fresh-op-per-call results:  {fresh_results}")
    print(f"Reused-op-across-calls results: {reused_results}")

    max_fresh_spread = max(fresh_results) - min(fresh_results)
    max_reused_spread = max(reused_results) - min(reused_results)
    cross_delta = abs(fresh_results[0] - reused_results[0])

    print(f"Fresh-call internal spread:  {max_fresh_spread:.3e}")
    print(f"Reused-call internal spread: {max_reused_spread:.3e}")
    print(f"Fresh vs reused (1st call):  {cross_delta:.3e}")

    # Machine-epsilon-scale tolerance, same bar as GPU_EXPECTATION_FIX.md's
    # own fresh-vs-legacy-path check (5.55e-17 there). Allow a slightly
    # looser bound (1e-12) since this compares two different code paths'
    # floating-point summation order, not a bit-identical implementation.
    tol = 1e-12
    assert max_fresh_spread < tol, (
        f"Fresh SparsePauliOp-per-call is non-deterministic across repeats "
        f"(spread={max_fresh_spread:.3e}) -- something is wrong independent "
        f"of the reuse question; investigate before trusting either path."
    )
    assert max_reused_spread < tol, (
        f"REUSING a SparsePauliOp across save_expectation_value calls is "
        f"NOT SAFE -- results drifted across repeated calls on the SAME "
        f"reused op (spread={max_reused_spread:.3e}). Do NOT cache "
        f"SparsePauliOp in interface.py's _evaluate_distributed_statevector "
        f"/ _expectation_on_gpu; revert to building it fresh every call."
    )
    assert cross_delta < tol, (
        f"Fresh-per-call and reused-across-calls paths disagree "
        f"(delta={cross_delta:.3e}) -- caching changes the answer, not "
        f"just the speed. Do NOT trust the caching fix as-is."
    )

    print("\nPASS: SparsePauliOp reuse across save_expectation_value calls "
          "is numerically safe at this tolerance. The caching fix in "
          "src/api/interface.py is verified correct, not just assumed.")


if __name__ == "__main__":
    test_fresh_vs_reused_sparse_pauli_op()
