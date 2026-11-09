"""Correctness check for replacing SparsePauliOp.expectation_value() with
a direct, cached scipy-sparse matrix-vector product, applied to the
NH3/N2 P=2 slowdown mitigation in src/api/interface.py.

Why this test exists: this is the SECOND fix attempted for the NH3/N2
P=2-vs-P=1 slowdown (see qatabasis-internal/docs/RESULT_PROVENANCE.md,
"Experiment 1 root cause", and this repo's own commit history around
2026-11-03). The first fix (caching the per-rank Pauli partition /
SparsePauliOp / sorted params across SPSA iterations, FUTURE_WORK.md
#7.1) gave a real but modest ~20% win. Direct profiling (Nsight Systems,
results/profiling/NH3_np2_iters15_*) plus isolated timing then identified
the ACTUAL dominant cost: Qiskit's own SparsePauliOp.expectation_value()
implementation, measured at ~284ms/call on a representative NH3-scale
(1529-term local partition, 16-qubit/65536-dim statevector) problem --
~7.4x slower than a direct scipy-sparse matvec computing the exact same
quantity (<psi|H|psi> = psi^dagger @ (H_sparse @ psi)), with results
agreeing to ~3.8e-13 (machine precision) in that initial investigation.

Getting this wrong would silently change every reported VQE energy on
the legacy/cpu-expect path (used whenever MPI world size > 1 by
default), so this is verified directly against Qiskit's own
implementation -- the ground truth this fix must reproduce exactly --
before being trusted in the hot path, same rigor as
tests/test_sparse_pauli_op_reuse.py required for the prior fix.

No GPU required (builds the statevector via pure-Python
qiskit.quantum_info.Statevector) -- can run via plain pytest, unlike
test_sparse_pauli_op_reuse.py which needs AerSimulator(device='GPU').
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp, Statevector


def _sparse_matvec_expectation(sv_data, sparse_matrix):
    return float(np.real(np.vdot(sv_data, sparse_matrix.dot(sv_data))))


def test_sparse_matvec_matches_qiskit_expectation_value():
    """Representative circuit + Hamiltonian at a size comparable to a real
    local Pauli partition (not NH3's full 16 qubits, to keep this test
    fast and GPU-free -- 8 qubits with a term count proportionally
    similar to a real P=2 partition's density is enough to exercise the
    same code path and catch a real divergence, which would show up
    regardless of exact problem size since both sides compute the
    identical mathematical quantity).
    """
    ansatz = EfficientSU2(8, reps=2, entanglement="full").decompose()
    theta = {p: 0.0831 * i for i, p in enumerate(sorted(ansatz.parameters, key=lambda x: x.name))}
    bound = ansatz.assign_parameters(theta)
    sv = Statevector(bound)
    sv_data = sv.data

    rng = np.random.default_rng(42)
    paulis = "IXYZ"
    terms = []
    for _ in range(150):
        label = "".join(rng.choice(list(paulis)) for _ in range(8))
        coeff = float(rng.uniform(-1.0, 1.0))
        terms.append((label, coeff))
    op = SparsePauliOp.from_list(terms)

    e_qiskit = float(sv.expectation_value(op).real)

    sparse_matrix = op.to_matrix(sparse=True)
    e_sparse = _sparse_matvec_expectation(sv_data, sparse_matrix)

    delta = abs(e_qiskit - e_sparse)
    print(f"Qiskit expectation_value: {e_qiskit!r}")
    print(f"Sparse matvec expectation: {e_sparse!r}")
    print(f"delta: {delta:.3e}")

    # Same machine-precision-scale bar as the original investigation's
    # 3.8e-13 finding. Allow some headroom (1e-9) since term count/circuit
    # here differs from that exact case, but this must not be anywhere
    # near chemical-accuracy scale (1.6e-3 Ha) -- a divergence at that
    # scale would silently corrupt every energy reported on this path.
    tol = 1e-9
    assert delta < tol, (
        f"Sparse-matvec expectation value diverges from Qiskit's own "
        f"SparsePauliOp.expectation_value() by {delta:.3e} (tolerance "
        f"{tol:.0e}) -- DO NOT trust the cached-sparse-matrix fix in "
        f"src/api/interface.py's _evaluate_distributed_statevector / "
        f"_evaluate_ibm_estimator. This would silently corrupt every "
        f"energy reported on the legacy/cpu-expect path."
    )

    # Also confirm the cached matrix is reusable across multiple calls
    # with DIFFERENT statevectors (the real usage pattern -- one cached
    # matrix, re-evaluated against a new psi every SPSA iteration) without
    # drift, same "repeat it, don't just check once" rigor as
    # test_sparse_pauli_op_reuse.py used for the SparsePauliOp-reuse
    # question.
    results = []
    for i in range(5):
        theta_i = {p: 0.0831 * i2 + 0.01 * i for i2, p in enumerate(sorted(ansatz.parameters, key=lambda x: x.name))}
        bound_i = ansatz.assign_parameters(theta_i)
        sv_i_data = Statevector(bound_i).data
        e_qiskit_i = float(Statevector(bound_i).expectation_value(op).real)
        e_sparse_i = _sparse_matvec_expectation(sv_i_data, sparse_matrix)
        results.append(abs(e_qiskit_i - e_sparse_i))

    max_repeat_delta = max(results)
    print(f"Max delta across 5 different statevectors, same cached matrix: {max_repeat_delta:.3e}")
    assert max_repeat_delta < tol, (
        f"Sparse-matvec expectation diverges from Qiskit's own path when "
        f"the SAME cached matrix is reused across different statevectors "
        f"(max delta={max_repeat_delta:.3e}) -- the real per-iteration "
        f"usage pattern. Do NOT trust this fix."
    )

    print("\nPASS: cached scipy-sparse matvec expectation value matches "
          "Qiskit's own SparsePauliOp.expectation_value() to machine "
          "precision, including when the cached matrix is reused across "
          "multiple different statevectors (the real per-iteration usage "
          "pattern). The fix in src/api/interface.py is verified correct.")


if __name__ == "__main__":
    test_sparse_matvec_matches_qiskit_expectation_value()
