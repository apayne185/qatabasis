# Available Molecules

## Built-in Registry

These molecules can be used directly by name with `ChemistryProblem.from_name()`:

```python
problem = ChemistryProblem.from_name("H2")     # any name from the table below
```

| Name | Formula | Electrons | Qubits | Pauli Terms | FCI Energy (Ha) | Ansatz Reps | Notes |
|------|---------|-----------|--------|-------------|-----------------|-------------|-------|
| `H2` | H₂ | 2 | 4 | 15 | -1.13727 | 1 | Fastest; ideal for testing and QPU runs |
| `LiH` | LiH | 4 | 12 | 631 | -7.8825 | 1 | Full active space (no core-electron freezing — see note below) |
| `BeH2` | BeH₂ | 6 | 14 | 666 | -15.5952 | 2 | Full active space (no core-electron freezing — see note below) |
| `H2O` | H₂O | 10 | 14 | 1086 | -75.0129 | 2 | Full active space (no core-electron freezing — see note below) |
| `NH3` | NH₃ | 10 | 16 | 3057 | -55.4546 | 3 | NISQ upper limit; long runtime |
| `N2` | N₂ | 14 | 20 | 2951 | -108.9544 | 2 | GPU crossover test (see `docs/GPU_EXPECTATION_FIX.md`) |
| `CO2` | CO₂ | 22 | 30 | ~16,170 | -187.6 | 1 | Ceiling test only, not a convergence run — 16k Pauli terms × default `MAX_ITERS` is a multi-day run; always pair with `MAX_ITERS<=10` |

**On "full active space" above**: all molecules built via `ChemistryProblem.from_name()` /
`ChemistryProblem.prepare()` — the path every benchmark and published result uses — run
the complete, untruncated STO-3G active space via `PySCFDriver` with no
`FreezeCoreTransformer`/`ActiveSpaceTransformer` applied. `MoleculeResolver.resolve(...,
freeze_core=True)` computes what a *reduced* active space's electron/qubit count would
be, but that computation only feeds `ResolutionResult.active_electrons` /
`estimated_qubits` (metadata/logging fields) — it is never applied to the actual
Hamiltonian construction. Confirmed: LiH's real, published qubit count (12) matches the
full 6-spatial-orbital STO-3G space exactly, not the freeze-core-reduced 10-qubit
estimate. If active-space reduction is wanted for a future molecule/basis, it would need
a real `FreezeCoreTransformer` wired into `ChemistryProblem.prepare()` — this does not
exist yet anywhere in the codebase.

All use the **STO-3G** minimal basis set. FCI energies are computed with PySCF Full Configuration Interaction.

## Custom Molecules

### Option A: Raw Geometry

Provide atom coordinates directly (Angstroms):

```python
# Single bond distance
problem = ChemistryProblem("H 0 0 0; H 0 0 0.74", name="H2_custom")

# Multi atom
problem = ChemistryProblem("C 0 0 0; O 0 0 1.128", name="CO")

# 3D geometry
problem = ChemistryProblem(
    "O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0",
    name="water"
)
```

### Option B: Molecule Resolver (SMILES, PubChem)

The resolver automatically looks up geometry from multiple sources

```python
from src.api.molecule_resolver import MoleculeResolver

resolver = MoleculeResolver(max_qubits=20)

# By common name (fetches from PubChem)
info = resolver.resolve("methane")

# By SMILES string (requires rdkit)
info = resolver.resolve("CCO")  # ethanol

# Use the resolved geometry
problem = ChemistryProblem(info.geometry, name=info.name)
```

Resolution cascade: **local registry** → **raw geometry** → **SMILES (rdkit)** → **PubChem API**

### Option C: Command Line

Pass molecule names directly to the benchmark runner:

```bash
make run NP=2 MOLECULES="H2 LiH"           # registry names
make run NP=2 MOLECULES="H2 BeH2 H2O"      # any combination
```

## Qubit Limits

The resolver enforces a configurable qubit cap (default 20) to prevent accidentally submitting circuits too large for NISQ hardware. Molecules exceeding this limit will be rejected with a `MoleculeTooBigError`.

## Adding New Molecules

Edit the `MOLECULE_REGISTRY` dictionary in `src/api/problems.py`:

```python
MOLECULE_REGISTRY = {
    "YourMolecule": {
        "geometry": "atom1 x y z; atom2 x y z; ...",
        "fci_energy": -X.XXXX,      # from literature or PySCF FCI
        "reps": 2,                   # ansatz repetitions (higher = more expressive, deeper circuit)
        "description": "Description, N electrons, M qubits",
    },
}
```
