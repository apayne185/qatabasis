FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopenmpi-dev \
    openmpi-bin \
    libcurl4-openssl-dev \
    libopenblas-dev \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libgfortran5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Core dependencies -- pinned to exact versions verified together
# (2026-09-22) via a real `pip install` resolution, not guessed. Previously
# fully unpinned: two Docker builds months apart could silently resolve
# different Qiskit/PySCF versions and produce different numerical results
# for the same benchmark -- a real reproducibility hole for a stack whose
# whole pitch is reproducible benchmark numbers. requirements.txt claims
# qiskit>=2.5.2,<3.0 and environment.yml claims qiskit>=1.0,<2.0; neither
# is what this Dockerfile actually installed before this pin -- a fresh
# resolve today lands on qiskit==2.5.2, matching requirements.txt, NOT
# environment.yml (environment.yml's pin is stale; see also its now
# nonexistent qiskit-aer-gpu dependency, a separate bug -- pip has no
# distribution under that name anymore, so `conda env create` on the
# native install path would fail outright on that line today).
RUN pip3 install --no-cache-dir \
    numpy==2.5.3 \
    scipy==1.18.1 \
    mpi4py==4.1.2 \
    pybind11==3.1.0 \
    qiskit==2.5.2 \
    qiskit-nature==0.8.0 \
    qiskit-ibm-runtime==0.49.0 \
    pyscf==2.14.0 \
    pytest==9.1.1 \
    matplotlib==3.11.2 \
    rdkit==2026.3.6

# GPU acceleration: cupy for CUDA, qiskit-aer built from source with GPU
# support. cupy-cuda12x pinned to the latest verified-available version as
# of 2026-09-22; qiskit-aer itself is NOT pinned here because it's built
# from source (--no-binary) against whatever CUDA toolkit this image
# provides, so pinning the pip package version alone wouldn't pin the
# actual build inputs -- if reproducing an exact Aer build matters, pin
# the base image tag (line 1) instead, which this Dockerfile already does.
RUN pip3 install --no-cache-dir cupy-cuda12x==14.2.0
RUN AER_THRUST_BACKEND=CUDA pip3 install --no-cache-dir qiskit-aer==0.17.2 --no-binary qiskit-aer

# Baseline-comparison deps (Pennylane Lightning). Kept as a separate layer
# so it caches cheaply after the heavy Aer-from-source build above.
# lightning-gpu is optional -- benchmarks/baseline_comparison.py falls back
# to lightning.qubit (CPU) if the GPU variant is not installed. Pinned to
# versions verified compatible with pennylane==0.45.1 as of 2026-09-22.
RUN pip3 install --no-cache-dir \
    pennylane==0.45.1 \
    pennylane-lightning==0.45.0 \
    pennylane-lightning-gpu==0.45.0

WORKDIR /workspace
COPY . /workspace

RUN mkdir -p build && cd build && \
    cmake .. \
      -DPython_EXECUTABLE=/usr/bin/python3.11 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
      -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

ENV PYTHONPATH="/workspace/build:/workspace"
ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV IBM_QUANTUM_TOKEN=""
ENV IBM_QUANTUM_INSTANCE=""
ENV IBM_QUANTUM_BACKEND="ibm_brisbane"
ENV IBM_QUANTUM_REGION="us-east"
ENV BACKEND="simulator"

CMD ["mpirun", "--allow-run-as-root", "-np", "2", "python3", "tests/test_layers_run.py"]