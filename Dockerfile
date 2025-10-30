# syntax=docker/dockerfile:1.7
# Devcontainer-like image for HEGNN with CUDA 11.8 + PyTorch 2.1

ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# System deps
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-venv \
      python3-pip \
      ca-certificates \
      git \
      build-essential \
      curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user with UID/GID 1000
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

# Workspace dir (mounted via compose)
RUN mkdir -p /workspace && chown ${USERNAME}:${USERNAME} /workspace
WORKDIR /workspace

# Copy Python requirements early for layer caching
COPY --chown=${USERNAME}:${USERNAME} requirements.txt /tmp/requirements.txt

# Switch to dev user and create a venv in the home directory
USER ${USERNAME}
ENV VENV=/home/${USERNAME}/.venv
RUN python3 -m venv ${VENV} && \
    source ${VENV}/bin/activate && \
    python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.1.0 with CUDA 11.8 wheels, PyG scatter/sparse wheels, and the repo requirements
# Then install DGL for CUDA 11.8 (try PyPI alias first; fall back to DGL wheel repo if needed)
# Finally install optional dev tooling (wandb) for experiment tracking
RUN set -euxo pipefail; \
    source ${VENV}/bin/activate; \
    pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118; \
    pip install -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
        torch_scatter==2.1.2+pt21cu118 torch_sparse==0.6.18+pt21cu118; \
    pip install -r /tmp/requirements.txt; \
    (pip install 'dgl-cu118==1.1.3' || pip install 'dgl==1.1.3+cu118' -f https://data.dgl.ai/wheels/cu118/repo.html); \
    pip install wandb

# Auto-activate the venv for interactive shells
RUN echo 'if [ -d "$HOME/.venv" ]; then source "$HOME/.venv/bin/activate"; fi' >> /home/${USERNAME}/.bashrc

# Make venv binaries default on PATH for non-interactive shells, too
ENV PATH=${VENV}/bin:${PATH}

# Default command keeps container alive for dev; exec into it with bash
CMD ["sleep", "infinity"]
