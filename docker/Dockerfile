FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    unzip \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core Python tools
RUN pip install --upgrade pip
RUN pip install \
    wandb \
    datasets \
    jupyterlab \
    "flash-attn<1.0.5"

# Clone the scGPT repo
RUN git clone https://github.com/bowang-lab/scGPT.git

# Remove R-dependent package from requirements
RUN sed -i '/scib/d' scGPT/docs/requirements.txt

# Install doc/tutorial dependencies (Python-only)
RUN pip install -r scGPT/docs/requirements.txt

# Set working directory inside the repo
WORKDIR /workspace/scGPT

# Install scGPT in editable mode
RUN pip install -e .

# Expose Jupyter port
EXPOSE 8888

# Default to bash
CMD ["/bin/bash"]

