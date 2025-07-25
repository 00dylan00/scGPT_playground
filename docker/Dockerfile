# Docker BUILD
# FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel


# Set working directory
# This keeps project directory seperate from system directory and uses it as a mount point
# for VSCode Dev Containers. Safe practice!
WORKDIR /workspace

# added bc Docker is a fucking annoying little piece of shit
ARG USERNAME=ddalton
ARG USER_UID=17447
ARG USER_GID=1003

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R $USERNAME:$USERNAME /workspace
    

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
# RUN pip install \
    # wandb \
    # datasets \
    # jupyterlab \
    # "flash-attn<1.0.5"



# WORKDIR where you'll mount your repo
WORKDIR /workspace/scGPT

CMD ["/bin/bash"]

# # Docker RUN    
# # Clone your modified scGPT repo
# # make sure it is getting latest
# RUN git clone https://github.com/00dylan00/scGPT.git && \
#     cd scGPT && git pull origin main


# # Set working directory inside your cloned repo
# WORKDIR /workspace/scGPT

# # (Optional) Remove R-based dependency from docs requirements
# RUN sed -i '/scib/d' docs/requirements.txt || true

# # Install tutorial/development dependencies
# RUN pip install -r docs/requirements.txt

# # Install scGPT in editable mode
# RUN pip install -e .

# # Expose Jupyter port
# EXPOSE 8888

# # Default shell
# CMD ["/bin/bash"]
