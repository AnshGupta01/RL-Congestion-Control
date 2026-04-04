FROM mcr.microsoft.com/devcontainers/python:3.11

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python ML + Streamlit stack (M1 safe)
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    streamlit \
    jupyter \
    notebook \
    ipykernel \
    torch \
    torchvision \
    torchaudio

WORKDIR /workspace

# Enable Jupyter kernel
RUN python -m ipykernel install --user


# streamlit run app.py --server.port 8501 --server.address 0.0.0.0