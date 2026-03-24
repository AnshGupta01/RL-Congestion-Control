FROM continuumio/miniconda3:latest

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Create env (optional) and install common libs
RUN conda install -y \
    python=3.11 \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    ipykernel \
&& conda clean -afy

# (Optional) set working dir and default command
WORKDIR /workspace
CMD ["python"]