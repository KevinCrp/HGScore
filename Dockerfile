FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
WORKDIR "/home"

# Install base utilities
RUN apt-get update && apt-get install -y build-essential \
    autoconf \
    htop \
    nano \
    wget \
    unzip \
    git
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

WORKDIR "/workspace"

RUN conda install -c conda-forge openbabel -y
RUN conda install scikit-learn-intelex -y
RUN conda install pandas -y
RUN conda install biopandas -c conda-forge -y
RUN conda install pyg=2.0.4 -c pyg -y
RUN conda install pytorch-lightning=1.5.9 -c conda-forge -y
RUN conda install -c conda-forge xorg-libxrender -y

RUN pip install scipy matplotlib
RUN pip install class-resolver
RUN pip install oddt
