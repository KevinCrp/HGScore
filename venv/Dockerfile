FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
WORKDIR "/home"

# Add missing Public Keys
RUN apt-get update 2>&1 | sed -ne 's?^.*NO_PUBKEY ??p' | xargs -r -- apt-key adv --keyserver keyserver.ubuntu.com --recv-keys

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

RUN conda install openbabel=3.1.1 -c conda-forge -y
RUN conda install scikit-learn-intelex=2021.5.0 -y
RUN conda install pandas=1.3.5 -y
RUN conda install biopandas=0.4.1 -c conda-forge -y
RUN conda install pyg=2.0.4 -c pyg -y
RUN conda install pytorch-lightning=1.5.9 -c conda-forge -y
RUN conda install xorg-libxrender=0.9.10 -c conda-forge -y
RUN conda install scipy=1.7.3 -c anaconda -y
RUN conda install matplotlib=3.5.3 -c conda-forge -y
RUN conda install oddt=0.7 -c conda-forge -y
RUN conda install seaborn=0.12.1 -c conda-forge -y

RUN pip install class-resolver==0.3.10
