FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
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
RUN conda install scikit-learn-intelex=2023.1.1 -y
RUN conda install pandas=2.1.1 -y
RUN conda install biopandas=0.4.1 -c conda-forge -y
RUN conda install pyg=2.4.0 -c pyg -y
RUN conda install pytorch-lightning=2.1.0 -c conda-forge -y
RUN conda install xorg-libxrender=0.9.10 -c conda-forge -y
RUN conda install scipy=1.7.3 -c anaconda -y
RUN conda install matplotlib=3.8.0 -c conda-forge -y
RUN conda install seaborn=0.12.2 -c conda-forge -y

RUN pip install oddt==0.7
RUN pip install class-resolver==0.3.10
RUN pip install pyg-lib -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
