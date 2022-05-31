FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
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
RUN conda install pytorch-lightning=1.5.9 -c conda-forge -y
RUN conda install -c conda-forge xorg-libxrender -y

RUN pip install scipy matplotlib
RUN pip install class-resolver
# As Long As conda pyg is 2.0.4
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-$1.11.0+cu113.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-$1.11.0+cu113.html
RUN pip install git+https://github.com/pyg-team/pytorch_geometric.git