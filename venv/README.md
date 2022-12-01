## How to build an environment

### Docker

1. Create the Docker image `docker build -t your_image_name`
2. Run your image `docker run -it your_image_name`
3. Access to your container `docker exec -it <container id> bash`

### Conda

1. Create the conda env `conda env create -f environment_[cpu|gpu].yml`
2. Activate the env `conda activate conda_env_HGCN_4_PLS_[cpu|gpu]`
3. Install Pyg-Lib :
> pyg-lib provides efficient GPU-based routines to parallelize workloads in heterogeneous graphs across different node types and edge types.


* CPU version: `pip install pyg-lib -f https://data.pyg.org/whl/torch-1.12.0+cpu.html`
* GPU version: `pip install pyg-lib -f https://data.pyg.org/whl/torch-1.12.0+cu113.html`