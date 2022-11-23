## How to build an environment

### Docker

1. Create the Docker image `docker build -t your_image_name`
2. Run your image `docker run -it your_image_name`
3. Access to your container `docker exec -it <container id> bash`

### Conda

1. Create the conda env `conda env create -f environment_[cpu|gpu].yml`
2. Activate the env `conda activate conda_env_HGCN_4_PLS_[cpu|gpu]`