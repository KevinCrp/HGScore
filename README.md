# Bipartite Graph for Protein-Ligand Scoring

## Usage

1. Create a conda environment or a Docker container with provided files
2. Download the PDBBind database from http://www.pdbbind.org.cn/ with `download_pdbbind.sh`. Extracted PDBBind complexes are stored in *data/raw/*
3. Our dataset split is provided with *data/*.csv* files. Or make your own split using the `split_pdbbind.py` script. Splits are Train, Val, Casf13, and Casf16
````bash
python split_pdbbind.py
````
4. Now in *data/* there are four csv files (*[train|val|casf13|casf16].csv*)
5. Use `data.py` to create all graphs
````bash
python data.py
````
6. Change model hyperparameters in `config.py`
7. Launch the training 
````bash
python train.py
````
8. Results will be saved in *experiments/BG_PLS/version_X*
9. Access them with 
````bash
tensorboard --logdir experiments/BG_PLS
````
10. ReTest a trained model on CASF 13 & 16 with `test_model_on_casf.py`
````bash
usage: test_model_on_casf.py [-h] -checkpoint_path CHECKPOINT_PATH [-plot] [-casf_13] [-casf_16]

optional arguments:
  -h, --help            show this help message and exit
  -checkpoint_path CHECKPOINT_PATH, -c CHECKPOINT_PATH
                        Path to the torch Checkpoint
  -plot, -p             Do plot
  -casf_13              Test on CASF Core set v2013
  -casf_16              Test on CASF Core set v2016
````
````bash
python test_model_on_casf.py -c models/model.ckpt -p -casf_13 -casf_16
````
11. Score a protein-ligand complex with `predict.py`
````bash
usage: predict.py [-h] -checkpoint_path CHECKPOINT_PATH -protein_path PROTEIN_PATH -ligand_path LIGAND_PATH

optional arguments:
  -h, --help            show this help message and exit
  -checkpoint_path CHECKPOINT_PATH, -c CHECKPOINT_PATH
                        Path to the torch Checkpoint
  -protein_path PROTEIN_PATH, -p PROTEIN_PATH
                        Path to the protein PDB file
  -ligand_path LIGAND_PATH, -l LIGAND_PATH
                        Path to the ligand MOL2 file
````