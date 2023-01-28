# HGScore

## *A Heterogeneous Graph Convolutional Network for Protein-Ligand Scoring*

![coverage badge](tests/badges/coverage.svg)
### Installation
1. Clone this repo
2. Create a conda environment or a Docker container with provided files. Dockerfile and YAML files are provided in the `./venv` directory
3. Install with `pip install .`


### Test
- Install pytest `pip install pytest`
- Run `pytest -v`

### Usage

1. Download the PDBBind database from http://www.pdbbind.org.cn/ with `scripts/download_pdbbind.sh`. Extracted PDBBind complexes are stored in *data/raw/*
You can also download the power_docking dataset for CASF's power docking assesment with `scripts/download_docking_power.sh`.
2. Our dataset split is provided with *data/\*.csv* files. Or make your own split using the `python scripts/split_pdbbind.py` script. Splits are Train, Val, Casf13, and Casf16
````bash
usage: split_pdbbind.py [-h] -data DATA [-nb_val NB_VAL]

optional arguments:
  -h, --help           show this help message and exit
  -data DATA, -d DATA  Path to the data directory
  -nb_val NB_VAL       Number of items from refined set used for validation (defaults to 1000)
````
3. Use `python scripts/create_graphs.py` to create all graphs
````bash
usage: create_graphs.py [-h] -data DATA [-pocket] [-cutoff CUTOFF] [-docking_power]

optional arguments:
  -h, --help            show this help message and exit
  -data DATA, -d DATA   Path to the data directory
  -pocket               Flag allowing to consider only the binding pocket as defined by PDBBind
  -cutoff CUTOFF, -c CUTOFF
                        The cutoff to consider a link between a protein-ligand atom pair (defaults to 4.0)
  -docking_power, -dp   Flag allowing to create the docking power dataset
````
4. Change model hyperparameters in `model_parameters.yaml`
5. Launch the training  `python scripts/trainer.py`
````bash
usage: trainer.py [-h] [-nb_epochs NB_EPOCHS] [-cutoff CUTOFF] -data DATA -model_parameters_path MODEL_PARAMETERS_PATH

optional arguments:
  -h, --help            show this help message and exit
  -nb_epochs NB_EPOCHS, -ep NB_EPOCHS
                        The maximum number of epochs (defaults to 100)
  -cutoff CUTOFF, -c CUTOFF
                        The cutoff to consider a link between a protein-ligand atom pair (defaults to 4.0)
  -data DATA, -d DATA   Path to the data directory
  -model_parameters_path MODEL_PARAMETERS_PATH, -mparam MODEL_PARAMETERS_PATH
                        Path to the yaml model parameters
````
6. Results will be saved in *experiments/HGScore/version_X*
7. Access them with 
````bash
tensorboard --logdir experiments/HGScore
````
8. Assess a trained model on CASF 13 & 16 with `python scripts/assess_model_on_casf.py`
````bash
usage: assess_model_on_casf.py [-h] -checkpoint_path CHECKPOINT_PATH [-plot] [-casf_13] [-casf_16] [-cutoff CUTOFF] [-docking_power]
                             [-docking_power_cutoff DOCKING_POWER_CUTOFF] [-pocket] -data DATA

optional arguments:
  -h, --help            show this help message and exit
  -checkpoint_path CHECKPOINT_PATH, -ckpt CHECKPOINT_PATH
                        Path to the torch Checkpoint
  -plot, -p             Do plot
  -casf_13              Test on CASF Core set v2013
  -casf_16              Test on CASF Core set v2016
  -cutoff CUTOFF, -c CUTOFF
                        The cutoff to consider a link between a protein-ligand atom pair (defaults to 4.0)
  -docking_power, -dp   Flag allowing to compute the docking power
  -docking_power_cutoff DOCKING_POWER_CUTOFF, -dpc DOCKING_POWER_CUTOFF
                        The RMSD cutoff (in angstrom) to define near-native docking pose for Docking Power (defaults to 2.0)
  -pocket               Flag allowing to consider only the binding pocket as defined by PDBBind
  -data DATA, -d DATA   Path to the data directory
````
````bash
python scripts/assess_model_on_casf.py -ckpt models/model.ckpt -c 4.0 -p -casf_13 -casf_16 -d data -pocket
````
9. Score a protein-ligand complex with `python scripts/predict.py`

*According to the PDBBind pocket extraction strategy we consider a residue as being part of the binding pocket if at least one residue's heavy atom is close to a cutoff (the cutoff used by PDBBind is 10.0A) of at least one ligand's heavy atom.*
````bash
usage: predict.py [-h] -checkpoint_path CHECKPOINT_PATH -protein_path PROTEIN_PATH -ligand_path LIGAND_PATH [-cutoff CUTOFF] [-extract_pocket]
                  [-extract_pocket_cutoff EXTRACT_POCKET_CUTOFF]

optional arguments:
  -h, --help            show this help message and exit
  -checkpoint_path CHECKPOINT_PATH, -ckpt CHECKPOINT_PATH
                        Path to the torch Checkpoint
  -protein_path PROTEIN_PATH, -p PROTEIN_PATH
                        Path to the protein PDB file
  -ligand_path LIGAND_PATH, -l LIGAND_PATH
                        Path to the ligand MOL2/PDB file
  -cutoff CUTOFF, -c CUTOFF
                        The cutoff to consider a link between a protein-ligand atom pair (Defaults to 4.0A)
  -extract_pocket       Extract the pocket according to the ligand's position, no necessary if the pocket is already provided by protein path
  -extract_pocket_cutoff EXTRACT_POCKET_CUTOFF
                        Cutoff for pocket extraction (Defaults to 10.0A)
````