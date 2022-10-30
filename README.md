# Bipartite Graph Convolutional Network for Protein-Ligand Scoring

## Usage

1. Create a conda environment or a Docker container with provided files
2. Download the PDBBind database from http://www.pdbbind.org.cn/ with `download_pdbbind.sh`. Extracted PDBBind complexes are stored in *data/raw/*
3. Our dataset split is provided with *data/\*.csv* files. Or make your own split using the `split_pdbbind.py` script. Splits are Train, Val, Casf13, and Casf16
````bash
python split_pdbbind.py
````
4. Now in *data/* there are four csv files (*[train|val|casf13|casf16].csv*)
5. Use `python data.py` to create all graphs
````bash
usage: data.py [-h] [-cutoff CUTOFF] [-docking_power]

optional arguments:
  -h, --help            show this help message and exit
  -cutoff CUTOFF, -c CUTOFF
                        The cutoff to consider a link between a protein-ligand atom pair (defaults to 4.0)
  -docking_power, -dp   Flag allowing to create the docking power dataset
````
6. Change model hyperparameters in `model_parameters.yaml`
7. Launch the training 
````bash
usage: trainer.py [-h] [-nb_epochs NB_EPOCHS] [-cutoff CUTOFF]

optional arguments:
  -h, --help            show this help message and exit
  -nb_epochs NB_EPOCHS, -ep NB_EPOCHS
                        The maximum number of epochs (defaults to 100).
  -cutoff CUTOFF, -c CUTOFF
                        The cutoff to consider a link between a protein-ligand atom pair (defaults to 4.0).
````
8. Results will be saved in *experiments/BG_PLS/version_X*
9. Access them with 
````bash
tensorboard --logdir experiments/BG_PLS
````
10. ReTest a trained model on CASF 13 & 16 with `test_model_on_casf.py`
````bash
usage: test_model_on_casf.py [-h] -checkpoint_path CHECKPOINT_PATH [-plot] [-casf_13] [-casf_16] [-cutoff CUTOFF] [-docking_power] [-docking_power_cutoff DOCKING_POWER_CUTOFF]

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
````
````bash
python test_model_on_casf.py -ckpt models/model.ckpt -c 4.0 -p -casf_13 -casf_16
````
11. Score a protein-ligand complex with `python predict.py`
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