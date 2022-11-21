import os.path as osp

import pytorch_lightning as pl
import torch
import torch_geometric as pyg
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin

import bgcn_4_pls.data as data
from bgcn_4_pls import model as md
from bgcn_4_pls.data import process_graph_from_files
from bgcn_4_pls.utilities.pockets import pocket_extraction

MODEL_HPARAM_PATH = "tests/model_parameters.yaml"
MODEL_PATH = 'models/model.ckpt'
DATA_ROOT = 'tests/data'
ATM_CUTOFF = 4.0
POCKET = True
PATH_TO_POCKET_PDB = 'tests/data/raw/4llx/4llx_pocket.pdb'
PATH_TO_PROTEIN_PDB = 'tests/data/raw/4llx/4llx_protein.pdb'
PATH_TO_LIGAND_MOL2 = 'tests/data/raw/4llx/4llx_ligand.mol2'
PATH_TO_CUSTOM_POCKET = 'tests/data/raw/4llx/4llx_own_pocket.pdb'
PATH_PL_LOGS = "tests"

def test_model_init_with_list():
    with open(MODEL_HPARAM_PATH, 'r') as f_yaml:
        model_parameters = yaml.safe_load(f_yaml)

    model = md.Model(
        hidden_channels_pa=model_parameters['hidden_channels_pa'],
        hidden_channels_la=model_parameters['hidden_channels_la'],
        num_layers=model_parameters['num_layers'],
        dropout=model_parameters['dropout'],
        heads=model_parameters['heads'],
        hetero_aggr=model_parameters['hetero_aggr'],
        mlp_channels=model_parameters['mlp_channels'],
        lr=model_parameters['lr'],
        weight_decay=model_parameters['weight_decay'],
        molecular_embedding_size=model_parameters['molecular_embedding_size'],
        plot_path="tests",
        str_for_hparams="InterMol length: {}A".format(ATM_CUTOFF))

    nb_param_trainable = model.get_nb_parameters(only_trainable=True)
    assert nb_param_trainable == 1177091
    nb_param = model.get_nb_parameters(only_trainable=False)
    assert nb_param == 1177091

def test_model_init_with_int():
    with open(MODEL_HPARAM_PATH, 'r') as f_yaml:
        model_parameters = yaml.safe_load(f_yaml)

    model = md.Model(
        hidden_channels_pa=36,
        hidden_channels_la=35,
        num_layers=model_parameters['num_layers'],
        dropout=model_parameters['dropout'],
        heads=model_parameters['heads'],
        hetero_aggr=model_parameters['hetero_aggr'],
        mlp_channels=model_parameters['mlp_channels'],
        lr=model_parameters['lr'],
        weight_decay=model_parameters['weight_decay'],
        molecular_embedding_size=model_parameters['molecular_embedding_size'],
        plot_path="tests",
        str_for_hparams="InterMol length: {}A".format(ATM_CUTOFF))

    nb_param_trainable = model.get_nb_parameters(only_trainable=True)
    assert nb_param_trainable == 252698
    nb_param = model.get_nb_parameters(only_trainable=False)
    assert nb_param == 252698


def test_model_predict_from_pocket():
    bipartite_graph = process_graph_from_files(PATH_TO_POCKET_PDB,
                                               PATH_TO_LIGAND_MOL2,
                                               atomic_distance_cutoff=ATM_CUTOFF)
    batch = pyg.data.Batch.from_data_list([bipartite_graph])

    model = md.Model.load_from_checkpoint(MODEL_PATH)

    score = model.predict(batch)
    assert round(score, 2) == 4.21


def test_model_forward_predict_from_protein():
    pocket_extraction(PATH_TO_PROTEIN_PDB, PATH_TO_LIGAND_MOL2,
                      PATH_TO_CUSTOM_POCKET, 10.0)
    bipartite_graph = process_graph_from_files(PATH_TO_CUSTOM_POCKET,
                                               PATH_TO_LIGAND_MOL2,
                                               atomic_distance_cutoff=ATM_CUTOFF)
    batch = pyg.data.Batch.from_data_list([bipartite_graph])

    model = md.Model.load_from_checkpoint(MODEL_PATH)

    score = model.predict(batch)
    assert round(score, 2) == 5.21



def test_training_gpu():
    gpus = torch.cuda.device_count()
    use_gpu = gpus > 0
    if use_gpu:
        accelerator = 'gpu' if use_gpu else None
        strategy = DDPPlugin(find_unused_parameters=False) if use_gpu else None
        devices = gpus if gpus > 0 else None
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=PATH_PL_LOGS+'/')
        with open(MODEL_HPARAM_PATH, 'r') as f_yaml:
            model_parameters = yaml.safe_load(f_yaml)

        model = md.Model(
            hidden_channels_pa=model_parameters['hidden_channels_pa'],
            hidden_channels_la=model_parameters['hidden_channels_la'],
            num_layers=model_parameters['num_layers'],
            dropout=model_parameters['dropout'],
            heads=model_parameters['heads'],
            hetero_aggr=model_parameters['hetero_aggr'],
            mlp_channels=model_parameters['mlp_channels'],
            lr=model_parameters['lr'],
            weight_decay=model_parameters['weight_decay'],
            molecular_embedding_size=model_parameters['molecular_embedding_size'],
            plot_path="tests",
            str_for_hparams="InterMol length: {}A".format(ATM_CUTOFF))

        datamodule = data.PDBBindDataModule(root=DATA_ROOT,
                                            atomic_distance_cutoff=ATM_CUTOFF,
                                            batch_size=5,
                                            num_workers=1,
                                            only_pocket=POCKET)

        trainer = pl.Trainer(accelerator=accelerator,
                             devices=devices,
                             strategy=strategy,
                             max_epochs=1,
                             log_every_n_steps=2000,
                             num_sanity_val_steps=0,
                             logger=tb_logger
                             )

        trainer.fit(model, datamodule)

def test_training_cpu():
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=PATH_PL_LOGS+'/')
    with open(MODEL_HPARAM_PATH, 'r') as f_yaml:
        model_parameters = yaml.safe_load(f_yaml)

    model = md.Model(
        hidden_channels_pa=model_parameters['hidden_channels_pa'],
        hidden_channels_la=model_parameters['hidden_channels_la'],
        num_layers=model_parameters['num_layers'],
        dropout=model_parameters['dropout'],
        heads=model_parameters['heads'],
        hetero_aggr=model_parameters['hetero_aggr'],
        mlp_channels=model_parameters['mlp_channels'],
        lr=model_parameters['lr'],
        weight_decay=model_parameters['weight_decay'],
        molecular_embedding_size=model_parameters['molecular_embedding_size'],
        plot_path="tests",
        str_for_hparams="InterMol length: {}A".format(ATM_CUTOFF))

    datamodule = data.PDBBindDataModule(root=DATA_ROOT,
                                        atomic_distance_cutoff=ATM_CUTOFF,
                                        batch_size=5,
                                        num_workers=1,
                                        only_pocket=POCKET)

    trainer = pl.Trainer(max_epochs=1,
                         log_every_n_steps=2000,
                         num_sanity_val_steps=0,
                         logger=tb_logger
                        )

    trainer.fit(model, datamodule)
        