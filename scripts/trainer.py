import argparse
import multiprocessing as mp
import os
import os.path as osp
from typing import Union

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

import bgcn_4_pls.data as data
import bgcn_4_pls.model as md


def train(atomic_distance_cutoff: float,
          nb_epochs: int, data_path: str,
          model_parameters_path: str):
    """Train the model

    Args:
        atomic_distance_cutoff (float): The cutoff to consider a link between a protein-ligand atom pair
        nb_epochs (int): The maximum number of epochs 
    """
    gpus = torch.cuda.device_count()
    use_gpu = gpus > 0
    accelerator = 'gpu' if use_gpu else None
    strategy = DDPPlugin(find_unused_parameters=False) if use_gpu else None
    devices = gpus if gpus > 0 else None
    exp_model_name = 'BGCN_4_PLS'
    experiments_path = osp.join('.', 'experiments')

    if not osp.isdir(experiments_path):
        os.mkdir(experiments_path)

    logger = pl.loggers.TensorBoardLogger(
        experiments_path, name=exp_model_name)

    version_path = osp.join(
        experiments_path, exp_model_name, 'version_' + str(logger.version))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=version_path,
                                                       save_top_k=1,
                                                       monitor="ep_end_val/loss",
                                                       mode='min')
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="ep_end_val/loss", mode="min", patience=20)

    callbacks = [pl.callbacks.LearningRateMonitor(
    ), checkpoint_callback, early_stopping_callback]

    with open(model_parameters_path, 'r') as f_yaml:
        model_parameters = yaml.safe_load(f_yaml)

    datamodule = data.PDBBindDataModule(root=data_path,
                                        atomic_distance_cutoff=atomic_distance_cutoff,
                                        batch_size=model_parameters['batch_size'],
                                        num_workers=mp.cpu_count(),
                                        only_pocket=True)

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
        plot_path=version_path,
        str_for_hparams="InterMol length: {}A".format(atomic_distance_cutoff))

    nb_param_trainable = model.get_nb_parameters(only_trainable=True)
    nb_param = model.get_nb_parameters(only_trainable=False)
    logger.log_metrics(
        {'nb_param_trainable': torch.tensor(nb_param_trainable)})
    logger.log_metrics({'nb_param': torch.tensor(nb_param)})

    trainer = pl.Trainer(accelerator=accelerator,
                         devices=devices,
                         strategy=strategy,
                         callbacks=callbacks,
                         max_epochs=nb_epochs,
                         logger=logger,
                         log_every_n_steps=2,
                         num_sanity_val_steps=0)

    trainer.fit(model, datamodule)

    group = torch.distributed.group.WORLD
    best_model_path = checkpoint_callback.best_model_path
    if gpus > 1:
        list_bmp = gpus * [None]
        torch.distributed.all_gather_object(
            list_bmp, best_model_path, group=group)
        best_model_path = list_bmp[0]

    print("Best Checkpoint path : ", best_model_path)
    test_best_model(best_model_path, datamodule, logger, 16)


@rank_zero_only
def test_best_model(best_model_path: str,
                    datamodule: pl.LightningDataModule,
                    logger: pl.loggers.TensorBoardLogger,
                    casf_version: Union[int, str]):
    """Test the given model on the dataloader CASF

    Args:
        best_model_path (str): Path to the best checkpoint
        datamodule (pl.LightningDataModule): PL Datamodule
        logger (pl.loggers.TensorBoardLogger): Tensorboard Logger
        casf_version (Union[int, str]): The Casf version, must be 13 or 16.
    """
    gpus = torch.cuda.device_count()
    use_gpu = gpus > 0
    accelerator = 'gpu' if use_gpu else None
    strategy = DDPPlugin(find_unused_parameters=False) if use_gpu else None
    devices = gpus if gpus > 0 else None
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=devices,
                         strategy=strategy,
                         max_epochs=1,
                         log_every_n_steps=0,
                         num_sanity_val_steps=0,
                         logger=logger)
    trained_model = md.Model.load_from_checkpoint(best_model_path)
    trained_model.set_casf_test(casf_version)
    trainer.test(trained_model, datamodule.casf_dataloader(casf_version))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nb_epochs', '-ep',
                        type=int,
                        help='The maximum number of epochs (defaults to 100)',
                        default=100)
    parser.add_argument('-cutoff', '-c',
                        type=float,
                        help='The cutoff to consider a link between a protein-ligand atom pair (defaults to 4.0)',
                        default=4.0)
    parser.add_argument('-data', '-d',
                        type=str,
                        required=True,
                        help='Path to the data directory')
    parser.add_argument('-model_parameters_path', '-mparam',
                        type=str,
                        required=True,
                        help='Path to the yaml model parameters')

    args = parser.parse_args()
    atomic_distance_cutoff = args.cutoff
    nb_epochs = args.nb_epochs
    data_path = args.data
    model_parameters_path = args.model_parameters_path
    train(atomic_distance_cutoff=atomic_distance_cutoff,
          nb_epochs=nb_epochs, data_path=data_path,
          model_parameters_path=model_parameters_path)
