import argparse
import multiprocessing as mp
import os.path as osp

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only

import config as cfg
import data
import model as md


def train(atomic_distance_cutoff: float):
    gpus = torch.cuda.device_count()
    use_gpu = gpus > 0
    strategy = 'ddp' if use_gpu else None
    exp_model_name = 'BG_PLS'

    logger = pl.loggers.TensorBoardLogger(
        cfg.experiments_path, name=exp_model_name)

    version_path = osp.join(
        cfg.experiments_path, exp_model_name, 'version_' + str(logger.version))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=version_path,
                                                       save_top_k=1,
                                                       monitor="ep_end_val/loss",
                                                       mode='min')
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="ep_end_val/loss", mode="min", patience=20)

    callbacks = [pl.callbacks.LearningRateMonitor(
    ), checkpoint_callback, early_stopping_callback]

    datamodule = data.PDBBindDataModule(root=cfg.data_path,
                                        atomic_distance_cutoff=atomic_distance_cutoff,
                                        batch_size=cfg.batch_size,
                                        num_workers=mp.cpu_count(),
                                        only_pocket=True,
                                        sample_percent=100.0)

    model = md.Model(
        hidden_channels_pa=cfg.hidden_channels_pa,
        hidden_channels_la=cfg.hidden_channels_pa,
        num_layers=cfg.num_layers,
        dropout=cfg.p_dropout,
        heads=cfg.heads,
        hetero_aggr=cfg.hetero_aggr,
        mlp_channels=cfg.mlp_channels,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        plot_path=version_path,
        num_timesteps=cfg.num_timesteps,
        str_for_hparams="InterMol length: {}A".format(atomic_distance_cutoff))

    nb_param_trainable = model.get_nb_parameters(only_trainable=True)
    nb_param = model.get_nb_parameters(only_trainable=False)
    logger.log_metrics(
        {'nb_param_trainable': torch.tensor(nb_param_trainable)})
    logger.log_metrics({'nb_param': torch.tensor(nb_param)})

    if use_gpu:
        trainer = pl.Trainer(accelerator='gpu',
                             devices=gpus,
                             strategy=strategy,
                             callbacks=callbacks,
                             max_epochs=cfg.nb_epochs,
                             logger=logger,
                             log_every_n_steps=2,
                             num_sanity_val_steps=0)
    else:
        trainer = pl.Trainer(callbacks=callbacks,
                             max_epochs=cfg.nb_epochs,
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
    test_best_model_16(best_model_path, datamodule, logger)


@rank_zero_only
def test_best_model_13(best_model_path: str,
                       datamodule: pl.LightningDataModule,
                       logger):
    """Test the given model on the dataloader CASF13

    Args:
        best_model_path (str): Path to the best checkpoint
        datamodule (pl.LightningDataModule): PL Datamodule
    """
    trainer = pl.Trainer(max_epochs=1,
                         log_every_n_steps=2,
                         num_sanity_val_steps=0,
                         logger=logger)
    trained_model = md.Model.load_from_checkpoint(best_model_path)
    trained_model.set_casf_test(13)
    trainer.test(trained_model, datamodule.casf_13_dataloader())


@rank_zero_only
def test_best_model_16(best_model_path: str,
                       datamodule: pl.LightningDataModule,
                       logger):
    """Test the given model on the dataloader CASF16

    Args:
        best_model_path (str): Path to the best checkpoint
        datamodule (pl.LightningDataModule): PL Datamodule
    """
    trainer = pl.Trainer(max_epochs=1,
                         log_every_n_steps=2,
                         num_sanity_val_steps=0,
                         logger=logger)
    trained_model = md.Model.load_from_checkpoint(best_model_path)
    trained_model.set_casf_test(16)
    trainer.test(trained_model, datamodule.casf_16_dataloader())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cutoff', '-c',
                        type=float,
                        help='The cutoff to consider a link between a protein-ligand atom pair',
                        default=4.0)
    args = parser.parse_args()
    atomic_distance_cutoff = args.cutoff
    train(atomic_distance_cutoff)
