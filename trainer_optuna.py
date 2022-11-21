import math
import multiprocessing as mp
import os
import os.path as osp
from typing import List, Tuple, Union

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from termcolor import cprint

import bgcn_4_pls.data as data
import bgcn_4_pls.model as md
import config_optuna as cfg_hpo

OPTUNA_VERSION = "A_avg_C"
# OPTUNA_VERSION = "AB_avg_C"
# OPTUNA_VERSION = "AC"

LOG_FILE = "trials_all_result_"+OPTUNA_VERSION+".txt"


def get_layer(first_2_pow, num_layers, factor=1):
    # layer = [2**first_2_pow]
    layer = [first_2_pow]
    for i in range(num_layers - 1):
        layer += [int(layer[i] * factor)]
    return layer


def train(hidden_channels_pa: List[int],
          hidden_channels_la: List[int],
          num_layers: int,
          heads: int,
          mlp_channels: List[int],
          lr: float,
          molecular_embedding_size: int,
          dropout: float,
          hetero_aggr: str,
          weight_decay: float,
          intermol_atomic_cutoff: float,
          nb_epochs: int,
          data_path: str) -> Tuple[float, int]:
    gpus = torch.cuda.device_count()
    use_gpu = gpus > 0
    accelerator = 'gpu' if use_gpu else None
    strategy = DDPStrategy(find_unused_parameters=False) if use_gpu else None
    devices = gpus if gpus > 0 else None

    exp_model_name = 'BG_PLS_OPTUNA_'+OPTUNA_VERSION
    experiments_path = osp.join('.', 'experiments')

    logger = pl.loggers.TensorBoardLogger(
        experiments_path, name=exp_model_name)

    version_path = osp.join(
        experiments_path, exp_model_name, 'version_' + str(logger.version))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=version_path,
                                                       save_top_k=1,
                                                       monitor="ep_end_val/pearson",
                                                       mode='max')
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="ep_end_val/loss", mode="min", patience=20)

    callbacks = [pl.callbacks.LearningRateMonitor(
    ), checkpoint_callback, early_stopping_callback]


    datamodule = data.PDBBindDataModule(root=data_path,
                                        atomic_distance_cutoff=intermol_atomic_cutoff,
                                        batch_size=32,
                                        num_workers=mp.cpu_count(),
                                        only_pocket=True)

    model = md.Model(
        hidden_channels_pa=hidden_channels_pa,
        hidden_channels_la=hidden_channels_la,
        num_layers=num_layers,
        dropout=dropout,
        heads=heads,
        hetero_aggr=hetero_aggr,
        mlp_channels=mlp_channels,
        lr=lr,
        weight_decay=weight_decay,
        molecular_embedding_size=molecular_embedding_size,
        plot_path=version_path,
        str_for_hparams="InterMol length: {}A".format(intermol_atomic_cutoff))

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
    final_ckpt_path = osp.join(version_path, 'final_checkpoint.ckpt')
    trainer.save_checkpoint(final_ckpt_path)

    test_best_model(final_ckpt_path, datamodule, logger, 13,
                    accelerator=accelerator,
                    devices=devices,
                    strategy=strategy,
                    info='final')
    test_best_model(final_ckpt_path, datamodule, logger, 16,
                    accelerator=accelerator,
                    devices=devices,
                    strategy=strategy,
                    info='final')

    best_model_path = checkpoint_callback.best_model_path
    test_best_model(best_model_path, datamodule, logger, 13,
                    accelerator=accelerator,
                    devices=devices,
                    strategy=strategy,
                    info='best')
    test_results_ckpt_best = test_best_model(best_model_path, datamodule, logger, 16,
                                             accelerator=accelerator,
                                             devices=devices,
                                             strategy=strategy,
                                             info='best')
    test_results_ckpt_best_pearson = test_results_ckpt_best[0]['casf_16_best/pearson']

    if math.isnan(test_results_ckpt_best_pearson):
        test_results_ckpt_best_pearson = 0.0
    return test_results_ckpt_best_pearson, logger.version


@rank_zero_only
def test_best_model(best_model_path: str,
                    datamodule: pl.LightningDataModule,
                    logger: pl.loggers.TensorBoardLogger,
                    casf_version: Union[int, str],
                    **kwargs):
    """Test the given model on the dataloader CASF
    Args:
        best_model_path (str): Path to the best checkpoint
        datamodule (pl.LightningDataModule): PL Datamodule
        logger (pl.loggers.TensorBoardLogger): Tensorboard Logger
        casf_version (Union[int, str]): The Casf version, must be 13 or 16.
        **kwargs: Additional arguments of pytorch_lightning.Trainer
    """
    gpus = torch.cuda.device_count()
    use_gpu = gpus > 0
    accelerator = kwargs.get('accelerator', None)
    devices = kwargs.get('devices', None)
    strategy = DDPStrategy(find_unused_parameters=False) if use_gpu else None
    info = kwargs.get('info', "")
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=devices,
                         strategy=strategy,
                         max_epochs=1,
                         log_every_n_steps=0,
                         num_sanity_val_steps=0,
                         logger=logger)
    trained_model = md.Model.load_from_checkpoint(best_model_path)
    trained_model.set_casf_test(casf_version, info=info)
    return trainer.test(trained_model, datamodule.casf_dataloader(casf_version))


def run_n_trains(trial) -> float:

    num_layers = trial.suggest_int("nb_layers_atomic",
                                   cfg_hpo.num_layers_atm_low,
                                   cfg_hpo.num_layers_atm_high)
    layer_size_factor_pa = trial.suggest_float("layer_size_factor_pa",
                                               cfg_hpo.layer_size_factor_low_pa,
                                               cfg_hpo.layer_size_factor_high_pa,
                                               step=0.1)
    layer_size_factor_la = trial.suggest_float("layer_size_factor_la",
                                               cfg_hpo.layer_size_factor_low_la,
                                               cfg_hpo.layer_size_factor_high_la,
                                               step=0.1)
    hidden_channels_pa = get_layer(trial.suggest_int(
        "1st_layer_pa",
        cfg_hpo.layer_1st_pa_low,
        cfg_hpo.layer_1st_pa_high,
        step=2),
        num_layers,
        layer_size_factor_pa)
    hidden_channels_la = get_layer(trial.suggest_int(
        "1st_layer_la",
        cfg_hpo.layer_1st_la_low,
        cfg_hpo.layer_1st_la_high,
        step=2),
        num_layers,
        layer_size_factor_la)
    heads = trial.suggest_int("heads",
                              cfg_hpo.heads_low,
                              cfg_hpo.heads_high)
    lr_pow = trial.suggest_int("learning_rate_pow",
                               cfg_hpo.learning_rate_pow_low,
                               cfg_hpo.learning_rate_pow_high)
    lr = pow(10, -lr_pow)
    molecular_embedding_size = trial.suggest_int("molecular_embedding_size",
                                                 cfg_hpo.nb_molecular_embedding_size_low,
                                                 cfg_hpo.nb_molecular_embedding_size_high)
    nb_mlp_layer = trial.suggest_int("nb_mlp_layer",
                                     cfg_hpo.nb_mlp_layer_low,
                                     cfg_hpo.nb_mlp_layer_high)
    mlp_channels = [hidden_channels_pa[-1] + hidden_channels_la[-1]]
    for i in range(nb_mlp_layer - 1):
        new_size = int(mlp_channels[i] / 2)
        if new_size > 1:
            mlp_channels += [new_size]
        else:
            break
    mlp_channels += [1]
    dropout = trial.suggest_float("dropout",
                                  cfg_hpo.dropout_low,
                                  cfg_hpo.dropout_high,
                                  step=0.01)
    hetero_aggr = trial.suggest_categorical("hetero_aggr",
                                            cfg_hpo.hetero_aggr)
    weight_decay_pow = trial.suggest_int("weight_decay_pow",
                                         cfg_hpo.weight_decay_pow_low,
                                         cfg_hpo.weight_decay_pow_high)
    weight_decay = pow(10, -weight_decay_pow)
    intermol_atomic_cutoff = trial.suggest_categorical("intermol_atomic_cutoff",
                                                       cfg_hpo.intermol_atomic_cutoff)

    best_value = 0.0
    nb_runs = cfg_hpo.nb_runs
    cprint("[HPO_{} Trial_{}: Params {}]".format(
        os.getpid(), trial.number, trial.params), 'green')
    with open(LOG_FILE, 'a') as f:
        f.write("\n[HPO_{} Trial_{}: Params {}]\n".format(
            os.getpid(), trial.number, trial.params))
    for i in range(nb_runs):
        cprint("[HPO_{} Trial_{}: launch training {} / {}]".format(os.getpid(),
                                                                   trial.number,
               i + 1, nb_runs), 'green')
        with open(LOG_FILE, 'a') as f:
            f.write(
                "[HPO_{} Trial_{}: launch training {} / {}]\n".format(os.getpid(),
                                                                      trial.number,
                                                                      i + 1,
                                                                      nb_runs))
        val, version = train(hidden_channels_pa,
                             hidden_channels_la,
                             num_layers,
                             heads,
                             mlp_channels,
                             lr,
                             molecular_embedding_size,
                             dropout,
                             hetero_aggr,
                             weight_decay,
                             intermol_atomic_cutoff,
                             cfg_hpo.nb_epochs,
                             'data')
        cprint("[HPO_{}  Trial_{} (version {}): training value: {}]".format(
            os.getpid(), trial.number, version, val), 'green')
        with open(LOG_FILE, 'a') as f:
            f.write("[HPO_{}  Trial_{} (version {}): training value: {}]\n".format(
                os.getpid(), trial.number, version, val))
        if val > best_value:
            best_value = val
            cprint("[HPO_{}  Trial_{} (version {}): new best value {}]".format(
                os.getpid(), trial.number, version, best_value), 'green')
            with open(LOG_FILE, 'a') as f:
                f.write("[HPO_{}  Trial_{} (version {}): new best value {}]\n".format(
                    os.getpid(), trial.number, version, best_value))
    return best_value


if __name__ == '__main__':
    storage = "mysql+pymysql://optuna:password@localhost/bg_pls_hpo_"+OPTUNA_VERSION
    study = optuna.create_study(storage=storage,
                                direction='maximize',
                                study_name='bg_pls_hpo_'+OPTUNA_VERSION,
                                load_if_exists=True)
    cprint("[HPO_{}: New set of trials]".format(os.getpid()), 'green')
    with open(LOG_FILE, 'a') as f:
        f.write("[HPO_{}: New set of trials]".format(os.getpid()))
    study.optimize(run_n_trains, n_trials=cfg_hpo.nb_trials)