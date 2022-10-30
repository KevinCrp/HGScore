import logging
import os.path as osp
import sys
from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torchmetrics.functional as tmf

import plotters
from casf.ranking_power import ranking_power
from casf.scoring_power import scoring_power
from networks.bipartite_afp import BGCN_4_PLS


class Model(pl.LightningModule):
    """A Pytorch Lightning Model
    """

    def __init__(self,
                 hidden_channels_pa: Union[int, List[int]],
                 hidden_channels_la: Union[int, List[int]],
                 num_layers: int,
                 dropout: float,
                 heads: int,
                 hetero_aggr: str,
                 mlp_channels: List,
                 lr: float,
                 weight_decay: float,
                 plot_path: str,
                 molecular_embedding_size: int,
                 str_for_hparams: str = ''):
        """_summary_

        Args:
            hidden_channels_pa (Union[int, List[int]]): The size of channels
                for the protein part
            hidden_channels_la (Union[int, List[int]]): The size of channels
                for the ligand part
            num_layers (int): The number of layers
            dropout (float): Dropout rate
            heads (int): Number of heads
            hetero_aggr (str): How the hetero aggregation is did
            mlp_channels (List): List of final MLP channels siz
            lr (float): The learning rate
            weight_decay (float): The weight decay
            plot_path (str): Where save plots
            molecular_embedding_size (int): Number of timestep for molecular embedding
            str_for_hparams (str, optional): Allowing to save supplementary
                information for tensorboard. Defaults to ''.
        """
        super().__init__()
        self.save_hyperparameters()
        if isinstance(hidden_channels_pa, int):
            hidden_channels_pa = num_layers * [hidden_channels_pa]
        if isinstance(hidden_channels_la, int):
            hidden_channels_la = num_layers * [hidden_channels_la]
        if len(hidden_channels_pa) != num_layers:
            logging.error("The num_layer doesn't match the given layer sizes in"
                          " config.ini")
            sys.exit()
        self.plot_path = plot_path
        self.model = BGCN_4_PLS(list_hidden_channels_pa=hidden_channels_pa,
                                list_hidden_channels_la=hidden_channels_la,
                                num_layers=num_layers,
                                hetero_aggr=hetero_aggr,
                                mlp_channels=mlp_channels,
                                molecular_embedding_size=molecular_embedding_size,
                                dropout=dropout,
                                heads=heads)

        self.loss_funct = F.mse_loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.prefix = 'pocket_'

    def get_nb_parameters(self, only_trainable: bool = False):
        """Get the number of model's parameters

        Args:
            only_trainable (bool, optional): Consider only trainable
                parameters. Defaults to False.

        Returns:
            int: The number of parameters
        """
        return self.model.get_nb_parameters(only_trainable)

    def forward(self, data: pyg.data.HeteroData) -> torch.Tensor:
        """Forward

        Args:
            data (pyg.data.HeteroData): A Hetero Batch

        Returns:
            torch.Tensor: The scores
        """
        return self.model(data.x_dict, data.edge_index_dict,
                          data.edge_attr_dict, data.batch_dict)

    def _common_step(self, batch: pyg.data.HeteroData, batch_idx: int,
                     stage: str) -> Tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor]:
        """Common step for Train, Val and Test

        Args:
            batch (pyg.data.HeteroData): A Batch
            batch_idx (int): The batch idx
            stage (str): A string indicated the stage (train, val, casf13,
                casf13)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (loss,
                predicted score, real affinities)
        """
        batch_size = batch.y.size(0)
        y_pred = self(batch)
        loss = self.loss_funct(y_pred.view(-1), batch.y.float())
        self.log("step/{}_loss".format(stage), loss, batch_size=batch_size,
                 sync_dist=True)
        return loss, y_pred.view(-1), batch.y

    def training_step(self, batch: pyg.data.HeteroData,
                      batch_idx: int) -> Dict:
        """Training step

        Args:
            batch (pyg.data.HeteroData): A Batch
            batch_idx (int): The batch idx

        Returns:
            Dict: (loss, predicted score, real affinities)
        """
        loss, preds, targets = self._common_step(batch, batch_idx, 'train')
        return {'loss': loss, 'train_preds': preds.detach(),
                'train_targets': targets.detach()}

    def validation_step(self, batch: pyg.data.HeteroData,
                        batch_idx: int) -> Dict:
        """Validation step

        Args:
            batch (pyg.data.HeteroData): A Batch
            batch_idx (int): The batch idx

        Returns:
            Dict: (loss, predicted score, real affinities)
        """
        loss, preds, targets = self._common_step(batch, batch_idx, 'val')
        return {'val_loss': loss, 'val_preds': preds,
                'val_targets': targets}

    def test_step(self, batch: pyg.data.HeteroData, batch_idx: int) -> Dict:
        """Testing step

        Args:
            batch (pyg.data.HeteroData): A Batch
            batch_idx (int): The batch idx

        Returns:
            Dict: (loss, predicted score, real affinities, test_cluster,
                pdb_id)
        """
        loss, preds, targets = self._common_step(batch, batch_idx, 'test')
        return {'test_loss': loss, 'test_preds': preds,
                'test_targets': targets, 'test_cluster': batch.cluster,
                'test_pdb_id': batch.pdb_id}

    def common_epoch_end(self, outputs: List, stage: str):
        """Called after each epoch (except test). CASF metrics are computed here

        Args:
            outputs (List): Outputs produced by train and val steps
            stage (str):  A string indicated the stage (train, val, casf13,
                casf13)
        """
        loss_name = 'loss' if stage == 'train' else "{}_loss".format(stage)
        loss_batched = torch.stack([x[loss_name] for x in outputs])
        avg_loss = loss_batched.mean()
        all_preds = torch.concat([x["{}_preds".format(stage)]
                                  for x in outputs])
        all_targets = torch.concat([x["{}_targets".format(stage)]
                                    for x in outputs])
        r2 = tmf.r2_score(all_preds, all_targets)
        pearson = tmf.pearson_corrcoef(all_preds, all_targets)
        metrics_dict = {
            "ep_end_{}/loss".format(stage): avg_loss,
            "ep_end_{}/r2_score".format(stage): r2,
            "ep_end_{}/pearson".format(stage): pearson
        }
        self.log_dict(metrics_dict, sync_dist=True)

    def set_casf_test(self, version: Union[int, str]):
        """Reset all names and values used in model testing according to the given version

        Args:
            version (Union[int, str]): The CASF's version, must be 13 or 16
        """
        if isinstance(version, int):
            version = str(version)
        self.casf_name = 'casf_' + version
        self.reg_linear_name = 'reg_linear_' + version + '.png'
        self.output_csv = 'scores_' + version + '.csv'

        self.ranking_nb_in_clusters = 0
        if version == '13':
            self.ranking_nb_in_clusters = 3
        elif version == '16':
            self.ranking_nb_in_clusters = 5

    def test_epoch_end(self, outputs: List):
        """Called after each test epoch. CASF metrics are computed and
            plots created here

        Args:
            outputs (List): Outputs produced by test steps
        """
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        all_preds = torch.concat([x["test_preds"]
                                  for x in outputs])
        all_targets = torch.concat([x["test_targets"]
                                    for x in outputs])
        all_clusters = torch.concat([x["test_cluster"]
                                    for x in outputs])
        all_pdb_id = [x["test_pdb_id"] for x in outputs]
        self.metrics_on_test(avg_loss, all_preds,
                             all_targets, all_clusters, all_pdb_id)

    def metrics_on_test(self, avg_loss: torch.Tensor, all_preds: torch.Tensor,
                        all_targets: torch.Tensor, all_clusters: torch.Tensor,
                        all_pdb_id: List[str]):
        """Computes Scoring and Ranking powers, logs metrics and plots them

        Args:
            avg_loss (torch.Tensor): The average loss on the Test set
            all_preds (torch.Tensor): All predicted scores
            all_targets (torch.Tensor): All targets
            all_clusters (torch.Tensor): For each complex, the corresponding
                cluster
            all_pdb_id (List[str]): For each complex, the correspoding PDB id
        """
        pearson, sd, nb_favorable, mae, rmse = scoring_power(
            preds=all_preds, targets=all_targets)
        spearman, kendall, pi = ranking_power(
            all_preds, all_targets, all_clusters, self.ranking_nb_in_clusters)
        plotters.plot_linear_reg(all_preds, all_targets,
                                 pearson, sd,
                                 osp.join(self.plot_path,
                                          self.reg_linear_name))
        plotters.save_predictions(all_pdb_id, all_preds,
                                  osp.join(self.plot_path, self.output_csv))

        r2 = tmf.r2_score(all_preds, all_targets)
        pearson = float('nan') if pearson is None else pearson
        sd = float('nan') if sd is None else sd

        metrics_dict = {
            self.casf_name + "/loss": avg_loss,
            self.casf_name + "/r2_score": r2,
            self.casf_name + "/pearson": pearson,
            self.casf_name + "/sd": sd,
            self.casf_name + "/mae": mae,
            self.casf_name + "/rmse": rmse,
            self.casf_name + "/spearman": spearman,
            self.casf_name + "/kendall": kendall,
            self.casf_name + "/pi": pi,
            self.casf_name + "/nb_favorable": torch.tensor(nb_favorable, dtype=torch.float32)
        }
        self.log_dict(metrics_dict, sync_dist=True)

    def validation_epoch_end(self, outputs: List):
        self.common_epoch_end(outputs, 'val')

    def training_epoch_end(self, outputs: List):
        self.common_epoch_end(outputs, 'train')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        return {"optimizer": optimizer}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def predict(self, data: pyg.data.Batch) -> float:
        """Use the model to predict scores

        Args:
            data (pyg.data.Batch): Data

        Returns:
            float: The predicted scores
        """
        self.eval()
        with torch.no_grad():
            score = self(data)
        return score[0].item()
