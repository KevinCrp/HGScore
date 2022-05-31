import argparse
import os.path as osp
from typing import Tuple

import torch
import torch_geometric as pyg

import config as cfg
import data
import model as md
import plotters
from casf.ranking_power import ranking_power_pt
from casf.scoring_power import scoring_power_pt


def predict_on_CASF(model: torch.nn.Module, dataloader: pyg.loader.DataLoader,
                    nb_in_cluster_ranking: int, csv_score_path: str,
                    do_plot: bool = False,
                    plot_path: str = 'plot.png') -> Tuple[float,
                                                          float, float, float,
                                                          float, float, float,
                                                          float]:
    """Predict score on the CASF (PDBBind core) set and returns scoring and ranking powers metrics

    Args:
        model (torch.nn.Module): The model
        dataloader (pyg.loader.DataLoader): A dataloader
        nb_in_cluster_ranking (int): The number if items in each cluster (v13 : 3; v16 : 5)
        csv_score_path (str): Path where save csv scores
        do_plot (bool, optional): Do plot. Defaults to False.
        plot_path (str, optional): Where save plot. Defaults to 'plot.png'.

    Returns:
        Tuple[float, float, float, float, float, float, float, float]: rp, sd, nb, mae, rmse, sp, ke, pi
    """
    model.eval()
    target = []
    pred = []
    cluster = []
    pdb_id = []
    for data in dataloader:
        y_pred = model(data)
        target += [data.y]
        pred += [y_pred]
        cluster += [data.cluster]
        pdb_id += [data.pdb_id]
    pred = torch.tensor(pred)
    target = torch.tensor(target)
    cluster = torch.tensor(cluster)
    rp, sd, nb, mae, rmse = scoring_power_pt(pred, target)
    sp, ke, pi = ranking_power_pt(pred, target, nb_in_cluster_ranking, cluster)
    plotters.save_predictions(pdb_id, pred, csv_score_path)
    if do_plot:
        plotters.plot_linear_reg(pred, target, rp, sd, plot_path)
    return rp, sd, nb, mae, rmse, sp, ke, pi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint_path', '-c', type=str,
                        help='Path to the torch Checkpoint', required=True)
    parser.add_argument('-plot', '-p', action='store_true',
                        help='Do plot')
    parser.add_argument('-casf_13', action='store_true',
                        help='Test on CASF Core set v2013')
    parser.add_argument('-casf_16', action='store_true',
                        help='Test on CASF Core set v2016')

    args = parser.parse_args()

    model_path = args.checkpoint_path

    plot_path = osp.split(model_path)
    plot_13_path = osp.join(
        plot_path[0], plot_path[1].replace('.', '_') + '_plot_13.png')
    csv_13_path = osp.join(
        plot_path[0], plot_path[1].replace('.', '_') + '_score_13.csv')
    plot_16_path = osp.join(
        plot_path[0], plot_path[1].replace('.', '_') + '_plot_16.png')
    csv_16_path = osp.join(
        plot_path[0], plot_path[1].replace('.', '_') + '_score_16.csv')

    model = md.Model.load_from_checkpoint(model_path)

    if args.casf_13:
        print("CASF 2013 Testing ...")
        dt_casf13 = data.CASFDataset(root=cfg.data_path, year='13',
                                     only_pocket=cfg.data_use_only_pocket)
        dl_casf13 = pyg.loader.DataLoader(dt_casf13,
                                          batch_size=1,
                                          num_workers=cfg.datamodule_num_worker)
        rp, sd, nb, mae, rmse, sp, ke, pi = predict_on_CASF(
            model=model, dataloader=dl_casf13, nb_in_cluster_ranking=3,
            csv_score_path=csv_13_path,
            do_plot=args.plot, plot_path=plot_13_path)
        print("\tScoring Power:")
        print("\t\tRp {}".format(round(rp, 2)))
        print("\t\tSD {}".format(round(sd, 2)))
        print("\t\tNb Favorable {}".format(nb))
        print("\t\tMAE {}".format(round(mae, 2)))
        print("\t\tRMSE {}".format(round(rmse, 2)))
        print("\tRanking Power:")
        print("\t\tRho {}".format(round(sp, 2)))
        print("\t\tTau {}".format(round(ke, 2)))
        print("\t\tPI {}".format(round(pi, 2)))

    if args.casf_16:
        print("CASF 2016 Testing ...")
        dt_casf16 = data.CASFDataset(root=cfg.data_path, year='16',
                                     only_pocket=cfg.data_use_only_pocket)
        dl_casf16 = pyg.loader.DataLoader(dt_casf16,
                                          batch_size=1,
                                          num_workers=cfg.datamodule_num_worker)
        rp, sd, nb, mae, rmse, sp, ke, pi = predict_on_CASF(
            model=model, dataloader=dl_casf16, nb_in_cluster_ranking=5,
            csv_score_path=csv_16_path,
            do_plot=args.plot, plot_path=plot_16_path)

        print("\tScoring Power:")
        print("\t\tRp {}".format(round(rp, 2)))
        print("\t\tSD {}".format(round(sd, 2)))
        print("\t\tNb Favorable {}".format(nb))
        print("\t\tMAE {}".format(round(mae, 2)))
        print("\t\tRMSE {}".format(round(rmse, 2)))
        print("\tRanking Power:")
        print("\t\tRho {}".format(round(sp, 2)))
        print("\t\tTau {}".format(round(ke, 2)))
        print("\t\tPI {}".format(round(pi, 2)))
