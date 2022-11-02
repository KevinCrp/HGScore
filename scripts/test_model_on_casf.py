import argparse
import multiprocessing as mp
import os.path as osp
from typing import Tuple

import pandas as pd
import torch
import torch_geometric as pyg
import tqdm

import bgcn_4_pls.config as cfg
import bgcn_4_pls.data as data
import bgcn_4_pls.model as md
import bgcn_4_pls.plotters as plotters
from bgcn_4_pls.casf.docking_power import docking_power_df
from bgcn_4_pls.casf.ranking_power import ranking_power
from bgcn_4_pls.casf.scoring_power import scoring_power


def predict_on_CASF(model: torch.nn.Module, dataloader: pyg.loader.DataLoader,
                    nb_in_cluster_ranking: int, csv_score_path: str,
                    do_plot: bool = False,
                    plot_path: str = 'plot.png') -> Tuple[float,
                                                          float, float, float,
                                                          float, float, float,
                                                          float]:
    """Predict score on the CASF (PDBBind core) set and returns scoring and
        ranking powers metrics

    Args:
        model (torch.nn.Module): The model
        dataloader (pyg.loader.DataLoader): A dataloader
        nb_in_cluster_ranking (int): The number if items in each
            cluster (v13 : 3; v16 : 5)
        csv_score_path (str): Path where save csv scores
        do_plot (bool, optional): Do plot. Defaults to False.
        plot_path (str, optional): Where save plot. Defaults to 'plot.png'.

    Returns:
        Tuple[float, float, float, float, float, float, float, float]: rp,
            sd, nb, mae, rmse, sp, ke, pi
    """
    preds = []
    targets = []
    clusters = []
    pdb_id = []
    for d in tqdm.tqdm(dataloader):
        preds += [model.predict(d)]
        targets += [d.y]
        clusters += [d.cluster]
        pdb_id += [d.pdb_id]
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    clusters = torch.tensor(clusters)

    rp, sd, nb, mae, rmse = scoring_power(preds=preds, targets=targets)
    sp, ke, pi = ranking_power(
        preds, targets, clusters, nb_in_cluster_ranking)
    plotters.save_predictions(pdb_id, preds, csv_score_path)
    if do_plot:
        plotters.plot_linear_reg(preds, targets, rp, sd, plot_path)
    return rp, sd, nb, mae, rmse, sp, ke, pi


def docking_power(model: torch.nn.Module,
                  dataloader: pyg.loader.DataLoader,
                  rmsd_cutoff: float,
                  plot_path: str):
    preds = []
    rmsd = []
    pdb_id = []
    decoy_id = []
    for d in tqdm.tqdm(dataloader):
        preds += [model.predict(d)]
        rmsd += [d.rmsd.item()]
        pdb_id += d.pdb_id
        decoy_id += d.decoy_id
    zipped_list = list(zip(pdb_id, decoy_id, rmsd, preds))
    df = pd.DataFrame(zipped_list,
                      columns=['pdb_id', '#code', 'rmsd', 'score'])
    return docking_power_df(docking_power_df=df,
                            rmsd_cutoff=rmsd_cutoff,
                            plot_path=plot_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint_path', '-ckpt', type=str,
                        help='Path to the torch Checkpoint', required=True)
    parser.add_argument('-plot', '-p', action='store_true',
                        help='Do plot')
    parser.add_argument('-casf_13', action='store_true',
                        help='Test on CASF Core set v2013')
    parser.add_argument('-casf_16', action='store_true',
                        help='Test on CASF Core set v2016')
    parser.add_argument('-cutoff', '-c',
                        type=float,
                        help='The cutoff to consider a link between a protein-ligand atom pair',
                        default=4.0)
    parser.add_argument('-docking_power', '-dp',
                        action='store_true',
                        help='Flag allowing to compute the docking power')
    parser.add_argument('-docking_power_cutoff', '-dpc',
                        type=float,
                        default=2.0,
                        help='The RMSD cutoff (in angstrom) to define near-native docking pose for Docking Power (defaults to 2.0)')
    parser.add_argument('-pocket',
                        action='store_true',
                        help='Flag allowing to consider only the binding pocket as defined by PDBBind')
    parser.add_argument('-data', '-d',
                        type=str,
                        required=True,
                        help='Path to the data directory')
    args = parser.parse_args()

    model_path = args.checkpoint_path
    atomic_distance_cutoff = args.cutoff
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
        dt_casf13 = data.CASFDataset(root=args.data, year='13',
                                     atomic_distance_cutoff=atomic_distance_cutoff,
                                     only_pocket=args.pocket)
        dl_casf13 = pyg.loader.DataLoader(dt_casf13,
                                          batch_size=1,
                                          num_workers=mp.cpu_count())
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
        dt_casf16 = data.CASFDataset(root=args.data, year='16',
                                     atomic_distance_cutoff=atomic_distance_cutoff,
                                     only_pocket=args.pocket)
        dl_casf16 = pyg.loader.DataLoader(dt_casf16,
                                          batch_size=1,
                                          num_workers=mp.cpu_count())
        rp, sd, nb, mae, rmse, sp, ke, pi = predict_on_CASF(
            model=model, dataloader=dl_casf16, nb_in_cluster_ranking=5,
            csv_score_path=csv_16_path,
            do_plot=args.plot, plot_path=plot_16_path)

        if args.docking_power:
            print("\tDocking Power (CASF 2016)")
            dt_dp = data.DockingPower_Dataset(root=args.data,
                                              year='16',
                                              atomic_distance_cutoff=atomic_distance_cutoff,
                                              only_pocket=args.pocket)
            dl_docking_power = pyg.loader.DataLoader(dt_dp,
                                                     batch_size=1,
                                                     num_workers=mp.cpu_count())

            docking_power_plot_path = osp.join(
                plot_path[0], plot_path[1].replace('.', '_') + '_docking_power_plot.png')
            docking_power_res_dict = docking_power(model=model,
                                                   dataloader=dl_docking_power,
                                                   rmsd_cutoff=args.docking_power_cutoff,
                                                   plot_path=docking_power_plot_path)

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

        if args.docking_power:
            print("\tDocking Power:")
            print("\t\tSp2 {}".format(docking_power_res_dict['sp2']))
            print("\t\tSp3 {}".format(docking_power_res_dict['sp3']))
            print("\t\tSp4 {}".format(docking_power_res_dict['sp4']))
            print("\t\tSp5 {}".format(docking_power_res_dict['sp5']))
            print("\t\tSp6 {}".format(docking_power_res_dict['sp6']))
            print("\t\tSp7 {}".format(docking_power_res_dict['sp7']))
            print("\t\tSp8 {}".format(docking_power_res_dict['sp8']))
            print("\t\tSp9 {}".format(docking_power_res_dict['sp9']))
            print("\t\tSp10 {}".format(docking_power_res_dict['sp10']))
            print("\t\tTOP1")
            print("\t\t\tSuccess {} ; Success Rate {}".format(docking_power_res_dict['top1_correct'],
                                                              docking_power_res_dict['top1_success']))
            print("\t\tTOP2")
            print("\t\t\tSuccess {} ; Success Rate {}".format(docking_power_res_dict['top2_correct'],
                                                              docking_power_res_dict['top2_success']))
            print("\t\tTOP3")
            print("\t\t\tSuccess {} ; Success Rate {}".format(docking_power_res_dict['top3_correct'],
                                                              docking_power_res_dict['top3_success']))
