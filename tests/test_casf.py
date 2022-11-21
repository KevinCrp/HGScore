import numpy as np
import torch
import pandas as pd
from bgcn_4_pls.casf.ranking_power import ranking_power
from bgcn_4_pls.casf.scoring_power import scoring_power
from bgcn_4_pls.casf.docking_power import docking_power_df

def test_scoring_power():
    preds = torch.Tensor([2.45, 1.67, 3.90, 6.35])
    targets = torch.Tensor([2.40, 3.09, 2.87, 7.9])

    rp, sd, nb_fav, mae, rmse = scoring_power(preds=preds, targets=targets)
    assert round(rp, 2) == 0.88
    assert round(sd, 2) == 1.22
    assert int(nb_fav) == 4
    assert round(mae, 2) == np.float32(0.97)
    assert round(rmse, 2) == 1.06


def test_ranking_power():
    preds = torch.Tensor([2.45, 1.67, 3.90, 6.35])
    targets = torch.Tensor([2.40, 0.78, 2.87, 7.9])
    clusters = torch.Tensor([1, 1, 2, 2])
    num_in_cluster = 2
    (spearman_mean,
     kendall_mean,
     pi_mean) = ranking_power(preds=preds,
                              targets=targets,
                              clusters=clusters,
                              num_in_cluster=num_in_cluster)
    assert round(spearman_mean, 2) == 0.67
    assert round(kendall_mean, 2) == 0.67
    assert round(pi_mean, 2) == 0.67


def test_docking_power():
    columns = ['pdb_id', '#code', 'rmsd', 'score']
    data = [['4llx', '4llx_ligand', 0.00, 4.212035],
            ['4llx', '4llx_208', 1.53, 3.623964],
            ['4llx', '4llx_213', 0.45, 4.090487],
            ['4llx', '4llx_236', 0.43, 3.929231],
            ['4llx', '4llx_243', 9.96, 3.269064],
            ['test', 'test_877', 0.00, 3.231743],
            ['test', 'test_882', 6.50, 3.561750],
            ['test', 'test_943', 0.98, 3.553943],
            ['test', 'test_967', 2.94, 3.422419],
            ['test', 'test_977', 5.55, 3.424685]]

    df = pd.DataFrame(data=data, columns=columns)
    res_dict, tops_label, tops = docking_power_df(df, rmsd_cutoff=2.0)

    assert res_dict['top3_correct'] == 2
    assert res_dict['sp10'] == 0.1