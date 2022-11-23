import torch

from hgcn_4_pls.plotters import (plot_docking_power_curve, plot_linear_reg,
                                 save_predictions)

PATH_PLOT_TEST = "tests/plot_test.png"
PATH_FILE_TEST = "tests/scores_test.csv"


def test_plot_linear_reg():
    preds = torch.Tensor([2.45, 1.67, 3.90, 6.35])
    targets = torch.Tensor([2.40, 3.09, 2.87, 7.9])
    plot_linear_reg(preds, targets, 1.2, 1.3, PATH_PLOT_TEST)


def test_save_predictions():
    pdb_id = [['pdb1'], ['pdb2'], ['pdb3'], ['pdb4']]
    preds = torch.Tensor([2.45, 1.67, 3.90, 6.35])
    save_predictions(pdb_id, preds, PATH_FILE_TEST)


def test_plot_docking_power_curve():
    tops_label = [1, 2, 3, 4]
    tops = [10.0, 15.0, 45.0, 67.0]
    plot_docking_power_curve(tops_label, tops, PATH_PLOT_TEST)
