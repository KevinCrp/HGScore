from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn import linear_model


def plot_linear_reg(p: torch.Tensor, t: torch.Tensor, pearson_r: float,
                    sd: float, filepath: str):
    """Plot predicted value versus real ones, and add linear regression

    Args:
        p (torch.Tensor): The tensor of predicted values
        t (torch.Tensor): The tensor of real values
        pearson_r (float): The Pearson Correlation coef
        sd (float): The standard deviation
        filepath (str): Path where save the plot
    """
    preds = p.cpu()
    targets = t.cpu()

    regr2 = linear_model.LinearRegression()
    regr2.fit(targets.numpy().reshape(-1, 1),
              preds.numpy().reshape(-1, 1))

    lineplot = sns.lineplot(x=targets.numpy(), y=targets.numpy(),
                            color="orange",
                            label="Perfect predictor")
    regplot = sns.regplot(x=targets.numpy(), y=preds.numpy(), ci=95,
                          label="Network output, linear regression, y = {}x + {}".format(round(regr2.coef_[0][0], 2),
                                                                                         round(regr2.intercept_[0], 2)))
    regplot.set(xlabel='Targeted scores',
                ylabel='Predicted scores')
    regplot.legend()
    fig_sns = regplot.get_figure()
    fig_sns.savefig(filepath, dpi=600)
    plt.close(fig_sns)


def save_predictions(pdb_id: List, preds: torch.Tensor, filepath: str):
    """Save predicted scores in a csv file, respecting the CASF format

    Args:
        pdb_id (List): List of pdb id
        preds (torch.Tensor): score
        filepath (str): Where save the csv
    """
    preds = np.round(preds.view(-1, 1).detach().cpu().numpy(), 2)
    pdb_id = np.array(pdb_id)
    t = np.concatenate((pdb_id, preds), axis=-1)
    df = pd.DataFrame(t, columns=['#code', 'score'])
    df.to_csv(filepath, index=False, header=True, sep=' ')


def plot_docking_power_curve(tops_label, tops, filepath: str):
    nb_top = len(tops_label)
    fig, ax = plt.subplots()
    ax.plot(tops_label, tops, color='blue', alpha=1.00)
    ax.set_xlabel('Top')
    ax.set_ylabel('Success rate (%)')
    ax.set_xlim((1, nb_top))
    ax.set_ylim((0, 110))
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.fill_between(tops_label, tops, 0, color='blue', alpha=.1)
    plt.savefig(filepath)
