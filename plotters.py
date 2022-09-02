from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    gt_regr = linear_model.LinearRegression()
    gt_regr.fit(targets.numpy().reshape(-1, 1),
                targets.numpy().reshape(-1, 1))
    gt_pred_rl = gt_regr.predict(targets.numpy().reshape(-1, 1))

    regr2 = linear_model.LinearRegression()
    regr2.fit(targets.numpy().reshape(-1, 1),
              preds.numpy().reshape(-1, 1))
    regr2_pred_rl = regr2.predict(targets.numpy().reshape(-1, 1))

    plt.scatter(targets, preds, c='blue', label='Network outputs')
    plt.plot(targets, gt_pred_rl, c='orange',
             label="Ground Truth Linear Regression")
    plt.plot(targets.numpy(), regr2_pred_rl, c='red',
             label="Linear Regression")

    plt.xlabel('Targeted Scores')
    plt.ylabel('Predicted Scores')
    plt.legend()
    plt.savefig(filepath)
    plt.close()


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
