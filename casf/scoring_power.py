from math import nan
import math
from typing import Tuple

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


def tensors_to_df(preds: torch.Tensor, targets: torch.Tensor) -> pd.DataFrame:
    """Save preds and targets tensor into a Dataframe to compute scoring power

    Args:
        preds (torch.Tensor): The predicted tensor
        targets (torch.Tensor): The real tensor

    Returns:
        pd.DataFrame: A Dataframe containing ['score', 'logKa'] columns (logKa == targets)
    """
    t = torch.cat((preds.view(-1, 1), targets.view(-1, 1)), -1).cpu().numpy()
    df = pd.DataFrame(t, columns=['score', 'logKa'])
    return df


def scoring_power(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float, float, float]:
    """Compute the CASF scoring power, adapted from the scoring_power 2016 code

    Args:
        preds (torch.Tensor): The predicted tensor
        targets (torch.Tensor): The real tensor

    Returns:
        Tuple[float, float, float, float, float]: Pearson Correlation coef, standard deviation, number of favorable items,
                                    Mean Absolute error, root mean squared error
    """
    df = tensors_to_df(preds, targets)
    # Predicted scores are round because in code proposed by CASF authors, the metrics are computed on rounded scores
    df.score = df.score.round(2)
    # From CASF Code
    df2 = df[df.score > 0]
    if df2.shape[0] < 2:
        return nan, nan, df2.shape[0], nan, nan
    nb_favorable_sample = (df.score > 0).sum()
    if nb_favorable_sample == 0:
        return None, None, nb_favorable_sample, nan, nan
    regr = linear_model.LinearRegression()
    regr.fit(df2[['score']], df2[['logKa']])
    testpredy = regr.predict(df2[['score']])
    pearsonr = scipy.stats.pearsonr(
        df2['logKa'].values, df2['score'].values)[0]
    testmse = mean_squared_error(df2[['logKa']], testpredy)
    mae = mean_absolute_error(df2[['logKa']], testpredy)
    rmse = math.sqrt(testmse)
    num = df2.shape[0]
    sd = np.sqrt((testmse * num) / (num - 1))
    return pearsonr, sd, float(nb_favorable_sample), mae, rmse
