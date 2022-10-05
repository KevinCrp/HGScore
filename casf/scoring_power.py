from math import nan
import math
from typing import  Tuple

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error


def scoring_power(**kwargs) -> Tuple[float, float, float, float, float]:
    """Compute the CASF scoring power, from two tensor (preds and targets)
        or from a list including both preds and targets

    Returns:
        Tuple[float, float, float, float, float]: Pearson Correlation coef, standard deviation, number of favorable items,
                                    Mean Absolute error, root mean squared error
    """    
    preds = kwargs.get('preds', None)
    targets = kwargs.get('targets', None)
    preds_target_lists = kwargs.get('preds_target_list', None)

    if preds is not None and targets is not None:
        t = torch.cat((preds.view(-1, 1), targets.view(-1, 1)), -1).cpu().numpy()
        df = pd.DataFrame(t, columns=['score', 'logKa'])
    elif preds_target_lists is not None:
        df = pd.DataFrame(preds_target_lists, columns=['score', 'logKa'])
    else:
        return nan
    return scoring_power_compute(df) 


def scoring_power_compute(df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    """Compute the CASF scoring power from a pandas Dataframe, adapted from the scoring_power 2016 code

    Args:
        df (pd.DataFrame): A pandas Dataframe with columns logKa (real values) and score (predictions)

    Returns:
        Tuple[float, float, float, float, float]: Pearson Correlation coef, standard deviation, number of favorable items,
                                    Mean Absolute error, root mean squared error
    """
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
