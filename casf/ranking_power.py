from typing import Tuple

import numpy as np
import pandas as pd
import torch


def cal_PI(df: pd.DataFrame) -> float:
    """Returns the CASF Predictive Index (PI) w

    Args:
        df (pd.DataFrame): A Dataframe containing columns score and logKa

    Returns:
        float: The PI
    """
    dfsorted = df.sort_values(['logKa'], ascending=True)
    W = []
    WC = []
    lst = list(dfsorted.index)
    for i in np.arange(0, df.shape[0]):
        xi = lst[i]
        score = float(dfsorted.loc[xi]['score'])
        bindaff = float(dfsorted.loc[xi]['logKa'])
        for j in np.arange(i+1, df.shape[0]):
            xj = lst[j]
            scoretemp = float(dfsorted.loc[xj]['score'])
            bindafftemp = float(dfsorted.loc[xj]['logKa'])
            w_ij = abs(bindaff-bindafftemp)
            W.append(w_ij)
            if score < scoretemp:
                WC.append(w_ij)
            elif score > scoretemp:
                WC.append(-w_ij)
            else:
                WC.append(0)
    pi = float(sum(WC))/float(sum(W))
    return pi


def ranking_power_pt(preds: torch.Tensor, target: torch.Tensor,
                     num_in_cluster: int, clusters: torch.Tensor) -> Tuple[float, float, float]:
    """Compute the CASF scoring power from PyTorch Tensors

    Args:
        preds (torch.Tensor): The predicted tensor
        target (torch.Tensor): The real tensor
        num_in_cluster (int): The number if items in each cluster (v13 : 3; v16 : 5)
        cluster (torch.Tensor): Cluster ID for ranking.

    Returns:
        Tuple[float, float, float]: Spearman’s rank correlation coefficient; Kendall’s rank correlation coefficient; Predictive Index
    """
    t = torch.cat((preds.view(-1, 1), target.view(-1, 1), clusters.view(-1, 1)), -1).cpu().numpy()
    df = pd.DataFrame(t, columns=['score', 'logKa','cluster'])
    return ranking_power(df, num_in_cluster)


def ranking_power(df: pd.DataFrame,
                  num_in_cluster: int) -> Tuple[float, float, float]:
    """Compute CASF Ranking power metrics:
     * Spearman’s rank correlation coefficient
     * Kendall’s rank correlation coefficient
     * Predictive Index

    Args:
        df (pd.DataFrame): A Dataframe containing columns score, logKa, and cluster
        num_in_cluster (int): The number if items in each cluster (v13 : 3; v16 : 5)

    Returns:
        Tuple[float, float, float]: Spearman’s rank correlation coefficient; Kendall’s rank correlation coefficient; Predictive Index
    """
    # Predicted scores are round because in code proposed by CASF authors, the metrics are computed on rounded scores
    df.score = df.score.round(2)
    nb_clusters = len(df) // num_in_cluster + 1
    cluster_dfs = df.groupby('cluster')
    spearman_sum = 0.0
    kendall_sum = 0.0
    pi_sum = 0.0
    for _, cluster_df in cluster_dfs:
        cluster_df_sorted = cluster_df.sort_values('score', ascending=False)
        spearman_coef = cluster_df_sorted.corr('spearman')['logKa']['score']
        kendall_coef = cluster_df_sorted.corr('kendall')['logKa']['score']
        pi = cal_PI(df=cluster_df_sorted)

        spearman_sum += spearman_coef
        kendall_sum += kendall_coef
        pi_sum += pi

    spearman_mean = spearman_sum/nb_clusters
    kendall_mean = kendall_sum/nb_clusters
    pi_mean = pi_sum/nb_clusters

    return spearman_mean, kendall_mean, pi_mean
