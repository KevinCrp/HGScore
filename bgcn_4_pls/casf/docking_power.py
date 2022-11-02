import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def docking_power_df(docking_power_df: pd.DataFrame,
                     rmsd_cutoff: float,
                     plot_path: str) -> dict:
    """Compute CASF 2016 Docking power

    Args:
        docking_power_df (pd.DataFrame): A DF containing all scores and rmsd
            for all docking power decoys
        rmsd_cutoff (float): The RMSD cutoff (in angstrom) to define near-native docking pose
        plot_path (str): Path where docking power curve will be saved

    Returns:
        dict: A dictionnary containing SP[2-10] and TOP[1-3]
    """
    # Adapted from CASF 2016/Docking_power.py

    nb_top = 50 + 1
    top_df = []
    tops = []
    tops_label = []
    pdb_list = list(set(docking_power_df['pdb_id'].to_list()))
    for j in np.arange(1, nb_top):
        top_df += [pd.DataFrame(index=pdb_list, columns=['success'])]
    SP2 = pd.DataFrame(index=pdb_list, columns=['spearman'])
    SP3 = pd.DataFrame(index=pdb_list, columns=['spearman'])
    SP4 = pd.DataFrame(index=pdb_list, columns=['spearman'])
    SP5 = pd.DataFrame(index=pdb_list, columns=['spearman'])
    SP6 = pd.DataFrame(index=pdb_list, columns=['spearman'])
    SP7 = pd.DataFrame(index=pdb_list, columns=['spearman'])
    SP8 = pd.DataFrame(index=pdb_list, columns=['spearman'])
    SP9 = pd.DataFrame(index=pdb_list, columns=['spearman'])
    SP10 = pd.DataFrame(index=pdb_list, columns=['spearman'])
    docking_results_columns = ['code', 'Rank1', 'RMSD1', 'Rank2',
                               'RMSD2', 'Rank3', 'RMSD3']
    docking_results = pd.DataFrame(index=range(1, len(pdb_list) + 1),
                                   columns=docking_results_columns)

    tmp = 1
    for pdb in pdb_list:
        docking_power_df_pdb = docking_power_df.loc[docking_power_df['pdb_id'] == pdb]
        df_sorted = docking_power_df_pdb.sort_values(
            by=['score'], ascending=[False])
        docking_results.loc[tmp]['Rank1'] = ''.join(df_sorted[0:1]['#code'])
        docking_results.loc[tmp]['RMSD1'] = float(df_sorted[0:1]['rmsd'])
        docking_results.loc[tmp]['Rank2'] = ''.join(df_sorted[1:2]['#code'])
        docking_results.loc[tmp]['RMSD2'] = float(df_sorted[1:2]['rmsd'])
        docking_results.loc[tmp]['Rank3'] = ''.join(df_sorted[2:3]['#code'])
        docking_results.loc[tmp]['RMSD3'] = float(df_sorted[2:3]['rmsd'])
        docking_results.loc[tmp]['code'] = pdb
        tmp += 1
        for j in np.arange(1, nb_top):
            minrmsd = df_sorted[0:j]['rmsd'].min()
            top = top_df[j-1]
            if minrmsd <= rmsd_cutoff:
                top.loc[pdb]['success'] = 1
            else:
                top.loc[pdb]['success'] = 0
        for s in np.arange(2, 11):
            sptemp = docking_power_df_pdb[docking_power_df_pdb.rmsd <= s]
            varname2 = 'SP' + str(s)
            sp = locals()[varname2]
            if float(sptemp.shape[0]) >= 5:
                sp.loc[pdb]['spearman'] = np.negative(
                    sptemp.corr('spearman')['rmsd']['score'])
            else:
                continue

    SP2 = SP2.dropna(subset=['spearman'])
    SP3 = SP3.dropna(subset=['spearman'])
    SP4 = SP4.dropna(subset=['spearman'])
    SP5 = SP5.dropna(subset=['spearman'])
    SP6 = SP6.dropna(subset=['spearman'])
    SP7 = SP7.dropna(subset=['spearman'])
    SP8 = SP8.dropna(subset=['spearman'])
    SP9 = SP9.dropna(subset=['spearman'])
    SP10 = SP10.dropna(subset=['spearman'])

    for j in np.arange(1, nb_top):
        top = top_df[j-1]
        top_succes = float(top['success'].sum()) / float(top.shape[0]) * 100
        tops += [top_succes]
        tops_label += [j]

    sp2 = round(SP2['spearman'].mean(), 2)
    sp3 = round(SP3['spearman'].mean(), 2)
    sp4 = round(SP4['spearman'].mean(), 2)
    sp5 = round(SP5['spearman'].mean(), 2)
    sp6 = round(SP6['spearman'].mean(), 2)
    sp7 = round(SP7['spearman'].mean(), 2)
    sp8 = round(SP8['spearman'].mean(), 2)
    sp9 = round(SP9['spearman'].mean(), 2)
    sp10 = round(SP10['spearman'].mean(), 2)

    top1_correct = top_df[0]['success'].sum()
    top2_correct = top_df[1]['success'].sum()
    top3_correct = top_df[2]['success'].sum()

    res_dict = {"sp2": sp2,
                "sp3": sp3,
                "sp4": sp4,
                "sp5": sp5,
                "sp6": sp6,
                "sp7": sp7,
                "sp8": sp8,
                "sp9": sp9,
                "sp10": sp10,
                "top1_success": round(tops[0], 2),
                "top1_correct": top1_correct,
                "top2_success": round(tops[1], 2),
                "top2_correct": top2_correct,
                "top3_success": round(tops[2], 2),
                "top3_correct": top3_correct}

    fig, ax = plt.subplots()
    ax.plot(tops_label, tops, color='blue', alpha=1.00)
    ax.set_xlabel('Top')
    ax.set_ylabel('Success rate (%)')
    ax.set_xlim((1, nb_top))
    ax.set_ylim((0, 110))
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.fill_between(tops_label, tops, 0, color='blue', alpha=.1)
    plt.savefig(plot_path)

    return res_dict
