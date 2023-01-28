import datetime
import os.path as osp

import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb


def residue_close_to_ligand(ligand_coords: np.ndarray,
                            res_coords: np.ndarray,
                            cutoff: float) -> bool:
    """Check if a protein's residue is close to a ligand according to the coordinates.
        Only heavy atoms are considered


    Args:
        ligand_coords (np.ndarray): The ligand's coordinates
        res_coords (np.ndarray): The protein's coordinates
        cutoff (float): Cutoff to consider a residue as close to the ligand

    Returns:
        bool: Is the residue close to the ligand
    """
    for res_coord in res_coords:
        for lig_coord in ligand_coords:
            distance = np.linalg.norm(res_coord - lig_coord)
            if distance <= cutoff:
                return True
    return False


def pocket_extraction(prot_path: str,
                      lig_path: str,
                      pocket_out_path: str,
                      cutoff: float):
    """Extract the protein binding pocket

    Args:
        prot_path (str): Path to the protein PDB
        lig_path (str): Path to the ligand (PDB or MOL2)
        pocket_out_path (str): Path where the extracted pocket will be saved
        cutoff (float): Cutoff to consider a residue as close to the ligand
    """
    ppdb_prot = PandasPdb()
    ppdb_prot.read_pdb(prot_path)

    ppdb_prot.df['ATOM'] = ppdb_prot.df['ATOM'][ppdb_prot.df['ATOM']
                                                ['element_symbol'] != 'H']

    ligand_filetype = osp.splitext(lig_path)[1].replace('.', '')
    if ligand_filetype.lower() == 'mol2':
        pmol2_lig = PandasMol2()
        pmol2_lig.read_mol2(lig_path)
        df_atom_lig = pmol2_lig.df[pmol2_lig.df['atom_type'] != 'H']
        ligand_coords = df_atom_lig[[
            'x', 'y', 'z']].to_numpy()
    elif ligand_filetype.lower() == 'pdb':
        ppdb_lig = PandasPdb()
        ppdb_lig.read_pdb(lig_path)
        df_atom_lig = ppdb_lig.df['ATOM'][ppdb_lig.df['ATOM']
                                          ['element_symbol'] != 'H']
        ligand_coords = df_atom_lig[[
            'x_coord', 'y_coord', 'z_coord']].to_numpy()

    df_grouped = ppdb_prot.df['ATOM'].groupby(['residue_number', 'chain_id'])
    list_df_in_site = []
    for _, group in df_grouped:
        res_coords = group[['x_coord', 'y_coord', 'z_coord']].to_numpy()
        if residue_close_to_ligand(ligand_coords, res_coords, cutoff=cutoff):
            list_df_in_site += [group]

    df_site = pd.concat(list_df_in_site).reset_index(drop=True)
    df_site['atom_number'] = [i+1 for i in range(df_site.shape[0])]

    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    ppdb_prot.df['OTHERS'].loc[0] = [
        'REMARK', '    Extracted by K.CRAMPON on {}'.format(now_str), 0]

    ppdb_prot.df['ATOM'] = df_site
    ppdb_prot.to_pdb(path=pocket_out_path,
                     records=['OTHERS', 'ATOM'],
                     append_newline=True)
