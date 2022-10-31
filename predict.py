import argparse
import datetime
import os.path as osp
import sys

import numpy as np
import pandas as pd
import torch_geometric as pyg
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb

import model as md
from data import clean_pdb, create_pyg_graph


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

    df_grouped = ppdb_prot.df['ATOM'].groupby('residue_number')
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


def make_bipartite_graph(protein_path: str,
                         ligand_path: str,
                         atomic_distance_cutoff: float) -> pyg.data.HeteroData:
    """Build a Bipartite Graph from a Protein and a Ligand files

    Args:
        protein_path (str): Path to the protien PDB
        ligand_path (str): Path to the ligand MOL2
        atomic_distance_cutoff (float): The cutoff to consider a link between a protein-ligand atom pair

    Returns:
        pyg.data.HeteroData: The bipartite graph
    """
    protein_dir, protein_name = osp.split(protein_path)

    clean_protein_path = 'clean_' + protein_name
    clean_protein_path = osp.join(protein_dir, clean_protein_path)
    if not osp.exists(clean_protein_path):
        clean_pdb(protein_path, clean_protein_path)
    bipartite_graph = create_pyg_graph(clean_protein_path, ligand_path,
                                       cutoff=atomic_distance_cutoff)
    return bipartite_graph


def predict(protein_path: str,
            ligand_path: str,
            model_path: str,
            atomic_distance_cutoff: float,
            extract_pocket: bool,
            pocket_cutoff: float = 0.0) -> float:
    """Predict the affinity score for a given complex

    Args:
        protein_path (str): Path to the protein PDB file
        ligand_path (str): Path to the ligand MOL2/PDB file
        model_path (str): Path to the torch Checkpoint
        atomic_distance_cutoff (float): The cutoff to consider a link between a protein-ligand atom pair
        extract_pocket (bool): Extract the pocket according to the ligand's position,
            no necessary if the pocket is already provided by protein path
        pocket_cutoff (float): Cutoff for pocket extraction. Defaults to 0.0.

    Returns:
        float: The complex's score
    """
    if extract_pocket:
        prot_dir, prot_name = osp.split(protein_path)
        pocket_path = osp.join(prot_dir, "bgpls_pocket_{}".format(prot_name))
        if not osp.isfile(pocket_path):
            pocket_extraction(protein_path, ligand_path,
                              pocket_path, pocket_cutoff)
        protein_path = pocket_path

    if not osp.exists(protein_path):
        print('Error {} not found'.format(protein_path))
        sys.exit()
    if not osp.exists(ligand_path):
        print('Error {} not found'.format(ligand_path))

    bipartite_graph = make_bipartite_graph(protein_path,
                                           ligand_path,
                                           atomic_distance_cutoff=atomic_distance_cutoff)
    batch = pyg.data.Batch.from_data_list([bipartite_graph])

    model = md.Model.load_from_checkpoint(model_path)

    score = model.predict(batch)
    return round(score, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint_path', '-ckpt', type=str,
                        help='Path to the torch Checkpoint', required=True)
    parser.add_argument('-protein_path', '-p', type=str,
                        help='Path to the protein PDB file', required=True)
    parser.add_argument('-ligand_path', '-l', type=str,
                        help='Path to the ligand MOL2/PDB file', required=True)
    parser.add_argument('-cutoff', '-c',
                        type=float,
                        help='The cutoff to consider a link between a protein-ligand atom pair (Defaults to 4.0A)',
                        default=4.0)
    parser.add_argument('-extract_pocket',
                        action='store_true',
                        help="Extract the pocket according to the ligand's position,"
                        " no necessary if the pocket is already provided by protein path")
    parser.add_argument('-extract_pocket_cutoff',
                        type=float,
                        help="Cutoff for pocket extraction (Defaults to 10.0A)",
                        default=10.0)

    args = parser.parse_args()

    model_path = args.checkpoint_path
    protein_path = args.protein_path
    ligand_path = args.ligand_path
    atomic_distance_cutoff = args.cutoff
    pocket_cutoff = args.extract_pocket_cutoff
    extract_pocket = args.extract_pocket

    score = predict(protein_path, ligand_path, model_path,
                    atomic_distance_cutoff, extract_pocket, pocket_cutoff)

    print("Score = {}".format(round(score, 2)))
