import argparse
import multiprocessing as mp
import os
import os.path as osp
import sys

import pandas as pd
import torch_geometric as pyg
from tqdm import tqdm

import hgcn_4_pls.model as md
from hgcn_4_pls.data import process_graph_from_files
from hgcn_4_pls.utilities.pockets import pocket_extraction


def predict(protein_path: str,
            ligand_path: str,
            model: md.Model,
            atomic_distance_cutoff: float,
            extract_pocket: bool,
            pocket_cutoff: float = 0.0) -> float:
    """Predict the affinity score for a given complex

    Args:
        protein_path (str): Path to the protein PDB file
        ligand_path (str): Path to the ligand MOL2/PDB file
        model_path (md.Model): The loaded model
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

    het_graph = process_graph_from_files(protein_path,
                                         ligand_path,
                                         atomic_distance_cutoff=atomic_distance_cutoff)
    batch = pyg.data.Batch.from_data_list([het_graph])

    score = model.predict(batch)
    return round(score, 2), protein_path, ligand_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint_path', '-ckpt', type=str,
                        help='Path to the torch Checkpoint', required=True)
    parser.add_argument('-complex_list_path', '-list', type=str,
                        help='Path to the csv file containing "protein_path,ligand_path" on each line', required=True)
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
    complex_list_path = args.complex_list_path
    atomic_distance_cutoff = args.cutoff
    pocket_cutoff = args.extract_pocket_cutoff
    extract_pocket = args.extract_pocket

    model = md.Model.load_from_checkpoint(model_path)

    df_dataset = pd.read_csv(complex_list_path)
    dir_res = osp.join(osp.split(complex_list_path)[0], 'results')
    if not osp.isdir(dir_res):
        os.mkdir(dir_res)

    nb_complexes = df_dataset.shape[0]
    dict_res = {}
    pool_args = []
    for i in range(nb_complexes):
        protein_path = df_dataset.iloc[i]['protein_path']
        ligand_path = df_dataset.iloc[i]['ligand_path']
        pool_args += [(protein_path, ligand_path, model,
                      atomic_distance_cutoff, extract_pocket, pocket_cutoff)]
    pool = mp.Pool(mp.cpu_count())
    data_list = pool.starmap(predict, pool_args, chunksize=100)

    for data in data_list:
        score = data[0]
        protein_path = data[1]
        ligand_path = data[2]
        if protein_path not in dict_res.keys():
            dict_res[protein_path] = pd.DataFrame(
                columns=['ligand_path', 'score'])
        dict_res[protein_path] = dict_res[protein_path].append({'ligand_path': ligand_path,
                                                                'score': round(score, 2)},
                                                               ignore_index=True)

    for protein_path in dict_res.keys():
        protein = osp.splitext(osp.split(protein_path)[1])[0]
        print(protein)
        print(dict_res[protein_path])
        results_file_path = osp.join(dir_res, protein+'.csv')
        print(results_file_path)
        dict_res[protein_path].to_csv(results_file_path, index=False)
