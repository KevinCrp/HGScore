import argparse
import os.path as osp
import sys

import torch_geometric as pyg

import model as md
from data import clean_pdb, create_pyg_graph


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint_path', '-ckpt', type=str,
                        help='Path to the torch Checkpoint', required=True)
    parser.add_argument('-protein_path', '-p', type=str,
                        help='Path to the protein PDB file', required=True)
    parser.add_argument('-ligand_path', '-l', type=str,
                        help='Path to the ligand MOL2 file', required=True)
    parser.add_argument('-cutoff', '-c',
                        type=float,
                        help='The cutoff to consider a link between a protein-ligand atom pair',
                        default=4.0)

    args = parser.parse_args()

    model_path = args.checkpoint_path
    protein_path = args.protein_path
    ligand_path = args.ligand_path
    atomic_distance_cutoff = args.cutoff

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

    print("Score = {}".format(round(score, 2)))
