import argparse
import os.path as osp
import sys

import torch_geometric as pyg

import bgcn_4_pls.model as md
from bgcn_4_pls.data import make_bipartite_graph, pocket_extraction


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
