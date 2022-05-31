import argparse
import os.path as osp
import sys
import torch

import torch_geometric as pyg

import model as md
from to_graph import clean_pdb, create_pyg_graph


def make_bipartite_graph(protein_path: str, ligand_path: str) -> pyg.data.HeteroData:
    protein_dir, clean_protein_path = osp.split(protein_path)
    clean_protein_path = 'clean_'+clean_protein_path
    clean_protein_path = osp.join(protein_dir, clean_protein_path)
    if not osp.exists(clean_protein_path):
        clean_pdb(protein_path, clean_protein_path)
    bipartite_graph = create_pyg_graph(clean_protein_path, ligand_path)
    return bipartite_graph

def predict(model: torch.nn.Module, data: pyg.data.Batch) -> float:
    model.eval()
    with torch.no_grad():
        score = model(batch)
    return score[0].item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint_path', '-c', type=str,
                        help='Path to the torch Checkpoint', required=True)
    parser.add_argument('-protein_path', '-p', type=str,
                        help='Path to the protein PDB file', required=True)
    parser.add_argument('-ligand_path', '-l', type=str,
                        help='Path to the ligand MOL2 file', required=True)

    args = parser.parse_args()

    model_path = args.checkpoint_path
    protein_path = args.protein_path
    ligand_path = args.ligand_path

    if not osp.exists(protein_path):
        print('Error {} not found'.format(protein_path))
        sys.exit()
    if not osp.exists(ligand_path):
        print('Error {} not found'.format(ligand_path))

    bipartite_graph = make_bipartite_graph(protein_path, ligand_path)
    batch = pyg.data.Batch.from_data_list([bipartite_graph])

    model = md.Model.load_from_checkpoint(model_path)

    score = predict(model, batch)

    print("Score = {}".format(round(score, 2)))
    
