import torch
import torch_geometric as pyg
from biopandas.pdb import PandasPdb

import featurizer as f_atm


def clean_pdb(pdb_path: str, out_filename: str):
    """Remove HETATM in the given PDB file

    Args:
        pdb_path (str): The input pdb file
        out_filename (str): Path where save the cleaned file
    """
    # Remove HETATM
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_path)
    ppdb.to_pdb(path=out_filename,
                records=['ATOM'],
                gz=False,
                append_newline=True)


def create_pyg_graph(protein_path: str,
                     ligand_path: str,
                     target: float = None,
                     cluster: int = None,
                     pdb_id: str = None) -> pyg.data.HeteroData:
    """Create a torch_geometric HeteroGraph of a protein-ligand complex

    Args:
        protein_path (str): Path to the protein file (PDB)
        ligand_path (str): Path to the liagnd file (MOL2)
        target (float, optional): Affinity target. Defaults to None.
        cluster (int, optional): Cluster ID for ranking. Defaults to None.
        pdb_id (int, optional): PDB ID for casf output. Defaults to None.

    Returns:
        pyg.data.HeteroData: 
    """
    (protein_atm_x,
     ligand_atm_x,

     protein_atm_to_protein_atm_edge_index,
     ligand_atm_to_ligand_atm_edge_index,
     ligand_atm_to_protein_atm_edge_index,
     protein_atm_to_ligand_atm_edge_index,

     protein_atm_to_protein_atm_edge_attr,
     ligand_atm_to_ligand_atm_edge_attr,
     ligand_atm_to_protein_atm_edge_attr,
     protein_atm_to_ligand_atm_edge_attr
     ) = f_atm.featurize(protein_path, ligand_path)

    data = pyg.data.HeteroData()

    data['protein_atoms'].x = torch.tensor(protein_atm_x)
    data['ligand_atoms'].x = torch.tensor(ligand_atm_x)

    data['protein_atoms', 'linked_to', 'protein_atoms'].edge_index = torch.tensor(
        protein_atm_to_protein_atm_edge_index)
    data['ligand_atoms', 'linked_to', 'ligand_atoms'].edge_index = torch.tensor(
        ligand_atm_to_ligand_atm_edge_index)
    data['ligand_atoms', 'interact_with', 'protein_atoms'].edge_index = torch.tensor(
        ligand_atm_to_protein_atm_edge_index)
    data['protein_atoms', 'interact_with', 'ligand_atoms'].edge_index = torch.tensor(
        protein_atm_to_ligand_atm_edge_index)

    data['protein_atoms', 'linked_to', 'protein_atoms'].edge_attr = torch.tensor(
        protein_atm_to_protein_atm_edge_attr)
    data['ligand_atoms', 'linked_to', 'ligand_atoms'].edge_attr = torch.tensor(
        ligand_atm_to_ligand_atm_edge_attr)
    data['ligand_atoms', 'interact_with', 'protein_atoms'].edge_attr = torch.tensor(
        ligand_atm_to_protein_atm_edge_attr)
    data['protein_atoms', 'interact_with', 'ligand_atoms'].edge_attr = torch.tensor(
        protein_atm_to_ligand_atm_edge_attr)

    data.y = target
    data.cluster = cluster
    data.pdb_id = pdb_id

    return data
