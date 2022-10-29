import os.path as osp
from typing import List, Tuple

import biopandas.pdb as bpdb
import numpy as np
import oddt.interactions as interactions
from oddt.spatial import distance
from oddt.toolkits.ob import Molecule, readfile
from openbabel import openbabel

from redirect import stderr_redirected


def open_pdb(filepath: str, hydrogens_removal: bool = True) -> Molecule:
    """Open and load a molecule from a PDB file.

    Args:
        filepath (str): The PDB file path
        hydrogens_removal (bool, optional): To remove hydrogen atoms. Defaults to True.

    Returns:
        Molecule: The loaded molecule
    """
    with stderr_redirected(to='obabel.err'):
        mol = next(readfile('pdb', filepath))
        if hydrogens_removal:
            mol.removeh()
        return mol


def open_mol2(filepath: str, hydrogens_removal: bool = True) -> Molecule:
    """Open and load a molecule from a MOL2 file.

    Args:
        filepath (str): The MOL2 file path
        hydrogens_removal (bool, optional): To remove hydrogen atoms. Defaults to True.

    Returns:
        Molecule: The loaded molecule
    """
    with stderr_redirected(to='obabel.err'):
        mol = next(readfile('mol2', filepath))
        if hydrogens_removal:
            mol.removeh()
        return mol


def atom_type_one_hot(atomic_num: int) -> List[int]:
    """Returns the one-hot encoded atom type [B, C, N, O, F, P, S, Others]

    Args:
        atomic_num (int): The atom's atomic number

    Returns:
        List[int]: The one-hot encoded type
    """
    one_hot = 8 * [0]
    used_atom_num = [5, 6, 7, 8, 9, 15, 16]  # B, C, N, O, F, P, S, and Others
    d_atm_num = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6}
    if atomic_num in used_atom_num:
        one_hot[d_atm_num[atomic_num]] = 1
    return one_hot


def atom_hybridization_one_hot(hybridization: int) -> List[int]:
    """"Returns the one-hot encoded atom hybridization [other, sp, sp2, sp3, sq. planar, trig. bipy, octahedral]

    Args:
        hybridization (int): Hybridization

    Returns:
        List[int]: The one-hot encoded hybridization
    """
    onehot_hybridization = 7 * [0]
    if hybridization not in [1, 2, 3, 4, 5, 6]:
        hybridization = 0
    onehot_hybridization[hybridization] = 1
    return onehot_hybridization


def atom_degree_one_hot(degree: int) -> List[int]:
    """"Returns the one-hot encoded atom heavy/hetero [0, 1, 2, 3, 4, 5, 6+]

    Args:
        degree (int): The Hetero/Heavy degree

    Returns:
        List[int]: The one-hot encoded degree
    """
    oh_degree = 7 * [0]
    if degree > 6:
        oh_degree[6] = 1
    else:
        oh_degree[degree] = 1
    return oh_degree


def get_bond_properties(bond: openbabel.OBBond) -> List:
    """Returns bond properties:
     * The bond length
     * Is in an aromatic ring
     * Is in a ring
     * Is simple
     * Is double
     * Is triple


    Args:
        bond (openbabel.OBBond): An Openbabel bond

    Returns:
        List: The properties list
    """
    order = bond.GetBondOrder()
    length = bond.GetLength()
    aromatic = bond.IsAromatic()
    ring = bond.IsInRing()
    return [length, aromatic, ring, order == 1, order == 2, order == 3]


def get_molecule_properties(mol: Molecule) -> Tuple[List, List, List, List]:
    """Computes all atomic and bond properties of a molecule

    Args:
        pybel_mol (Molecule): A ODDT Molecule

    Returns:
        Tuple[List, List, List]: (The atoms' properties,
            the edge index, the edge attr)
    """
    # oh_ means one-hot encoded
    oh_atom_type = np.array(
        list(map(atom_type_one_hot, mol.atom_dict['atomicnum'].tolist())))
    oh_hybridization = np.array(
        list(map(atom_hybridization_one_hot, mol.atom_dict['hybridization'].tolist())))
    partial_charge = mol.atom_dict['charge'].reshape(-1, 1)
    hydrophobic = mol.atom_dict['ishydrophobe'].reshape(-1, 1)
    isaromatic = mol.atom_dict['isaromatic'].reshape(-1, 1)
    # is atom H-bond acceptor
    isacceptor = mol.atom_dict['isacceptor'].reshape(-1, 1)
    isdonor = mol.atom_dict['isdonor'].reshape(-1, 1)  # is atom H-bond donor

    # is atom H-bond donor Hydrogen
    isdonorh = mol.atom_dict['isdonorh'].reshape(-1, 1)
    isminus = mol.atom_dict['isminus'].reshape(-1, 1)
    isplus = mol.atom_dict['isplus'].reshape(-1, 1)

    atom_properties_list = np.concatenate((oh_atom_type, oh_hybridization,
                                           partial_charge, hydrophobic,
                                           isaromatic, isacceptor, isdonor,
                                           isdonorh, isminus, isplus), axis=1).tolist()
    edge_index = [[], []]
    edge_attr = []
    for bond in mol.bonds:
        ob_bond = bond.OBBond
        begin_id = ob_bond.GetBeginAtom().GetIdx() - 1
        end_id = ob_bond.GetEndAtom().GetIdx() - 1
        edge_index[0] += [begin_id, end_id]
        edge_index[1] += [end_id, begin_id]
        edge_attr += [get_bond_properties(ob_bond),
                      get_bond_properties(ob_bond)]

    return (atom_properties_list,
            edge_index, edge_attr)


def extract_atom_id_from_oddt_interractions(mol1_atoms_array, mol2_atoms_array):
    dico = {}
    for mol1_atom, mol2_atom in zip(mol1_atoms_array, mol2_atoms_array):
        mol1_atm_id = int(mol1_atom[0])
        mol2_atom_id = int(mol2_atom[0])
        if mol1_atm_id not in dico.keys():
            dico[mol1_atm_id] = []
        dico[mol1_atm_id] += [mol2_atom_id]
    return dico


def extract_residu_id_from_oddt_interractions(protein_residus_array):
    list_residu = []
    if protein_residus_array.shape[0] != 0:
        for np_row in np.nditer(protein_residus_array):
            list_residu.append(np_row.tolist()[2])
    return list_residu


def atom_pair_in_dico(dico, mol1_atom_id, mol2_atom_id):
    if mol1_atom_id in dico.keys():
        return mol2_atom_id in dico[mol1_atom_id]
    return False


def is_pi(res_name: str, atom_name: str) -> bool:
    if res_name == 'HIS':
        if (atom_name == 'CG' or atom_name == 'CD2' or atom_name == 'NE2'
                or atom_name == 'CE1' or atom_name == 'ND1'):
            return True
        return False
    elif res_name == 'PHE':
        if (atom_name == 'CG' or atom_name == 'CD2' or atom_name == 'CE2'
                or atom_name == 'CZ' or atom_name == 'CE1'
                or atom_name == 'CD1'):
            return True
        return False
    elif res_name == 'TYR':
        if (atom_name == 'CG' or atom_name == 'CD1' or atom_name == 'CE1'
                or atom_name == 'CE2' or atom_name == 'CD2'
                or atom_name == 'CZ'):
            return True
        return False
    elif res_name == 'TRP':
        if (atom_name == 'CG' or atom_name == 'CD1' or atom_name == 'NE1'
                or atom_name == 'CE2' or atom_name == 'CD2'
                or atom_name == 'CE3' or atom_name == 'CZ2'
                or atom_name == 'CZ3' or atom_name == 'CH2'):
            return True
        return False
    return False


def close_contact_to_dict(protein_close_contacts: np.ndarray,
                          ligand_close_contacts: np.ndarray) -> dict:
    dict_close_contacts = {}
    for protein_atm, ligand_atm in zip(protein_close_contacts, ligand_close_contacts):
        protein_atm_id = int(protein_atm[0])
        ligand_atom_id = int(ligand_atm[0])
        if ligand_atom_id not in dict_close_contacts.keys():
            dict_close_contacts[ligand_atom_id] = []
        dict_close_contacts[ligand_atom_id] += [[
            protein_atm_id, protein_atm, ligand_atm]]
    return dict_close_contacts


def get_bonds_protein_ligand(protein: Molecule, ligand: Molecule,
                             cutoff: float,
                             list_atom_name: List[str]) -> Tuple[List, List, List, List]:
    """Returns the bond between the protein and the ligand regarding the cutoff.
     All ligand must have at least one edge with an protein's atom.

    Args:
        protein (Molecule): protein
        ligand (Molecule):  ligand
        cutoff (float): The maximal distance between two atoms to connect them with an edge.
        list_atom_name (List[str]): List of PDB atom name, use for pi interactions

    Returns:
        Tuple(List, List, List, List): Protein to Ligand Edge Index, Ligand to
            Protein Edge Index, Protein to Ligand Edge Attr, Ligand to Protein Edge Attr
    """
    close_contact_protein, close_contact_ligand = interactions.close_contacts(
        protein.atom_dict, ligand.atom_dict, cutoff=cutoff)

    hbond_protein, hbond_ligand, _ = interactions.hbonds(
        protein, ligand, cutoff=cutoff)
    dico_hbonds = extract_atom_id_from_oddt_interractions(
        hbond_protein, hbond_ligand)

    hydrophobic_contact_protein, hydrophobic_contact_ligand = interactions.hydrophobic_contacts(
        protein, ligand, cutoff=cutoff)
    dico_hydrophobic_contact = extract_atom_id_from_oddt_interractions(
        hydrophobic_contact_protein, hydrophobic_contact_ligand)

    salt_bridges_protein, salt_bridges_ligand = interactions.salt_bridges(
        protein, ligand, cutoff=cutoff)
    dico_salt_bridges = extract_atom_id_from_oddt_interractions(
        salt_bridges_protein, salt_bridges_ligand)

    # pi_stacking
    pi_stacking_protein_residue, pi_stacking_ligand, _, _ = interactions.pi_stacking(
        protein, ligand, cutoff=cutoff)
    list_residus_pi_stacking = extract_residu_id_from_oddt_interractions(
        pi_stacking_protein_residue)

    # pi_cation
    pi_cation_protein_residue, pi_cation_ligand, _ = interactions.pi_cation(
        protein, ligand, cutoff=cutoff)
    list_residus_pi_cation = extract_residu_id_from_oddt_interractions(
        pi_cation_protein_residue)

    protein_atm_to_res_dict = {}
    for np_row in np.nditer(protein.atom_dict):
        protein_atm_to_res_dict[np_row.tolist()[0]] = np_row.tolist()[9]

    p_to_l_edge_index = [[], []]
    p_to_l_edge_attr = []
    l_to_p_edge_index = [[], []]
    l_to_p_edge_attr = []
    dict_close_contacts = close_contact_to_dict(
        close_contact_protein, close_contact_ligand)
    dists = distance(protein.atom_dict['coords'], ligand.atom_dict['coords'])
    for ligand_atom_id in range(len(ligand.atoms)):
        if ligand_atom_id in dict_close_contacts.keys():
            for close_contact in dict_close_contacts[ligand_atom_id]:
                protein_atm_id = close_contact[1][0]
                protein_atm = close_contact[1]
                ligand_atm = close_contact[2]
                dist = distance([protein_atm[1]], [ligand_atm[1]])[0][0]
                protein_atom_res = protein_atm_to_res_dict[protein_atm_id]
                protein_atom_name = list_atom_name[protein_atm_id]
                res_name = protein_atm[11]

                atom_is_pi = is_pi(res_name, protein_atom_name)

                is_hbond = atom_pair_in_dico(
                    dico_hbonds, protein_atm_id, ligand_atom_id)
                is_hydrophobic_contact = atom_pair_in_dico(
                    dico_hydrophobic_contact, protein_atm_id, ligand_atom_id)
                is_salt_bridge = atom_pair_in_dico(
                    dico_salt_bridges, protein_atm_id, ligand_atom_id)
                is_pi_stacking = protein_atom_res in list_residus_pi_stacking and atom_is_pi
                is_pi_cation = protein_atom_res in list_residus_pi_cation and atom_is_pi

                p_to_l_edge_index[0] += [int(protein_atm_id)]
                p_to_l_edge_index[1] += [int(ligand_atom_id)]
                l_to_p_edge_index[0] += [int(ligand_atom_id)]
                l_to_p_edge_index[1] += [int(protein_atm_id)]
                p_to_l_edge_attr += [[dist,
                                      is_hbond, is_hydrophobic_contact, is_salt_bridge,
                                      is_pi_stacking, is_pi_cation]]
                l_to_p_edge_attr += [[dist,
                                      is_hbond, is_hydrophobic_contact, is_salt_bridge,
                                      is_pi_stacking, is_pi_cation]]
        else:
            closer_protein_atm_id = np.argmin(dists[:, ligand_atom_id])
            dist = dists[closer_protein_atm_id, ligand_atom_id]
            p_to_l_edge_index[0] += [int(closer_protein_atm_id)]
            p_to_l_edge_index[1] += [int(ligand_atom_id)]
            l_to_p_edge_index[0] += [int(ligand_atom_id)]
            l_to_p_edge_index[1] += [int(closer_protein_atm_id)]
            p_to_l_edge_attr += [[dist, False, False, False, False, False]]
            l_to_p_edge_attr += [[dist, False, False, False, False, False]]
    return p_to_l_edge_index, l_to_p_edge_index, p_to_l_edge_attr, l_to_p_edge_attr


def featurize(protein_path: str, ligand_path: str, cutoff: float,
              ligand_filetype: str = None) -> Tuple:
    """Featurize a protein and a ligand to a set of nodes and edges

    Args:
        protein_path (str): Path to the protein file (PDB)
        ligand_path (str): Path to the ligand file
        ligand_filetype (str): Type of the Ligand (PDB or MOL2). Defaults to None

    Returns:
        Tuple: Nodes and edges
    """
    # protein_path can be protein or pocket path
    protein = open_pdb(protein_path, hydrogens_removal=True)
    protein.protein = True
    ligand = None
    if ligand_filetype is None:
        ligand_filetype = osp.splitext(ligand_path).replace('.', '')
    if ligand_filetype.lower() == 'mol2':
        ligand = open_mol2(ligand_path, hydrogens_removal=True)
    elif ligand_filetype.lower() == 'pdb':
        ligand = open_pdb(ligand_path, hydrogens_removal=True)
    assert ligand is not None, 'Error when loading ligand file'
    # Get PDB atom name
    ppdb = bpdb.PandasPdb()
    ppdb.read_pdb(protein_path)
    atom_df = ppdb.df['ATOM'][ppdb.df['ATOM']['element_symbol'] != 'H']
    list_atom_name = atom_df['atom_name'].tolist()

    (protein_atom_properties_list,
     protein_edge_index, protein_edge_attr) = get_molecule_properties(protein)
    (ligand_atom_properties_list,
     ligand_edge_index, ligand_edge_attr) = get_molecule_properties(ligand)

    (p_atm_to_l_edge_index, l_to_p_atm_edge_index, p_atm_to_l_edge_attr,
     l_to_p_atm_edge_attr) = get_bonds_protein_ligand(protein, ligand,
                                                      cutoff=cutoff,
                                                      list_atom_name=list_atom_name)

    return (protein_atom_properties_list,  # protein_atoms.x
            ligand_atom_properties_list,  # ligand_atoms.x

            protein_edge_index,  # protein_atoms <-> protein_atoms
            ligand_edge_index,  # ligand_atoms <-> ligand_atoms
            l_to_p_atm_edge_index,  # ligand_atoms ->  protein_atom
            p_atm_to_l_edge_index,  # protein_atoms -> ligand_atoms

            protein_edge_attr,  # protein_atoms <-> protein_atoms
            ligand_edge_attr,  # ligand_atoms <-> ligand_atoms
            l_to_p_atm_edge_attr,  # ligand_atoms ->  protein_atoms
            p_atm_to_l_edge_attr  # protein_atoms -> ligand_atoms
            )
