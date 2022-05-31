import os
import sys
from typing import List, Tuple

import scipy.spatial
from openbabel import openbabel, pybel

MAX_BOND_ATOMIC_DISTANCE = 7.0


def redirect_c_std_err():
    """Redirect the c std error to the file obabel.err
    """
    # https://stackoverflow.com/questions/8804893/redirect-stdout-from-python-for-c-calls
    sys.stderr.flush()
    newstderr = os.dup(2)
    err_out = os.open('obabel.err', os.O_CREAT | os.O_WRONLY | os.O_APPEND)
    os.dup2(err_out, 2)
    os.close(err_out)
    sys.stderr = os.fdopen(newstderr, 'w')


def open_pdb(filepath: str, hydrogens_removal: bool = True) -> pybel.Molecule:
    """Open and load a molecule from a PDB file.

    Args:
        filepath (str): The PDB file path
        hydrogens_removal (bool, optional): To remove hydrogen atoms. Defaults to True.

    Returns:
        pybel.Molecule: The loaded molecule
    """
    redirect_c_std_err()
    pymol = next(pybel.readfile('pdb', filepath))
    if hydrogens_removal:
        pymol.removeh()
    return pymol


def open_mol2(filepath: str, hydrogens_removal: bool = True) -> pybel.Molecule:
    """Open and load a molecule from a MOL2 file.

    Args:
        filepath (str): The MOL2 file path
        hydrogens_removal (bool, optional): To remove hydrogen atoms. Defaults to True.

    Returns:
        pybel.Molecule: The loaded molecule
    """
    redirect_c_std_err()
    pymol = next(pybel.readfile('mol2', filepath))
    if hydrogens_removal:
        pymol.removeh()
    return pymol


def atom_type_one_hot(atomic_num: int) -> List[int]:
    """Returns the one-hot encoded atom type [B, C, N, O, F, P, S, Others]

    Args:
        atomic_num (int): The atom's atomic number

    Returns:
        List[int]: The one-hot encoded type
    """
    one_hot = 8*[0]
    used_atom_num = [5, 6, 7, 8, 9, 15, 16]  # B, C, N, O, F, P, S, Others
    d_atm_num = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6}
    if atomic_num in used_atom_num:
        one_hot[d_atm_num[atomic_num]] = 1
    return one_hot


def atom_hyb_one_hot(atom: pybel.Atom) -> List[int]:
    """"Returns the one-hot encoded atom hybridization [other, sp, sp2, sp3, sq. planar, trig. bipy, octahedral]

    Args:
        atom (pybel.Atom): A Pybel atom

    Returns:
        List[int]: The one-hot encoded hybridization
    """
    oh_hyb = 7*[0]
    hyb = atom.GetHyb()
    if hyb not in [1, 2, 3, 4, 5, 6]:
        hyb = 0
    oh_hyb[hyb] = 1
    return oh_hyb


def atom_heavy_degree_one_hot(atom: pybel.Atom) -> List[int]:
    """Returns the one-hot encoded atom heavy degree [0, 1, 2, 3, 4, 5, 6+]

    Args:
        atom (pybel.Atom): A Pybel atom

    Returns:
        List[int]: The one-hot encoded heavy degree
    """
    degree = 7*[0]
    atm_deg = atom.GetHvyDegree()
    if atm_deg > 6:
        atm_deg = 6
    degree[atm_deg] = 1
    return degree


def atom_hetero_degree_one_hot(atom: pybel.Atom) -> List[int]:
    """Returns the one-hot encoded atom hetero degree [0, 1, 2, 3, 4, 5, 6+]

    Args:
        atom (pybel.Atom): A Pybel atom

    Returns:
        List[int]: The one-hot encoded hetero degree
    """
    degree = 7*[0]
    atm_deg = atom.GetHeteroDegree()
    if atm_deg > 6:
        atm_deg = 6
    degree[atm_deg] = 1
    return degree


def get_atom_properties(atom: pybel.Atom) -> Tuple[List, float, float, float]:
    """Returns atomic properties: 
     * The one-hot encoded atom type
     * The one-hot encoded hybridization
     * The one-hot encoded heavy degree
     * The one-hot encoded hetero degree
     * Partial Charge
     * Is the atom hydrophobic
     * Is in an aromatic ring
     * Is HBond acceptor
     * Is HBond donor
     * Is in a ring
     * The [x, y, z] position

    Args:
        atom (pybel.Atom):  A Pybel atom

    Returns:
        Tuple[List, float, float, float]: The properties list and the x, y, z coordinates
    """
    # From TFbio - Kalansanty (https://gitlab.com/cheminfIBB/tfbio/-/blob/master/tfbio/data.py)
    smarts_hydrophobic = '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]'
    one_hot_enc_type = atom_type_one_hot(atom.GetAtomicNum())
    one_hot_enc_hyb = atom_hyb_one_hot(atom)
    one_hot_enc_heavy_degree = atom_heavy_degree_one_hot(atom)
    one_hot_enc_hetero_degree = atom_hetero_degree_one_hot(atom)
    hydrophobic = atom.MatchesSMARTS(smarts_hydrophobic)
    atom_property = [one_hot_enc_type + one_hot_enc_hyb +
                     one_hot_enc_heavy_degree + one_hot_enc_hetero_degree +
                     [atom.GetPartialCharge(),
                      hydrophobic,
                      atom.IsAromatic(),
                      atom.IsHbondAcceptor(),
                      atom.IsHbondDonor(),
                      atom.IsInRing()
                      ]]

    return atom_property, atom.GetX(), atom.GetY(), atom.GetZ()


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

# get_molecule_atomic_properties


def get_molecule_properties(pybel_mol: pybel.Molecule) -> Tuple[List, List, List, List]:
    """Computes all atomic and bond properties of a molecule

    Args:
        pybel_mol (pybel.Molecule): A Pybel Molecule

    Returns:
        Tuple[List, List, List, List]: (The atoms' properties, the atoms' position, the edge index, the edge attr)
    """
    atom_properties_list = []
    edge_index = [[], []]
    edge_attr = []
    pos_list = []
    for atom in openbabel.OBMolAtomIter(pybel_mol.OBMol):  # idx start by 1
        atom_properties, atom_x, atom_y, atom_z = get_atom_properties(atom)
        atom_properties_list += atom_properties
        pos_list += [[atom_x, atom_y, atom_z]]
    for bond in openbabel.OBMolBondIter(pybel_mol.OBMol):
        begin_id = bond.GetBeginAtom().GetIdx() - 1
        end_id = bond.GetEndAtom().GetIdx() - 1
        edge_index[0] += [begin_id, end_id]
        edge_index[1] += [end_id, begin_id]
        edge_attr += [get_bond_properties(bond), get_bond_properties(bond)]

    return (atom_properties_list,
            pos_list,
            edge_index, edge_attr)


def get_bonds_protein_ligand(protein_atm_pos: List, ligand_atm_pos: List, threshold: float = 1.0):
    """Returns the bond between the protein and the ligand regarding the threshold.
     All ligand must have at least one edge with an protein's atom.

    Args:
        protein_atm_pos (List): All protein's atoms position
        ligand_atm_pos (List):  All protein's atoms position
        threshold (float, optional): The maximal distance between two atoms to connect them with an edge. Defaults to 1.0.

    Returns:
        _type_: Protein to Ligand Edge Index, Ligand to Protein Edge Index, Protein to Ligand Edge Attr, Ligand to Protein Edge Attr
    """
    p_to_l_edge_index = [[], []]
    p_to_l_edge_attr = []
    l_to_p_edge_index = [[], []]
    l_to_p_edge_attr = []
    dist_mat = scipy.spatial.distance.cdist(
        protein_atm_pos, ligand_atm_pos)
    nb_protein_atm = len(protein_atm_pos)
    for ligand_node in range(len(ligand_atm_pos)):
        min_dist = 1000.0
        idx_min_dist = -1
        for prot_node in range(nb_protein_atm):
            dist = dist_mat[prot_node, ligand_node].item()
            if dist < min_dist:
                min_dist = dist
                idx_min_dist = prot_node
            if dist <= threshold:
                p_to_l_edge_index[0] += [prot_node]
                p_to_l_edge_index[1] += [ligand_node]
                l_to_p_edge_index[0] += [ligand_node]
                l_to_p_edge_index[1] += [prot_node]
                p_to_l_edge_attr += [[dist]]
                l_to_p_edge_attr += [[dist]]
        if min_dist > threshold:  # 0 link between atom i in ligand and protein
            p_to_l_edge_index[0] += [idx_min_dist]
            p_to_l_edge_index[1] += [ligand_node]
            l_to_p_edge_index[0] += [ligand_node]
            l_to_p_edge_index[1] += [idx_min_dist]
            p_to_l_edge_attr += [[min_dist]]
            l_to_p_edge_attr += [[min_dist]]
    return p_to_l_edge_index, l_to_p_edge_index, p_to_l_edge_attr, l_to_p_edge_attr


def featurize(protein_path: str, ligand_path: str) -> Tuple:
    """Featurize a protein and a ligand to a set of nodes and edges

    Args:
        protein_path (str): Path to the protein file (PDB)
        ligand_path (str): Path to the ligand file (MOL2)

    Returns:
        Tuple: Nodes and edges
    """
    # protein_path can be protein or pocket path
    protein = open_pdb(protein_path, hydrogens_removal=True)
    ligand = open_mol2(ligand_path, hydrogens_removal=True)

    (protein_atom_properties_list, protein_atm_pos,
     protein_edge_index, protein_edge_attr) = get_molecule_properties(protein)
    (ligand_atom_properties_list, ligand_atm_pos,
     ligand_edge_index, ligand_edge_attr) = get_molecule_properties(ligand)

    p_atm_to_l_edge_index, l_to_p_atm_edge_index, p_atm_to_l_edge_attr, l_to_p_atm_edge_attr = get_bonds_protein_ligand(
        protein_atm_pos, ligand_atm_pos, threshold=MAX_BOND_ATOMIC_DISTANCE)

    return (protein_atom_properties_list,  # protein_atoms.x
            ligand_atom_properties_list,  # ligand_atoms.x

            protein_edge_index,  # protein_atoms <-> protein_atoms
            ligand_edge_index,  # ligand_atoms <-> ligand_atoms
            l_to_p_atm_edge_index,  # ligand_atoms ->  protein_atom
            p_atm_to_l_edge_index,  # protein_atoms -> ligand_atoms

            protein_edge_attr,  # protein_atoms <-> protein_atoms
            ligand_edge_attr,  # ligand_atoms <-> ligand_atoms
            l_to_p_atm_edge_attr,  # ligand_atoms ->  protein_atoms
            p_atm_to_l_edge_attr,  # protein_atoms -> ligand_atoms
            )
