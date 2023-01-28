from HGScore.data import clean_pdb
from HGScore.featurizer import (atom_degree_one_hot,
                                   atom_hybridization_one_hot,
                                   atom_type_one_hot, featurize, is_pi,
                                   open_mol2, open_pdb)

PATH_TO_PROTEIN_PDB = 'tests/data/raw/4llx/4llx_protein.pdb'
PATH_TO_POCKET_PDB = 'tests/data/raw/4llx/4llx_pocket.pdb'
PATH_TO_CLEAN_PROTEIN_PDB = 'tests/data/raw/4llx/4llx_pocket_clean.pdb'
PATH_TO_LIGAND_MOL2 = 'tests/data/raw/4llx/4llx_ligand.mol2'
PATH_TO_LIGAND_PDB = 'tests/data/raw/4llx/4llx_ligand.pdb'


def test_open_pdb():
    nb_atom_with_H = len(
        open_pdb(filepath=PATH_TO_PROTEIN_PDB, hydrogens_removal=False).atoms)
    assert(nb_atom_with_H == 3389)
    nb_atom_withouy_H = len(
        open_pdb(filepath=PATH_TO_PROTEIN_PDB, hydrogens_removal=True).atoms)
    assert(nb_atom_withouy_H == 2846)


def test_open_mol2():
    nb_atom_with_H = len(
        open_mol2(filepath=PATH_TO_LIGAND_MOL2, hydrogens_removal=False).atoms)
    assert(nb_atom_with_H == 18)
    nb_atom_withouy_H = len(
        open_mol2(filepath=PATH_TO_LIGAND_MOL2, hydrogens_removal=True).atoms)
    assert(nb_atom_withouy_H == 9)


def test_atom_type_one_hot():
    assert atom_type_one_hot(5) == [1, 0, 0, 0, 0, 0, 0, 0]
    assert atom_type_one_hot(6) == [0, 1, 0, 0, 0, 0, 0, 0]
    assert atom_type_one_hot(7) == [0, 0, 1, 0, 0, 0, 0, 0]
    assert atom_type_one_hot(8) == [0, 0, 0, 1, 0, 0, 0, 0]
    assert atom_type_one_hot(9) == [0, 0, 0, 0, 1, 0, 0, 0]
    assert atom_type_one_hot(15) == [0, 0, 0, 0, 0, 1, 0, 0]
    assert atom_type_one_hot(16) == [0, 0, 0, 0, 0, 0, 1, 0]
    assert atom_type_one_hot(1) == [0, 0, 0, 0, 0, 0, 0, 1]


def test_atom_hybridization_one_hot():
    assert atom_hybridization_one_hot(1) == [0, 1, 0, 0, 0, 0, 0]
    assert atom_hybridization_one_hot(2) == [0, 0, 1, 0, 0, 0, 0]
    assert atom_hybridization_one_hot(3) == [0, 0, 0, 1, 0, 0, 0]
    assert atom_hybridization_one_hot(4) == [0, 0, 0, 0, 1, 0, 0]
    assert atom_hybridization_one_hot(5) == [0, 0, 0, 0, 0, 1, 0]
    assert atom_hybridization_one_hot(6) == [0, 0, 0, 0, 0, 0, 1]
    assert atom_hybridization_one_hot(0) == [1, 0, 0, 0, 0, 0, 0]


def test_atom_degree_one_hot():
    assert atom_degree_one_hot(0) == [1, 0, 0, 0, 0, 0, 0]
    assert atom_degree_one_hot(1) == [0, 1, 0, 0, 0, 0, 0]
    assert atom_degree_one_hot(2) == [0, 0, 1, 0, 0, 0, 0]
    assert atom_degree_one_hot(3) == [0, 0, 0, 1, 0, 0, 0]
    assert atom_degree_one_hot(4) == [0, 0, 0, 0, 1, 0, 0]
    assert atom_degree_one_hot(5) == [0, 0, 0, 0, 0, 1, 0]
    assert atom_degree_one_hot(85) == [0, 0, 0, 0, 0, 0, 1]


def check_featurize(protein_atom_properties_list,
                    ligand_atom_properties_list,
                    protein_edge_index,
                    ligand_edge_index,
                    l_to_p_atm_edge_index,
                    p_atm_to_l_edge_index,
                    protein_edge_attr,
                    ligand_edge_attr,
                    l_to_p_atm_edge_attr,
                    p_atm_to_l_edge_attr):

    assert len(protein_atom_properties_list) == 199
    assert len(protein_atom_properties_list[0]) == 23
    assert len(ligand_atom_properties_list) == 9
    assert len(ligand_atom_properties_list[0]) == 23
    assert len(protein_edge_index) == 2
    assert len(protein_edge_index[0]) == 390
    assert len(ligand_edge_index) == 2
    assert len(ligand_edge_index[0]) == 18
    assert len(l_to_p_atm_edge_index) == 2
    assert len(l_to_p_atm_edge_index[0]) == 31
    assert len(p_atm_to_l_edge_index) == 2
    assert len(p_atm_to_l_edge_index[0]) == 31
    assert len(protein_edge_attr) == 390
    assert len(protein_edge_attr[0]) == 6
    assert len(ligand_edge_attr) == 18
    assert len(ligand_edge_attr[0]) == 6
    assert len(l_to_p_atm_edge_attr) == 31
    assert len(l_to_p_atm_edge_attr[0]) == 6
    assert len(p_atm_to_l_edge_attr) == 31
    assert len(p_atm_to_l_edge_attr[0]) == 6


def test_is_pi():
    # Key are residues / Values are atom_names with pi interaction
    dico = {
        'HIS': ['CG', 'CD2', 'NE2', 'CE1', 'ND1'],
        'PHE': ['CG', 'CD2', 'CE2', 'CZ', 'CE1', 'CD1'],
        'TYR': ['CG', 'CD1', 'CE1', 'CE2', 'CD2', 'CZ'],
        'TRP': ['CG', 'CD1', 'NE1', 'CE2', 'CD2', 'CE3', 'CZ2', 'CZ3', 'CH2']
    }
    for res_name in dico.keys():
        for atom_name in dico[res_name]:
            assert is_pi(res_name=res_name, atom_name=atom_name)
    assert not is_pi(res_name='ALA', atom_name='CA')
    assert not is_pi(res_name='HIS', atom_name='CA')
    assert not is_pi(res_name='PHE', atom_name='CA')
    assert not is_pi(res_name='TYR', atom_name='CA')
    assert not is_pi(res_name='TRP', atom_name='CA')


def test_featurize_ligand_mol2():
    clean_pdb(PATH_TO_POCKET_PDB, PATH_TO_CLEAN_PROTEIN_PDB)
    (protein_atom_properties_list,  # protein_atoms.x
     ligand_atom_properties_list,  # ligand_atoms.x

     protein_edge_index,  # protein_atoms <-> protein_atoms
     ligand_edge_index,  # ligand_atoms <-> ligand_atoms
     l_to_p_atm_edge_index,  # ligand_atoms ->  protein_atom
     p_atm_to_l_edge_index,  # protein_atoms -> ligand_atoms

     protein_edge_attr,  # protein_atoms <-> protein_atoms
     ligand_edge_attr,  # ligand_atoms <-> ligand_atoms
     l_to_p_atm_edge_attr,  # ligand_atoms ->  protein_atoms
     p_atm_to_l_edge_attr) = featurize(protein_path=PATH_TO_CLEAN_PROTEIN_PDB,
                                       ligand_path=PATH_TO_LIGAND_MOL2,
                                       cutoff=4.0)
    check_featurize(protein_atom_properties_list,
                    ligand_atom_properties_list,
                    protein_edge_index,
                    ligand_edge_index,
                    l_to_p_atm_edge_index,
                    p_atm_to_l_edge_index,
                    protein_edge_attr,
                    ligand_edge_attr,
                    l_to_p_atm_edge_attr,
                    p_atm_to_l_edge_attr)


def test_featurize_ligand_pdb():
    clean_pdb(PATH_TO_POCKET_PDB, PATH_TO_CLEAN_PROTEIN_PDB)
    (protein_atom_properties_list,  # protein_atoms.x
     ligand_atom_properties_list,  # ligand_atoms.x

     protein_edge_index,  # protein_atoms <-> protein_atoms
     ligand_edge_index,  # ligand_atoms <-> ligand_atoms
     l_to_p_atm_edge_index,  # ligand_atoms ->  protein_atom
     p_atm_to_l_edge_index,  # protein_atoms -> ligand_atoms

     protein_edge_attr,  # protein_atoms <-> protein_atoms
     ligand_edge_attr,  # ligand_atoms <-> ligand_atoms
     l_to_p_atm_edge_attr,  # ligand_atoms ->  protein_atoms
     p_atm_to_l_edge_attr) = featurize(protein_path=PATH_TO_CLEAN_PROTEIN_PDB,
                                       ligand_path=PATH_TO_LIGAND_PDB,
                                       cutoff=4.0)

    check_featurize(protein_atom_properties_list,
                    ligand_atom_properties_list,
                    protein_edge_index,
                    ligand_edge_index,
                    l_to_p_atm_edge_index,
                    p_atm_to_l_edge_index,
                    protein_edge_attr,
                    ligand_edge_attr,
                    l_to_p_atm_edge_attr,
                    p_atm_to_l_edge_attr)


def test_featurize_ligand_pdb_intermol():
    clean_pdb(PATH_TO_POCKET_PDB, PATH_TO_CLEAN_PROTEIN_PDB)
    (protein_atom_properties_list,  # protein_atoms.x
     ligand_atom_properties_list,  # ligand_atoms.x

     protein_edge_index,  # protein_atoms <-> protein_atoms
     ligand_edge_index,  # ligand_atoms <-> ligand_atoms
     l_to_p_atm_edge_index,  # ligand_atoms ->  protein_atom
     p_atm_to_l_edge_index,  # protein_atoms -> ligand_atoms

     protein_edge_attr,  # protein_atoms <-> protein_atoms
     ligand_edge_attr,  # ligand_atoms <-> ligand_atoms
     l_to_p_atm_edge_attr,  # ligand_atoms ->  protein_atoms
     p_atm_to_l_edge_attr) = featurize(protein_path=PATH_TO_CLEAN_PROTEIN_PDB,
                                       ligand_path=PATH_TO_LIGAND_PDB,
                                       cutoff=1.0)

    assert len(l_to_p_atm_edge_index[0]) == 9
    assert len(p_atm_to_l_edge_index[0]) == 9
    assert len(l_to_p_atm_edge_attr) == 9
    assert len(p_atm_to_l_edge_attr) == 9
