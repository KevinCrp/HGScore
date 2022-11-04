from bgcn_4_pls.featurizer import open_pdb
from bgcn_4_pls.utilities.pockets import pocket_extraction
from bgcn_4_pls.utilities.scpdb_split import load_index, split_dict, check_no_overlapping
PATH_TO_LIGAND_MOL2 = 'tests/data/raw/4llx/4llx_ligand.mol2'
PATH_TO_LIGAND_PDB = 'tests/data/raw/4llx/4llx_ligand.pdb'
PATH_TO_PROTEIN_PDB = 'tests/data/raw/4llx/4llx_protein.pdb'
PATH_TO_POCKET_TEST = 'tests/data/raw/4llx/4llx_pocket_test.pdb'


def test_pocket_extraction_mol2_ligand():
    pocket_extraction(PATH_TO_PROTEIN_PDB, PATH_TO_LIGAND_MOL2,
                      PATH_TO_POCKET_TEST, 10.0)
    nb_atom_with_H = len(
        open_pdb(filepath=PATH_TO_POCKET_TEST, hydrogens_removal=False).atoms)
    assert nb_atom_with_H == 458


def test_pocket_extraction_pdb_ligand():
    pocket_extraction(PATH_TO_PROTEIN_PDB, PATH_TO_LIGAND_PDB,
                      PATH_TO_POCKET_TEST, 10.0)
    nb_atom_with_H = len(
        open_pdb(filepath=PATH_TO_POCKET_TEST, hydrogens_removal=False).atoms)
    assert nb_atom_with_H == 458


PATH_TO_INDEX_CASF16 = "tests/data/index/CoreSet_2016.dat"
PATH_TO_INDEX_REF= "tests/data/index/INDEX_refined_data.2020"

def test_load_index():
    d = load_index(index_path=PATH_TO_INDEX_CASF16,
                   with_cluster=True,
                   exluded_pdb=['4llx'])
    assert len(d.keys()) == 284

def test_load_index():
    d = load_index(index_path=PATH_TO_INDEX_REF,
                   with_cluster=False,
                   exluded_pdb=[])
    assert len(d.keys()) == 5316

def test_split_dict():
    d = load_index(index_path=PATH_TO_INDEX_CASF16,
                   with_cluster=True,
                   exluded_pdb=[])
    d1, d2 = split_dict(d, 85)
    assert len(d1.keys()) == 85
    assert len(d2.keys()) == 200

def test_check_no_overlapping():
    d = load_index(index_path=PATH_TO_INDEX_CASF16,
                   with_cluster=True,
                   exluded_pdb=[])
    d1, d2 = split_dict(d, 85)
    assert check_no_overlapping(d1, d2) == True
    assert check_no_overlapping(d1, d1) == False

