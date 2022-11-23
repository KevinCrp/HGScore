import os
import os.path as osp
from glob import glob

import torch

from hgcn_4_pls.data import (CASFDataset, DockingPower_Dataset,
                             DockingPowerDataModule, PDBBindDataModule,
                             PDBBindDataset, process_and_save_graph_from_dir,
                             process_and_save_graph_from_files,
                             process_graph_from_dir, process_graph_from_files)

DATA_ROOT = 'tests/data'
ATM_CUTOFF = 4.0
POCKET = True

PATH_TO_PDB_DIR = 'tests/data/raw/4llx'
PATH_TO_GRAPH_TEST = 'tests/data/raw/4llx/graph.pt'
PATH_TO_PROTEIN_PDB = 'tests/data/raw/4llx/4llx_protein.pdb'
PATH_TO_POCKET_PDB = 'tests/data/raw/4llx/4llx_pocket.pdb'
PATH_TO_CLEAN_PROTEIN_PDB = 'tests/data/raw/4llx/4llx_pocket_clean.pdb'
PATH_TO_LIGAND_MOL2 = 'tests/data/raw/4llx/4llx_ligand.mol2'
PATH_TO_LIGAND_PDB = 'tests/data/raw/4llx/4llx_ligand.pdb'


def rm_processed_file(filepath):
    if osp.isfile(filepath):
        os.remove(filepath)


def remove_decoys_file():
    decoys_path = 'tests/data/decoys_docking/*.mol2'
    list_files = glob(decoys_path)

    for path in list_files:
        fname = path.split('/')[-1]
        if not 'decoys' in fname:
            os.remove(path)


def test_process_graph_from_files():
    g = process_graph_from_files(protein_path=PATH_TO_POCKET_PDB,
                                 ligand_path=PATH_TO_LIGAND_MOL2,
                                 atomic_distance_cutoff=ATM_CUTOFF,
                                 target=4.36,
                                 cluster=1,
                                 pdb_id='4llx',
                                 rmsd=1.0,
                                 decoy_id='ligand')
    assert g['protein_atoms'].x.shape[0] == 199
    assert g['protein_atoms'].x.shape[1] == 23
    assert g['ligand_atoms'].x.shape[0] == 9
    assert g['ligand_atoms'].x.shape[1] == 23
    assert g[('ligand_atoms', 'interact_with', 'protein_atoms')
             ].edge_attr.shape[0] == 31


def test_process_graph_from_dir():
    g = process_graph_from_dir(dir_path=PATH_TO_PDB_DIR,
                               atomic_distance_cutoff=ATM_CUTOFF,
                               only_pocket=POCKET,
                               target=4.36,
                               cluster=1,
                               pdb_id='4llx',
                               rmsd=1.0,
                               decoy_id='ligand')
    assert g['protein_atoms'].x.shape[0] == 199
    assert g['protein_atoms'].x.shape[1] == 23
    assert g['ligand_atoms'].x.shape[0] == 9
    assert g['ligand_atoms'].x.shape[1] == 23
    assert g[('ligand_atoms', 'interact_with', 'protein_atoms')
             ].edge_attr.shape[0] == 31


def test_process_and_save_graph_from_files():
    rm_processed_file(PATH_TO_GRAPH_TEST)
    process_and_save_graph_from_files(protein_path=PATH_TO_POCKET_PDB,
                                      ligand_path=PATH_TO_LIGAND_MOL2,
                                      processed_filename=PATH_TO_GRAPH_TEST,
                                      atomic_distance_cutoff=ATM_CUTOFF,
                                      target=4.36,
                                      cluster=1,
                                      pdb_id='4llx',
                                      rmsd=1.0,
                                      decoy_id='ligand')
    g = torch.load(PATH_TO_GRAPH_TEST)
    assert g['protein_atoms'].x.shape[0] == 199
    assert g['protein_atoms'].x.shape[1] == 23
    assert g['ligand_atoms'].x.shape[0] == 9
    assert g['ligand_atoms'].x.shape[1] == 23
    assert g[('ligand_atoms', 'interact_with', 'protein_atoms')
             ].edge_attr.shape[0] == 31


def test_process_and_save_graph_from_dir():
    rm_processed_file(PATH_TO_GRAPH_TEST)
    process_and_save_graph_from_dir(dir_path=PATH_TO_PDB_DIR,
                                    processed_filename=PATH_TO_GRAPH_TEST,
                                    atomic_distance_cutoff=ATM_CUTOFF,
                                    only_pocket=POCKET,
                                    target=4.36,
                                    cluster=1,
                                    pdb_id='4llx',
                                    rmsd=1.0,
                                    decoy_id='ligand')
    g = torch.load(PATH_TO_GRAPH_TEST)
    assert g['protein_atoms'].x.shape[0] == 199
    assert g['protein_atoms'].x.shape[1] == 23
    assert g['ligand_atoms'].x.shape[0] == 9
    assert g['ligand_atoms'].x.shape[1] == 23
    assert g[('ligand_atoms', 'interact_with', 'protein_atoms')
             ].edge_attr.shape[0] == 31


def test_pdbbind_dataset_train():
    dataset_path = 'tests/data/processed/pocket_4.0_train.pt'
    rm_processed_file(dataset_path)
    PDBBindDataset(root=DATA_ROOT,
                   stage='train',
                   atomic_distance_cutoff=ATM_CUTOFF,
                   only_pocket=POCKET)
    dt = torch.load(dataset_path)[0]
    assert len(dt['y']) == 5


def test_pdbbind_dataset_val():
    dataset_path = 'tests/data/processed/pocket_4.0_val.pt'
    rm_processed_file(dataset_path)
    PDBBindDataset(root=DATA_ROOT,
                   stage='val',
                   atomic_distance_cutoff=ATM_CUTOFF,
                   only_pocket=POCKET)
    dt = torch.load(dataset_path)[0]
    assert len(dt['y']) == 3


def test_casf_13():
    dataset_path = 'tests/data/processed/pocket_4.0_casf_13.pt'
    rm_processed_file(dataset_path)
    CASFDataset(root=DATA_ROOT,
                year='13',
                atomic_distance_cutoff=ATM_CUTOFF,
                only_pocket=POCKET)
    dt = torch.load(dataset_path)[0]
    assert len(dt['y']) == 4


def test_casf_13():
    dataset_path = 'tests/data/processed/pocket_4.0_casf_16.pt'
    rm_processed_file(dataset_path)
    CASFDataset(root=DATA_ROOT,
                year='16',
                atomic_distance_cutoff=ATM_CUTOFF,
                only_pocket=POCKET)
    dt = torch.load(dataset_path)[0]
    assert len(dt['y']) == 4


def test_docking_power():
    dataset_path = 'tests/data/processed/pocket_4.0_dockingP.pt'
    rm_processed_file(dataset_path)
    remove_decoys_file()
    DockingPower_Dataset(root=DATA_ROOT,
                         year='16',
                         atomic_distance_cutoff=ATM_CUTOFF,
                         only_pocket=POCKET)
    dt = torch.load(dataset_path)[0]
    assert len(dt['pdb_id']) == 358


def test_PDBBindDataModule():
    dm = PDBBindDataModule(root=DATA_ROOT,
                           atomic_distance_cutoff=ATM_CUTOFF,
                           batch_size=2,
                           num_workers=1,
                           only_pocket=POCKET,
                           persistent_workers=True)
    assert dm is not None
    dm.setup()
    dl = dm.train_dataloader()
    assert len(dl) == 3
    dl = dm.val_dataloader()
    assert len(dl) == 2
    dl = dm.casf_dataloader(13)
    assert len(dl) == 4
    dl = dm.casf_dataloader(16)
    assert len(dl) == 4
    dl = dm.casf_dataloader(95)
    assert dl is None
    dl = dm.test_dataloader()
    assert len(dl) == 4


def test_DockingPowerDataModule():
    dm = DockingPowerDataModule(root=DATA_ROOT,
                                atomic_distance_cutoff=ATM_CUTOFF,
                                batch_size=2,
                                num_workers=1,
                                only_pocket=POCKET,
                                persistent_workers=True)
    assert dm is not None
    dm.setup()
    dl = dm.power_docking_dataloader()
    assert len(dl) == 179
