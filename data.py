import argparse
import multiprocessing as mp
import os
import os.path as osp

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric as pyg
from biopandas.pdb import PandasPdb

import config as cfg
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
                     pdb_id: str = None,
                     cutoff: float = None,
                     rmsd: float = 0.0,
                     decoy_id: str = None
                     ) -> pyg.data.HeteroData:
    """Create a torch_geometric HeteroGraph of a protein-ligand complex

    Args:
        protein_path (str): Path to the protein file (PDB)
        ligand_path (str): Path to the liagnd file (MOL2)
        target (float, optional): Affinity target. Defaults to None.
        cluster (int, optional): Cluster ID for ranking. Defaults to None.
        pdb_id (int, optional): PDB ID for casf output. Defaults to None.
        cutoff (float): The maximal distance between two atoms to connect them with an edge. Defaults to None.
        rmsd (float, optional): Used for docking power. Defaults to 0.0.
        decoy_id (str, optional): Contains the ligand id for docking power. Defaults to None.

    Returns:
        pyg.data.HeteroData
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
     ) = f_atm.featurize(protein_path, ligand_path, cutoff=cutoff)

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
        protein_atm_to_protein_atm_edge_attr).float()
    data['ligand_atoms', 'linked_to', 'ligand_atoms'].edge_attr = torch.tensor(
        ligand_atm_to_ligand_atm_edge_attr).float()
    data['ligand_atoms', 'interact_with', 'protein_atoms'].edge_attr = torch.tensor(
        ligand_atm_to_protein_atm_edge_attr).float()
    data['protein_atoms', 'interact_with', 'ligand_atoms'].edge_attr = torch.tensor(
        protein_atm_to_ligand_atm_edge_attr).float()

    data.y = target
    data.cluster = cluster
    data.pdb_id = pdb_id
    data.rmsd = rmsd
    data.decoy_id = decoy_id

    return data


def process_graph(raw_path: str,
                  processed_filename: str,
                  atomic_distance_cutoff: float,
                  only_pocket: bool = False,
                  target: float = None, cluster: int = None,
                  pdb_id: str = None,
                  rmsd: float = 0.0) -> str:
    """Create a graph from PDBs

    Args:
        raw_path (str): Path to the directory containing raw files.
        processed_filename (str): Path to the processed file.
        atomic_distance_cutoff (float): Cutoff for inter-atomic distance.
        only_pocket (bool, optional): Use only the binding pocket or not. Defaults to False.
        target (float, optional): The target affinity. Defaults to None.
        cluster (int, optional): Cluster ID for ranking. Defaults to None.
        pdb_id (int, optional): PDB ID for casf output. Defaults to None.
        rmsd (float, optional): Used for docking power. Defaults to 0.0.

    Returns:
        str: Path where the graph is saved
    """
    pdb_id = raw_path.split('/')[-1]
    protein_path = osp.join(
        raw_path, pdb_id + '_pocket.pdb') if only_pocket else osp.join(raw_path, pdb_id + '_protein.pdb')
    protein_path_clean = osp.join(
        raw_path, pdb_id + '_pocket_clean.pdb') if only_pocket else osp.join(raw_path, pdb_id + '_protein_clean.pdb')
    ligand_path = osp.join(raw_path, pdb_id + '_ligand.mol2')
    if not osp.isfile(protein_path_clean):
        clean_pdb(protein_path, protein_path_clean)
    g = create_pyg_graph(
        protein_path_clean, ligand_path, target, cluster, pdb_id,
        atomic_distance_cutoff, rmsd)
    torch.save(g, processed_filename)
    return processed_filename


class PDBBindDataset(pyg.data.InMemoryDataset):
    """Torch Geometric Dataset, used for Train and Val set
    """

    def __init__(self, root: str, stage: str,
                 atomic_distance_cutoff: float,
                 only_pocket: bool = False):
        """init

        Args:
            root (str): The data directory
            stage (str): Train or val
            atomic_distance_cutoff (float): Cutoff for inter-atomic distance.
            only_pocket (bool, optional): Use only the binding pocket or not. Defaults to False.
        """
        self.stage = stage
        self.atomic_distance_cutoff = atomic_distance_cutoff
        self.only_pocket = only_pocket
        self.prefix = 'pocket' if only_pocket else 'protein'
        self.df = pd.read_csv(
            osp.join(cfg.data_path, '{}.csv'.format(stage))).set_index('pdb_id')
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        filename_list = []
        for pdb_id in self.df.index:
            filename_list.append(pdb_id)
        return filename_list

    @property
    def processed_file_names(self):
        return [osp.join(self.processed_dir, '{}_{}_{}.pt'.format(self.prefix,
                                                                  self.atomic_distance_cutoff,
                                                                  self.stage))]

    def process(self):
        print('\t', self.stage)
        pool_args = []
        i = 0
        for raw_path in self.raw_paths:
            filename = osp.join(self.processed_dir,
                                'TMP_{}__{}_{}_data_{}.pt'.format(self.prefix,
                                                                  self.atomic_distance_cutoff,
                                                                  self.stage, i))
            pdb_id = raw_path.split('/')[-1]
            pool_args.append((raw_path, filename, self.atomic_distance_cutoff,
                              self.only_pocket,
                             self.df.loc[pdb_id]['target']))
            i += 1
        pool = mp.Pool(cfg.preprocessing_nb_cpu)
        data_path_list = list(pool.starmap(process_graph, pool_args))
        data_list = []
        for p in data_path_list:
            data_list.append(torch.load(p))
            os.remove(p)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class CASFDataset(pyg.data.InMemoryDataset):
    """Torch Geometric Dataset, used for Test CASF13 and CASF16
    """

    def __init__(self, root: str, year: str,
                 atomic_distance_cutoff: float,
                 only_pocket: bool = False):
        self.year = year
        self.atomic_distance_cutoff = atomic_distance_cutoff
        self.only_pocket = only_pocket
        self.prefix = 'pocket' if only_pocket else 'protein'
        self.df = pd.read_csv(
            osp.join(cfg.data_path, 'casf{}.csv'.format(year))).set_index('pdb_id')
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        filename_list = []
        for pdb_id in self.df.index:
            filename_list.append(pdb_id)
        return filename_list

    @property
    def processed_file_names(self):
        return [osp.join(self.processed_dir, '{}_{}_casf_{}.pt'.format(self.prefix,
                                                                       self.atomic_distance_cutoff,
                                                                       self.year))]

    def process(self):
        i = 0
        print('\tCASF ', self.year)
        pool_args = []
        for raw_path in self.raw_paths:
            filename = osp.join(self.processed_dir,
                                'TMP_{}{}_data_{}.pt'.format(self.prefix,
                                                             self.year, i))
            pdb_id = raw_path.split('/')[-1]
            pool_args.append((raw_path, filename, self.atomic_distance_cutoff,
                              self.only_pocket,
                             self.df.loc[pdb_id]['target'],
                             self.df.loc[pdb_id]['cluster'],
                             pdb_id))
            i += 1
        pool = mp.Pool(cfg.preprocessing_nb_cpu)
        data_path_list = list(pool.starmap(process_graph, pool_args))
        data_list = []
        for p in data_path_list:
            data_list.append(torch.load(p))
            os.remove(p)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class PDBBindDataModule(pl.LightningDataModule):
    """PyTorch Lightning Datamodule
    """

    def __init__(self, root: str,
                 atomic_distance_cutoff: float,
                 batch_size: int = 1, num_workers: int = 1,
                 only_pocket: bool = False, sample_percent: bool = 100.0):
        """_summary_

        Args:
            root (str): Path to the data directory.
            atomic_distance_cutoff (float): Cutoff for inter-atomic distance.
            batch_size (int, optional): The batch size. Defaults to 1.
            num_workers (int, optional): The number of workers. Defaults to 1.
            only_pocket (bool, optional): Use only the binding pocket or not. Defaults to False.
            sample_percent (float, optional): Use it to train/validate on a subset of the dataset
        """
        self.atomic_distance_cutoff = atomic_distance_cutoff
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.only_pocket = only_pocket
        self.persistent_workers = True
        self.sample_percent = sample_percent
        super().__init__()

    def prepare_data(self):
        pass

    def setup(self, stage=''):
        self.dt_train = PDBBindDataset(
            root=self.root, atomic_distance_cutoff=self.atomic_distance_cutoff,
            stage='train', only_pocket=self.only_pocket)
        self.dt_val = PDBBindDataset(
            root=self.root, atomic_distance_cutoff=self.atomic_distance_cutoff,
            stage='val', only_pocket=self.only_pocket)
        if self.sample_percent != 100.0:
            nb_item_subset_train = int(
                len(self.dt_train) * self.sample_percent / 100)
            self.dt_train = self.dt_train.index_select(
                range(0, nb_item_subset_train))
            nb_item_subset_val = int(
                len(self.dt_val) * self.sample_percent / 100)
            self.dt_val = self.dt_val.index_select(
                range(0, nb_item_subset_val))
        self.dt_casf_13 = CASFDataset(root=cfg.data_path, year='13',
                                      atomic_distance_cutoff=self.atomic_distance_cutoff,
                                      only_pocket=cfg.data_use_only_pocket)
        self.dt_casf_16 = CASFDataset(root=cfg.data_path, year='16',
                                      atomic_distance_cutoff=self.atomic_distance_cutoff,
                                      only_pocket=cfg.data_use_only_pocket)

    def train_dataloader(self):
        return pyg.loader.DataLoader(self.dt_train,
                                     batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return pyg.loader.DataLoader(self.dt_val,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)

    def casf_13_dataloader(self):
        return pyg.loader.DataLoader(self.dt_casf_13,
                                     batch_size=1,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)

    def casf_16_dataloader(self):
        return pyg.loader.DataLoader(self.dt_casf_16,
                                     batch_size=1,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return pyg.loader.DataLoader(self.dt_casf_16,
                                     batch_size=1,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)


if __name__ == '__main__':
    # To Create datasets
    parser = argparse.ArgumentParser()
    parser.add_argument('-cutoff', '-c',
                        type=float,
                        help='If not set, config.py atomic_distance_cutoff is used')
    args = parser.parse_args()
    atomic_distance_cutoff = args.cutoff
    if atomic_distance_cutoff is None:
        atomic_distance_cutoff = cfg.atomic_distance_cutoff
    PDBBindDataset(root=cfg.data_path,
                   stage='train',
                   atomic_distance_cutoff=atomic_distance_cutoff,
                   only_pocket=cfg.data_use_only_pocket)
    PDBBindDataset(root=cfg.data_path,
                   stage='val',
                   atomic_distance_cutoff=atomic_distance_cutoff,
                   only_pocket=cfg.data_use_only_pocket)

    CASFDataset(root=cfg.data_path,
                year='13',
                atomic_distance_cutoff=atomic_distance_cutoff,
                only_pocket=cfg.data_use_only_pocket)
    CASFDataset(root=cfg.data_path,
                year='16',
                atomic_distance_cutoff=atomic_distance_cutoff,
                only_pocket=cfg.data_use_only_pocket)
