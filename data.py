import multiprocessing as mp
import os
import os.path as osp

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric as pyg

import config as cfg
import to_graph


def save_graph(raw_path: str, processed_path: str,
               only_pocket: bool = False,
               target: float = None, cluster: int = None,
               pdb_id: str = None) -> str:
    """Create a graph and save it

    Args:
        raw_path (str): Path to the directory containing raw files
        processed_path (str): Path where save the created garph
        only_pocket (bool, optional): Use only the binding pocket or not. Defaults to False.
        target (float, optional): The target affinity. Defaults to None.
        cluster (int, optional): Cluster ID for ranking. Defaults to None.
        pdb_id (int, optional): PDB ID for casf output. Defaults to None.

    Returns:
        str: Path where the graph is saved
    """
    pdb_id = raw_path.split('/')[-1]
    protein_path = osp.join(
        raw_path, pdb_id+'_pocket.pdb') if only_pocket else osp.join(raw_path, pdb_id+'_protein.pdb')
    protein_path_clean = osp.join(
        raw_path, pdb_id+'_pocket_clean.pdb') if only_pocket else osp.join(raw_path, pdb_id+'_protein_clean.pdb')
    ligand_path = osp.join(raw_path, pdb_id+'_ligand.mol2')
    if not osp.isfile(protein_path_clean):
        to_graph.clean_pdb(protein_path, protein_path_clean)
    g = to_graph.create_pyg_graph(
        protein_path_clean, ligand_path, target, cluster, pdb_id)
    torch.save(g, processed_path)
    return processed_path


class PDBBindDataset(pyg.data.InMemoryDataset):
    """Torch Geometric Dataset, used for Train and Val set
    """

    def __init__(self, root: str, stage: str, only_pocket: bool = False,
                 transform=None,
                 pre_transform=None):
        """init

        Args:
            root (str): The data directory
            stage (str): Train or val 
            only_pocket (bool, optional): Use only the binding pocket or not. Defaults to False.
            transform (_type_, optional): Defaults to None.
            pre_transform (_type_, optional): Defaults to None.
        """
        self.stage = stage
        self.only_pocket = only_pocket
        self.prefix = 'pocket_' if only_pocket else 'protein_'
        self.df = pd.read_csv(
            osp.join(cfg.data_path, '{}.csv'.format(stage))).set_index('pdb_id')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        filename_list = []
        raw_dir = osp.join(self.root, 'raw')
        for pdb_id in self.df.index:
            filename_list.append(osp.join(raw_dir, pdb_id))
        return filename_list

    @property
    def processed_file_names(self):
        return [osp.join(self.processed_dir, '{}{}.pt'.format(self.prefix, self.stage))]

    def process(self):
        i = 0
        print('\t', self.stage)
        pool_args = []
        for raw_path in self.raw_paths:
            filename = osp.join(self.processed_dir,
                                'TMP_{}{}_data_{}.pt'.format(self.prefix,
                                                             self.stage, i))
            pdb_id = raw_path.split('/')[-1]
            pool_args.append((raw_path, filename, self.only_pocket,
                             self.df.loc[pdb_id]['target']))
            i += 1
        pool = mp.Pool(cfg.preprocessing_nb_cpu)
        data_path_list = list(pool.starmap(save_graph, pool_args))
        data_list = []
        for p in data_path_list:
            data_list.append(torch.load(p))
            os.remove(p)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class CASFDataset(pyg.data.InMemoryDataset):
    """Torch Geometric Dataset, used for Test CASF13 and CASF16
    """

    def __init__(self, root: str, year: str, only_pocket: bool = False,
                 transform=None,
                 pre_transform=None):
        self.year = year
        self.only_pocket = only_pocket
        self.prefix = 'pocket_' if only_pocket else 'protein_'
        self.df = pd.read_csv(
            osp.join(cfg.data_path, 'casf{}.csv'.format(year))).set_index('pdb_id')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        filename_list = []
        raw_dir = osp.join(self.root, 'raw')
        for pdb_id in self.df.index:
            filename_list.append(osp.join(raw_dir, pdb_id))
        return filename_list

    @property
    def processed_file_names(self):
        return [osp.join(self.processed_dir, '{}casf_{}.pt'.format(self.prefix, self.year))]

    def process(self):
        i = 0
        print('\tCASF ', self.year)
        pool_args = []
        for raw_path in self.raw_paths:
            filename = osp.join(self.processed_dir,
                                'TMP_{}{}_data_{}.pt'.format(self.prefix,
                                                             self.year, i))
            pdb_id = raw_path.split('/')[-1]
            pool_args.append((raw_path, filename, self.only_pocket,
                             self.df.loc[pdb_id]['target'],
                             self.df.loc[pdb_id]['cluster'],
                             pdb_id))
            i += 1
        pool = mp.Pool(cfg.preprocessing_nb_cpu)
        data_path_list = list(pool.starmap(save_graph, pool_args))
        data_list = []
        for p in data_path_list:
            data_list.append(torch.load(p))
            os.remove(p)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class PDBBindDataModule(pl.LightningDataModule):
    """PyTorch Lightning Datamodule
    """

    def __init__(self, root: str, batch_size: int = 1, num_workers: int = 1,
                 only_pocket: bool = False):
        """_summary_

        Args:
            root (str): Path to the data directory
            batch_size (int, optional): The batch size. Defaults to 1.
            num_workers (int, optional): The number of workers. Defaults to 1.
            only_pocket (bool, optional): Use only the binding pocket or not. Defaults to False.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.only_pocket = only_pocket
        self.persistent_workers = True
        super().__init__()

    def prepare_data(self):
        pass

    def setup(self, stage=''):
        self.dt_train = PDBBindDataset(
            root=self.root, stage='train', only_pocket=self.only_pocket)
        self.dt_val = PDBBindDataset(
            root=self.root, stage='val', only_pocket=self.only_pocket)
        self.dt_casf_13 = CASFDataset(root=cfg.data_path, year='13',
                                      only_pocket=cfg.data_use_only_pocket)
        self.dt_casf_16 = CASFDataset(root=cfg.data_path, year='16',
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


if __name__ == '__main__':
    # To Create datasets
    PDBBindDataset(root=cfg.data_path, stage='train',
                   only_pocket=cfg.data_use_only_pocket)
    PDBBindDataset(root=cfg.data_path, stage='val',
                   only_pocket=cfg.data_use_only_pocket)

    CASFDataset(root=cfg.data_path, year='13',
                only_pocket=cfg.data_use_only_pocket)
    CASFDataset(root=cfg.data_path, year='16',
                only_pocket=cfg.data_use_only_pocket)
