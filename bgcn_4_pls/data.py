import multiprocessing as mp
import os
import os.path as osp
import re
from dataclasses import dataclass, field
from typing import Dict, List, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric as pyg
from biopandas.mol2 import PandasMol2, split_multimol2
from biopandas.pdb import PandasPdb

import bgcn_4_pls.featurizer as f_atm


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


def read_decoy_rmsd(path: str) -> pd.DataFrame:
    """Load the docking power decoys' rmsd into a Dataframe

    Args:
        path (str): Path to the csv containing the docking power decoys' rmsd

    Returns:
        pd.DataFrame: The Dataframe
    """
    lst_for_df = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            find = re.search("^#", line)
            if find is None:
                line_tab = line.split()
                lst_for_df.append(line_tab)
    df = pd.DataFrame(lst_for_df, columns=['#code', 'rmsd'])
    return df


def clean_backbone_str(lines: List[str]) -> List[str]:
    """Clear the MOL2 ATOM lines by removing "BACKBONE" at the end of the lines

    Args:
        lines (List[str]): The MOL2 ATOM lines

    Returns:
        List[str]: The cleaned lines
    """
    for i in range(len(lines)):
        lines[i] = lines[i].replace("BACKBONE\n", '\n')
    return lines


def split_multi_mol2(mol2_path: str, rmsd_path: str) -> Dict:
    """For a given docking power complex, split into a Dict all decoys

    Args:
        mol2_path (str): Path to the decoys mol2 file
        rmsd_path (str): Path to the decoys rmsd file

    Returns:
        Dict: A dictionnary
    """
    mol2_dict = {}
    pdmol = PandasMol2()
    df_rmsd = read_decoy_rmsd(rmsd_path)
    filepath, filename = osp.split(mol2_path)
    for mol2 in split_multimol2(mol2_path):
        mol2[1] = clean_backbone_str(mol2[1])
        pdmol.read_mol2_from_list(mol2_lines=mol2[1], mol2_code=mol2[0])
        decoy_path = osp.join(filepath, "{}.mol2".format(pdmol.code))
        rmsd = df_rmsd.loc[df_rmsd['#code'] == pdmol.code]['rmsd']
        mol2_dict[pdmol.code] = (decoy_path, float(rmsd.item()))
        if not osp.isfile(decoy_path):
            with open(decoy_path, 'w') as f_decoy:
                f_decoy.write(pdmol.mol2_text)
    return mol2_dict


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
        ligand_path (str): Path to the liagnd file (MOL2/PDB)
        target (float, optional): Affinity target. Defaults to None.
        cluster (int, optional): Cluster ID for ranking. Defaults to None.
        pdb_id (int, optional): PDB ID for casf output. Defaults to None.
        cutoff (float): The maximal distance between two atoms to connect them with an edge. Defaults to None.
        rmsd (float, optional): Used for docking power. Defaults to 0.0.
        decoy_id (str, optional): Contains the ligand id for docking power. Defaults to None.

    Returns:
        pyg.data.HeteroData, Containing several sets of nodes, and different sets of
            edges that can link nodes of the same set or of different sets
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


def process_graph_from_files(protein_path: str,
                             ligand_path: str,
                             atomic_distance_cutoff: float,
                             target: float = None,
                             cluster: int = None,
                             pdb_id: str = None,
                             rmsd: float = 0.0,
                             decoy_id: str = None) -> pyg.data.HeteroData:
    """Build a Bipartite Graph from a protein (whole or pocket) and a ligand files

    Args:
        protein_path (str): Path to the protien PDB (or the pocket)
        ligand_path (str): Path to the ligand MOL2/PDB
        atomic_distance_cutoff (float): The cutoff to consider a link between a protein-ligand atom pair
        target (float, optional): The target affinity. Defaults to None.
        cluster (int, optional): Cluster ID for ranking. Defaults to None.
        pdb_id (int, optional): PDB ID for casf output. Defaults to None.
        rmsd (float, optional): Used for docking power. Defaults to 0.0.
        decoy_id (str, optional): Contains the ligand id for docking power. Defaults to None.

    Returns:
        pyg.data.HeteroData: The bipartite graph
    """
    protein_dir, protein_name = osp.split(protein_path)

    clean_protein_path = 'clean_' + protein_name
    clean_protein_path = osp.join(protein_dir, clean_protein_path)
    if not osp.exists(clean_protein_path):
        clean_pdb(protein_path, clean_protein_path)
    g = create_pyg_graph(clean_protein_path,
                         ligand_path,
                         target,
                         cluster, pdb_id,
                         atomic_distance_cutoff,
                         rmsd,
                         decoy_id)
    return g


def process_graph_from_dir(dir_path: str,
                           atomic_distance_cutoff: float,
                           only_pocket: bool = False,
                           target: float = None,
                           cluster: int = None,
                           pdb_id: str = None,
                           rmsd: float = 0.0,
                           decoy_id: str = None) -> str:
    """Create a graph from PDBBind

    Args:
        dir_path (str): Path to the directory containing raw files (protein/pocket and ligand).
        atomic_distance_cutoff (float): Cutoff for inter-atomic distance.
        only_pocket (bool, optional): Use only the binding pocket or not. Defaults to False.
        target (float, optional): The target affinity. Defaults to None.
        cluster (int, optional): Cluster ID for ranking. Defaults to None.
        pdb_id (int, optional): PDB ID for casf output. Defaults to None.
        rmsd (float, optional): Used for docking power. Defaults to 0.0.
        decoy_id (str, optional): Contains the ligand id for docking power. Defaults to None.

    Returns:
        str: Path where the graph is saved
    """
    pdb_id = dir_path.split('/')[-1]
    protein_path = osp.join(
        dir_path, pdb_id + '_pocket.pdb') if only_pocket else osp.join(dir_path, pdb_id + '_protein.pdb')
    ligand_path = osp.join(dir_path, pdb_id + '_ligand.mol2')
    g = process_graph_from_files(protein_path,
                                 ligand_path,
                                 atomic_distance_cutoff,
                                 target,
                                 cluster,
                                 pdb_id,
                                 rmsd,
                                 decoy_id)
    return g


def process_and_save_graph_from_files(protein_path: str,
                                      ligand_path: str,
                                      processed_filename: str,
                                      atomic_distance_cutoff: float,
                                      target: float = None,
                                      cluster: int = None,
                                      pdb_id: str = None,
                                      rmsd: float = 0.0,
                                      decoy_id: str = None) -> str:
    """Create a graph from PDBBind and save it in processed_filename

    Args:
        processed_filename (str): Path to the processed file.
        atomic_distance_cutoff (float): Cutoff for inter-atomic distance.
        dir_path (str, optional): Path to the directory containing raw files. Defaults to None.
        target (float, optional): The target affinity. Defaults to None.
        cluster (int, optional): Cluster ID for ranking. Defaults to None.
        pdb_id (int, optional): PDB ID for casf output. Defaults to None.
        rmsd (float, optional): Used for docking power. Defaults to 0.0.
        decoy_id (str, optional): Contains the ligand id for docking power. Defaults to None.

    Returns:
        str: Path where the graph is saved
    """
    g = process_graph_from_files(protein_path,
                                 ligand_path,
                                 atomic_distance_cutoff,
                                 target,
                                 cluster,
                                 pdb_id,
                                 rmsd,
                                 decoy_id)
    torch.save(g, processed_filename)
    return processed_filename


def process_and_save_graph_from_dir(dir_path: str,
                                    processed_filename: str,
                                    atomic_distance_cutoff: float,
                                    only_pocket: bool = False,
                                    target: float = None,
                                    cluster: int = None,
                                    pdb_id: str = None,
                                    rmsd: float = 0.0,
                                    decoy_id: str = None) -> str:
    """Create a graph from PDBBind and save it in processed_filename

    Args:
        processed_filename (str): Path to the processed file.
        atomic_distance_cutoff (float): Cutoff for inter-atomic distance.
        dir_path (str, optional): Path to the directory containing raw files. Defaults to None.
        only_pocket (bool, optional): Use only the binding pocket or not. Defaults to False.
        target (float, optional): The target affinity. Defaults to None.
        cluster (int, optional): Cluster ID for ranking. Defaults to None.
        pdb_id (int, optional): PDB ID for casf output. Defaults to None.
        rmsd (float, optional): Used for docking power. Defaults to 0.0.
        decoy_id (str, optional): Contains the ligand id for docking power. Defaults to None.

    Returns:
        str: Path where the graph is saved
    """

    g = process_graph_from_dir(dir_path,
                               atomic_distance_cutoff,
                               only_pocket,
                               target,
                               cluster,
                               pdb_id,
                               rmsd,
                               decoy_id)
    torch.save(g, processed_filename)
    return processed_filename


@dataclass
class PDBBindDataset(pyg.data.InMemoryDataset):
    """Torch Geometric Dataset, used for Train and Val set
    """
    root: str  # The data directory
    stage: str  # train or val
    atomic_distance_cutoff: float  # Cutoff for inter-atomic distance.
    # Use only the binding pocket or not. Defaults to False.
    only_pocket: bool = field(default=False)

    # post init function
    def __post_init__(self):
        self.prefix = 'pocket' if self.only_pocket else 'protein'
        self.df = pd.read_csv(
            osp.join(self.root, '{}.csv'.format(self.stage))).set_index('pdb_id')
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        filename_list = []
        for pdb_id in self.df.index:
            filename_list.append(pdb_id)
        return filename_list

    @property
    def processed_file_names(self):
        return ['{}_{}_{}.pt'.format(self.prefix,
                                     self.atomic_distance_cutoff,
                                     self.stage)]

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
        pool = mp.Pool(mp.cpu_count())
        data_path_list = list(pool.starmap(
            process_and_save_graph_from_dir, pool_args))
        data_list = []
        for p in data_path_list:
            data_list.append(torch.load(p))
            os.remove(p)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


@dataclass
class CASFDataset(pyg.data.InMemoryDataset):
    """Torch Geometric Dataset, used for Test CASF13 and CASF16
    """
    root: str  # The data directory
    year: str  # The CASF year version
    atomic_distance_cutoff: float  # Cutoff for inter-atomic distance.
    # Use only the binding pocket or not. Defaults to False.
    only_pocket: bool = field(default=False)

    # post init function
    def __post_init__(self):
        self.prefix = 'pocket' if self.only_pocket else 'protein'
        self.df = pd.read_csv(
            osp.join(self.root, 'casf{}.csv'.format(self.year))).set_index('pdb_id')
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        filename_list = []
        for pdb_id in self.df.index:
            filename_list.append(pdb_id)
        return filename_list

    @property
    def processed_file_names(self):
        return ['{}_{}_casf_{}.pt'.format(self.prefix,
                                          self.atomic_distance_cutoff,
                                          self.year)]

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
        pool = mp.Pool(mp.cpu_count())
        data_path_list = list(pool.starmap(
            process_and_save_graph_from_dir, pool_args))
        data_list = []
        for p in data_path_list:
            data_list.append(torch.load(p))
            os.remove(p)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


@dataclass
class DockingPower_Dataset(pyg.data.InMemoryDataset):
    """Torch Geometric Dataset, used for Docking Power
    """

    root: str  # The data directory
    year: str  # The CASF year version
    atomic_distance_cutoff: float  # Cutoff for inter-atomic distance.
    # Use only the binding pocket or not. Defaults to False.
    only_pocket: bool = field(default=False)

    def __post_init__(self):
        self.prefix = 'pocket' if self.only_pocket else 'protein'
        self.decoy_path = osp.join(self.root, 'decoys_docking')
        self.df = pd.read_csv(
            osp.join(self.root, 'casf{}.csv'.format(self.year))).set_index('pdb_id')
        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        filename_list = []
        for pdb_id in self.df.index:
            filename_list.append(pdb_id)
        return filename_list

    @property
    def processed_file_names(self):
        return ['{}_{}_dockingP.pt'.format(self.prefix,
                                           self.atomic_distance_cutoff)]

    def process(self):
        i = 0
        print('\tDocking Power ', self.year)
        pool_args = []
        nb_pdb = 0
        for raw_path in self.raw_paths:
            nb_pdb += 1
            pdb_id = raw_path.split('/')[-1]
            decoys_mol2 = osp.join(
                self.decoy_path, "{}_decoys.mol2".format(pdb_id))
            decoys_rmsd = osp.join(
                self.decoy_path, "{}_rmsd.dat".format(pdb_id))
            mol2_dict = split_multi_mol2(decoys_mol2, decoys_rmsd)
            filename = osp.join(self.processed_dir,
                                'TMP_DP_{}{}_data_{}.pt'.format(self.prefix,
                                                                self.year, i))
            protein_path = osp.join(
                raw_path, pdb_id + '_pocket.pdb') if self.only_pocket else osp.join(raw_path, pdb_id + '_protein.pdb')
            ligand_path = osp.join(raw_path, "{}_ligand.mol2".format(pdb_id))
            pool_args.append((protein_path,
                              ligand_path,
                              filename,
                              self.atomic_distance_cutoff,
                              self.only_pocket,
                              pdb_id,
                              0.0,
                              '{}_ligand'.format(pdb_id)))
            i += 1
            for key in mol2_dict.keys():
                filename = osp.join(self.processed_dir,
                                    'TMP_DP_{}{}_data_{}.pt'.format(self.prefix,
                                                                    self.year,
                                                                    i))
                pool_args.append((protein_path,
                                  mol2_dict[key][0],
                                  filename,
                                  self.atomic_distance_cutoff,
                                  self.only_pocket,
                                  pdb_id,
                                  mol2_dict[key][1],
                                  key))
                i += 1
        pool = mp.Pool(mp.cpu_count())
        data_path_list = list(pool.starmap(process_and_save_graph_from_files,
                                           pool_args))
        data_list = []
        for p in data_path_list:
            data_list.append(torch.load(p))
            os.remove(p)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


@dataclass
class PDBBindDataModule(pl.LightningDataModule):
    """PyTorch Lightning Datamodule
    """
    root: str  # Path to the data directory.
    atomic_distance_cutoff: float  # Cutoff for inter-atomic distance.
    batch_size: int = field(default=1)  # The batch size. Defaults to 1.
    # The number of workers. Defaults to 1.
    num_workers: int = field(default=1)
    # Use only the binding pocket or not. Defaults to False.
    only_pocket: bool = field(default=False)
    # Use persistent workers in dataloader
    persistent_workers: bool = field(default=True)

    def __post_init__(self):
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

        self.dt_casf_13 = CASFDataset(root=self.root, year='13',
                                      atomic_distance_cutoff=self.atomic_distance_cutoff,
                                      only_pocket=self.only_pocket)
        self.dt_casf_16 = CASFDataset(root=self.root, year='16',
                                      atomic_distance_cutoff=self.atomic_distance_cutoff,
                                      only_pocket=self.only_pocket)

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

    def casf_dataloader(self, casf_version: Union[int, str]):
        if isinstance(casf_version, int):
            casf_version = str(casf_version)
        if casf_version == '13':
            return self.casf_13_dataloader()
        elif casf_version == '16':
            return self.casf_16_dataloader()
        return None

    def test_dataloader(self):
        return pyg.loader.DataLoader(self.dt_casf_16,
                                     batch_size=1,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)


@dataclass
class DockingPowerDataModule(pl.LightningDataModule):
    root: str  # Path to the data directory.
    atomic_distance_cutoff: float  # Cutoff for inter-atomic distance.
    batch_size: int = field(default=1)  # The batch size. Defaults to 1.
    # The number of workers. Defaults to 1.
    num_workers: int = field(default=1)
    # Use only the binding pocket or not. Defaults to False.
    only_pocket: bool = field(default=False)
    # Use persistent workers in dataloader
    persistent_workers: bool = field(default=True)

    def __post_init__(self):
        super().__init__()

    def prepare_data(self):
        pass

    def setup(self, stage=''):
        self.dt_docking_power = DockingPower_Dataset(root=self.root,
                                                     year='16',
                                                     atomic_distance_cutoff=self.atomic_distance_cutoff,
                                                     only_pocket=self.only_pocket)

    def power_docking_dataloader(self):
        return pyg.loader.DataLoader(self.dt_docking_power,
                                     batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers,
                                     persistent_workers=self.persistent_workers)
