import os.path as osp
import random
from typing import Dict, List, Tuple

import config as cfg


def load_index(index_path: str, exluded_pdb: List = [], with_cluster: bool=False) -> Dict:
    """Load a csv index into a dictionnary

    Args:
        index_path (str): Path to the csv file
        exluded_pdb (List, optional): PDB ids excluded when loading. Defaults to [].
        with_cluster (bool, optional): Set to True to store target cluster from CASF. Defaults to False.

    Returns:
        Dict: A dictionnary containing PDB ids and affinity target
    """
    set_dict = {}
    with open(index_path, 'r') as f_index:
        lines = f_index.readlines()
        for line in lines:
            if not line.startswith('#'):
                line_tab = line.replace('\n', '').split()
                if line_tab[0] in set_dict.keys():
                    print('Warning: {} already in the dictionnary'.format(
                        line_tab[0]))
                if not line_tab[0] in exluded_pdb:
                    if with_cluster:
                        set_dict[line_tab[0]] = (line_tab[3], line_tab[5])
                    else:
                        set_dict[line_tab[0]] = line_tab[3]
    return set_dict


def dict_to_csv(set_dict: Dict, filepath: str, with_cluster: bool=False):
    """Save a distionnary in a csv file

    Args:
        set_dict (Dict): A dictionnary
        filepath (str): Path where save the created csv file
        with_cluster (bool, optional): Set to True to store target cluster from CASF. Defaults to False.
    """
    with open(filepath, 'w') as csvfile:
        if with_cluster:
            csvfile.write('pdb_id,target,cluster')
        else:
            csvfile.write('pdb_id,target')
        for pdb_id in set_dict.keys():
            if with_cluster:
                csvfile.write('\n{},{},{}'.format(pdb_id, set_dict[pdb_id][0], set_dict[pdb_id][1]))
            else:
                csvfile.write('\n{},{}'.format(pdb_id, set_dict[pdb_id]))


def check_no_overlapping(dict_keys_1: List, dict_keys_2: List) -> bool:
    """Checks that there is no overlap between two lists of PDB ids

    Args:
        dict_keys_1 (List): The first list of PDB ids
        dict_keys_2 (List): The second list of PDB ids

    Returns:
        bool: True if there is no overlap
    """
    for pdb_id in dict_keys_1:
        nb_in_both = 0
        if pdb_id in dict_keys_2:
            nb_in_both += 1
    return nb_in_both == 0


def split_dict(dico: Dict, nb: int) -> Tuple[Dict, Dict]:
    """Split a dictionnary in two, the first contains nb random items,
    the second all others items

    Args:
        dico (Dict): A dictionnary
        nb (int): The number of items move to the first dictionnary

    Returns:
        Tuple[Dict, Dict]: The two created dictionnaries
    """
    keys_list = list(dico.keys())
    random.shuffle(keys_list)
    split_0 = keys_list[:nb]
    dico_split_0 = {}
    dico_split_1 = {}
    for key in dico:
        if key in split_0:
            dico_split_0[key] = dico[key]
        else:
            dico_split_1[key] = dico[key]
    return dico_split_0, dico_split_1


if __name__ == '__main__':

    # Test = Core Set (from CASF 2016 and 2013)
    # Validation = Extract from (Refined Set (from PDBBind 2020) - Test - CoreSet2013)
    # Train = General Set (from PDBBind 2020) - Validation

    val_size_size = 1000

    dict_data_core_16 = load_index(cfg.indexes_path['core'], with_cluster=True)
    dict_data_core_13 = load_index(cfg.indexes_path['core_2013'], with_cluster=True)
    
    dict_data_refined = load_index(
        cfg.indexes_path['refined'], exluded_pdb=list(
            dict_data_core_16.keys()) + list(dict_data_core_13.keys()))

    dict_data_val, dict_refined_for_train = split_dict(
        dict_data_refined, val_size_size)

    dict_data_general = load_index(cfg.indexes_path['general'], exluded_pdb=list(
        dict_data_core_16.keys()) + list(dict_data_refined.keys()) +
        list(dict_data_core_13.keys()))

    dict_data_train = {**dict_data_general, **dict_refined_for_train}

    # Check no overlap between sets
    if not check_no_overlapping(dict_data_core_13.keys(), dict_data_val.keys()):
        print('Warning: Overlapping between Core2013 and Val')

    if not check_no_overlapping(dict_data_core_16.keys(), dict_data_val.keys()):
        print('Warning: Overlapping between Core2016 and Val')

    if not check_no_overlapping(dict_data_val.keys(), dict_data_train.keys()):
        print('Warning: Overlapping between Train and Val')

    data_path = cfg.data_path
    dict_to_csv(dict_data_val, osp.join(data_path, 'val.csv'))
    dict_to_csv(dict_data_train, osp.join(data_path, 'train.csv'))
    dict_to_csv(dict_data_core_16, osp.join(data_path, 'casf16.csv'), with_cluster=True)
    dict_to_csv(dict_data_core_13, osp.join(data_path, 'casf13.csv'), with_cluster=True)
