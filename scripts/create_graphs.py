import argparse

from hgcn_4_pls.data import CASFDataset, DockingPower_Dataset, PDBBindDataset

if __name__ == '__main__':
    # To Create datasets
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '-d',
                        type=str,
                        required=True,
                        help='Path to the data directory')
    parser.add_argument('-pocket',
                        action='store_true',
                        help='Flag allowing to consider only the binding pocket as defined by PDBBind')
    parser.add_argument('-cutoff', '-c',
                        type=float,
                        help='The cutoff to consider a link between a protein-ligand atom pair (defaults to 4.0)',
                        default=4.0)
    parser.add_argument('-docking_power', '-dp',
                        action='store_true',
                        help='Flag allowing to create the docking power dataset')
    args = parser.parse_args()

    data_root = args.data
    atomic_distance_cutoff = args.cutoff
    only_pocket = args.pocket

    PDBBindDataset(root=data_root,
                   stage='train',
                   atomic_distance_cutoff=atomic_distance_cutoff,
                   only_pocket=only_pocket)
    PDBBindDataset(root=args.data,
                   stage='val',
                   atomic_distance_cutoff=atomic_distance_cutoff,
                   only_pocket=only_pocket)

    CASFDataset(root=data_root,
                year='13',
                atomic_distance_cutoff=atomic_distance_cutoff,
                only_pocket=only_pocket)
    CASFDataset(root=data_root,
                year='16',
                atomic_distance_cutoff=atomic_distance_cutoff,
                only_pocket=only_pocket)

    if args.docking_power:
        DockingPower_Dataset(root=data_root,
                             year='16',
                             atomic_distance_cutoff=atomic_distance_cutoff,
                             only_pocket=only_pocket)
