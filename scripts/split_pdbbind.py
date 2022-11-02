import argparse

from bgcn_4_pls.data import split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '-d',
                        type=str,
                        required=True,
                        help='Path to the data directory')
    parser.add_argument('-nb_val',
                        type=int,
                        default=1000,
                        help='Number of items from refined set used for validation (defaults to 1000)')
                
    args = parser.parse_args()

    split(data_path=args.data, nb_item_in_val=args.nb_val)
