from typing import Dict, List, Tuple

import torch
import torch_geometric as pyg
from layers.bipartite_afp_layers import (AFP_GATE_GRUConv_InterMol,
                                         AFP_GATE_GRUConv_IntraMol,
                                         AFP_GATGRUConv_InterMol,
                                         AFP_GATGRUConv_IntraMol,
                                         AFP_GATGRUConvMol, molecular_pooling)
from torch_geometric.nn import HeteroConv

NB_ATOM_FTS = 39
NB_INTRA_BOND_FTS = 6
NB_INTER_BOND_FTS = 6


def get_het_conv_first_layer(hidden_channels_pa: int,
                             hidden_channels_la: int,
                             dropout: float = 0.0) -> Dict:
    """Produce a dictionnary for heterogeneous graph convolution. Used to create the first layer of our network

    Args:
        hidden_channels_pa (int): The input channels size for the protein
        hidden_channels_la (int): The input channels size for the ligand
        dropout (float, optional): Dropout rate. Defaults to 0.0.

    Returns:
        Dict: A dictionnary containing the layer
    """
    conv_dict = {}

    conv_dict[('protein_atoms', 'linked_to', 'protein_atoms')] = AFP_GATE_GRUConv_IntraMol(
        NB_ATOM_FTS, hidden_channels_pa, dropout=dropout,
        edge_dim=NB_INTRA_BOND_FTS)

    conv_dict[('ligand_atoms', 'linked_to', 'ligand_atoms')] = AFP_GATE_GRUConv_IntraMol(
        NB_ATOM_FTS, hidden_channels_la, dropout=dropout,
        edge_dim=NB_INTRA_BOND_FTS)

    conv_dict[('ligand_atoms', 'interact_with', 'protein_atoms')] = AFP_GATE_GRUConv_InterMol(
        (NB_ATOM_FTS, NB_ATOM_FTS), hidden_channels_pa, dropout=dropout,
        edge_dim=NB_INTER_BOND_FTS)

    conv_dict[('protein_atoms', 'interact_with', 'ligand_atoms')] = AFP_GATE_GRUConv_InterMol(
        (NB_ATOM_FTS, NB_ATOM_FTS), hidden_channels_la, dropout=dropout,
        edge_dim=NB_INTER_BOND_FTS)
    return conv_dict


def get_het_conv_layer(list_hidden_channels_pa: List[int],
                       list_hidden_channels_la: List[int],
                       heads: int = 1,
                       dropout: float = 0.0) -> Dict:
    """Produce a dictionnary for heterogeneous graph convolution. Used to create the following layers of our network

    Args:
        list_hidden_channels_pa (List[int]): The channels sizes for the protein
        list_hidden_channels_la (List[int]): The channels sizes for the ligand
        heads (int, optional): NUmber of heads. Defaults to 1.
        dropout (float, optional): Dropout rate. Defaults to 0.0.

    Returns:
        Dict: A dictionnary containing the layer
    """
    list_dico = []
    for in_channels_pa, in_channels_la, hidden_channels_pa, hidden_channels_la in zip(list_hidden_channels_pa[:-1], list_hidden_channels_la[:-1], list_hidden_channels_pa[1:], list_hidden_channels_la[1:]):
        conv_dict = {}

        conv_dict[('protein_atoms', 'linked_to', 'protein_atoms')] = AFP_GATGRUConv_IntraMol(
            in_channels_pa, hidden_channels_pa, hidden_channels_pa,
            edge_dim=None, heads=heads, dropout=dropout)

        conv_dict[('ligand_atoms', 'linked_to', 'ligand_atoms')] = AFP_GATGRUConv_IntraMol(
            in_channels_la, hidden_channels_la, hidden_channels_la,
            edge_dim=None, heads=heads, dropout=dropout)

        conv_dict[('ligand_atoms', 'interact_with', 'protein_atoms')] = AFP_GATGRUConv_InterMol(
            (in_channels_la,
             in_channels_pa), hidden_channels_pa, hidden_channels_pa,
            edge_dim=None, heads=heads, dropout=dropout)

        conv_dict[('protein_atoms', 'interact_with', 'ligand_atoms')] = AFP_GATGRUConv_InterMol(
            (in_channels_pa,
             in_channels_la), hidden_channels_la, hidden_channels_la,
            edge_dim=None, heads=heads, dropout=dropout)
        list_dico.append(conv_dict)
    return list_dico


class HeteroAFP_Atomic(torch.nn.Module):
    """The atomic embedding part
    """

    def __init__(self,
                 list_hidden_channels_pa: List[int],
                 list_hidden_channels_la: List[int],
                 num_layers: int,
                 dropout: float,
                 heads: int,
                 hetero_aggr: str = 'sum',
                 verbose: bool = False):
        """Construct the atomic embedding part

        Args:
            list_hidden_channels_pa (List[int]): The channels sizes for the protein
            list_hidden_channels_la (List[int]): The channels sizes for the ligand
            num_layers (int): The number of layers
            dropout (float): Dropout rate
            heads (int): Number of heads
            hetero_aggr (str, optional): How the hetero aggregation is did. Defaults to 'sum'.
            verbose (bool, optional): Verbose. Defaults to False.
        """
        super().__init__()
        first_layer_dict = get_het_conv_first_layer(
            list_hidden_channels_pa[0], list_hidden_channels_la[0], dropout)
        list_other_layer_dict = get_het_conv_layer(
            list_hidden_channels_pa, list_hidden_channels_la, heads, dropout)
        if verbose:
            print(first_layer_dict)
            print('--\n')
            for dico in list_other_layer_dict:
                print(dico)
                print('--\n')
        self.conv_list = torch.nn.ModuleList(
            [HeteroConv(first_layer_dict, aggr=hetero_aggr)])
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            self.conv_list.append(HeteroConv(
                list_other_layer_dict[i], aggr=hetero_aggr))

    def forward(self, x: Dict, edge_index: Dict,
                edge_attr: Dict) -> Dict:
        """Forward

        Args:
            x (Dict): Dictionnary of node features
            edge_index (Dict): Dictionnary of edge_index
            edge_attr (Dict): Dictionnary of edge attributes

        Returns:
            Dict: The atomic embedding
        """
        x_dict = self.conv_list[0](x, edge_index, edge_attr)
        x = {key: x.relu() for key, x in x_dict.items()}
        edge_attr = {key: edge_attr[key] for key in [
            ('ligand_atoms', 'interact_with', 'protein_atoms'),
            ('protein_atoms', 'interact_with', 'ligand_atoms')]}
        for conv in self.conv_list[1:]:
            x_dict = conv(x, edge_index)
            x = {key: x.relu() for key, x in x_dict.items()}
        return x


class AFP_Hetero_Molecular(torch.nn.Module):
    """The molecular embedding part
    """

    def __init__(self,
                 hidden_channels_pa: int,
                 hidden_channels_la: int,
                 out_channels_pa: int,
                 out_channels_la: int,
                 num_timesteps: int,
                 dropout: float,
                 heads: int):
        """Construct the molecular embedding part

        Args:
            hidden_channels_pa (int): The size of channels for the protein part
            hidden_channels_la (int): The size of channels for the ligand part
            out_channels_pa (int): The size of output channels for the protein part
            out_channels_la (int): The size of output channels for the ligand part
            num_timesteps (int): Number of timestep for molecular embedding
            dropout (float): Dropout rate
            heads (int): Number of heads
        """
        super().__init__()
        self.gcn_pa = self.gcn_la = None
        self.lin_pa = self.lin_la = None
        self.num_timesteps = num_timesteps

        self.gcn_pa = AFP_GATGRUConvMol(
            hidden_channels_pa, hidden_channels_pa, hidden_channels_pa,
            dropout, None, heads)
        self.lin_pa = torch.nn.Linear(hidden_channels_pa, out_channels_pa)

        self.gcn_la = AFP_GATGRUConvMol(
            hidden_channels_la, hidden_channels_la, hidden_channels_la,
            dropout, None, heads)
        self.lin_la = torch.nn.Linear(hidden_channels_la, out_channels_la)

    def forward(self, x_dict: Dict,
                edge_index_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward

        Args:
            x (Dict): Dictionnary of node features
            edge_index (Dict): Dictionnary of edge_index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The molecular embedding of (protein, ligand)
        """
        for _ in range(self.num_timesteps):
            x_dict['pa_embedding'] = self.gcn_pa(x_dict['protein_atoms'],
                                                 x_dict['pa_embedding'],
                                                 edge_index_dict['pa_embedding'])
            x_dict['la_embedding'] = self.gcn_la(x_dict['ligand_atoms'],
                                                 x_dict['la_embedding'],
                                                 edge_index_dict['la_embedding'])
        y_pa = self.lin_pa(x_dict['pa_embedding'])
        y_la = self.lin_la(x_dict['la_embedding'])
        return y_pa, y_la


class BG_LPS(torch.nn.Module):
    """The Bipartite Graph for Ligand-Protein Scoring network
    """

    def __init__(self,
                 list_hidden_channels_pa: List[int],
                 list_hidden_channels_la: List[int],
                 num_layers: int,
                 hetero_aggr: str,
                 mlp_channels: List[int],
                 num_timesteps: int,
                 dropout: float,
                 heads: int,
                 verbose: bool = False):
        """Construct the model

        Args:
            list_hidden_channels_pa (List[int]): The sizes of channels for the protein part
            list_hidden_channels_la (List[int]): The sizes of channels for the ligand part
            num_layers (int): The number of layers
            hetero_aggr (str): How the hetero aggregation is did
            mlp_channels (List[int]): List of final MLP channels size
            num_timesteps (int): Number of timestep for molecular embedding
            dropout (float): Dropout rate
            heads (int): Number of heads
            verbose (bool, optional): Verbose. Defaults to False.
        """
        super().__init__()
        self.gcn_atm = HeteroAFP_Atomic(
            list_hidden_channels_pa=list_hidden_channels_pa,
            list_hidden_channels_la=list_hidden_channels_la,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads,
            hetero_aggr=hetero_aggr,
            verbose=verbose)
        self.gcn_mol = AFP_Hetero_Molecular(hidden_channels_pa=list_hidden_channels_pa[-1],
                                            hidden_channels_la=list_hidden_channels_la[-1],
                                            out_channels_pa=list_hidden_channels_pa[-1],
                                            out_channels_la=list_hidden_channels_la[-1],
                                            num_timesteps=num_timesteps,
                                            dropout=dropout,
                                            heads=heads)

        self.mlp = pyg.nn.models.MLP(
            channel_list=mlp_channels, dropout=dropout)

    def forward(self, x: Dict, edge_index: Dict, edge_attr: Dict,
                batch: Dict) -> torch.Tensor:
        """Forward

        Args:
            x (Dict): Dictionnary of atomic node features
            edge_index (Dict): Dictionnary of edge index
            edge_attr (Dict): Dictionnary of edge attributes
            batch (Dict): Dictionnary of batches

        Returns:
            torch.Tensor: The score
        """
        y_dict = self.gcn_atm(x, edge_index, edge_attr)
        mol_x_dict, mol_edge_index_dict = molecular_pooling(
            y_dict, edge_index, batch)
        x_pa, x_la = self.gcn_mol(mol_x_dict, mol_edge_index_dict)
        x_fp = torch.cat((x_pa, x_la), axis=1)
        return self.mlp(x_fp)

    def get_nb_parameters(self, only_trainable: bool = False) -> int:
        """Get the number of network's parameters

        Args:
            only_trainable (bool, optional): Consider only trainable parameters. Defaults to False.

        Returns:
            int: The number of parameters
        """
        nb = 0
        if only_trainable:
            nb += sum(p.numel()
                      for p in self.gcn_atm.parameters() if p.requires_grad)
            nb += sum(p.numel()
                      for p in self.gcn_mol.parameters() if p.requires_grad)
            nb += sum(p.numel()
                      for p in self.mlp.parameters() if p.requires_grad)
            return nb
        nb += sum(p.numel() for p in self.gcn_atm.parameters())
        nb += sum(p.numel() for p in self.gcn_mol.parameters())
        nb += sum(p.numel() for p in self.mlp.parameters())
        return nb
