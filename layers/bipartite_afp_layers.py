import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax


def glorot(value: Any):
    """From pyg.nn.inits
    """
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


def constant(value: Any, fill_value: float):
    """From pyg.nn.inits
    """
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant(v, fill_value)


def zeros(value: Any):
    """From pyg.nn.inits
    """
    constant(value, 0.)


class GATEConv(MessagePassing):
    """From pyg.nn.models.attentive_fp
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out += self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AFP_GATE_GRUConv_IntraMol(nn.Module):
    """A layer gathering a GATEConv and a GRU layer. First step in attentive_fp atomic embedding
       For intramolecular network
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float,
                 edge_dim: int, heads: int = 1, add_self_loops: bool = False):
        """Construct a GATE_GRUConv layer

        Args:
            in_channels (int): The input channel size
            out_channels (int): The output channels size
            dropout (float): The dropout rate
            edge_dim (int): The edge dimension
            heads (int, optional): Number of heads for the GATEConv part. Defaults to 1.
            add_self_loops (bool, optional): Add self loops. Defaults to False.
        """
        super().__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_channels, out_channels)
        self.gate_conv = GATEConv(out_channels, out_channels, dropout=dropout,
                                  edge_dim=edge_dim)
        self.gru = nn.GRUCell(out_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Process data graph through the layer

        Args:
            x (torch.Tensor): Nodes features
            edge_index (torch.Tensor): Edge index
            edge_attr (torch.Tensor): Edge attributes 

        Returns:
            torch.Tensor: The new nodes attributes
        """
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x)
        return x


class AFP_GATE_GRUConv_InterMol(nn.Module):
    """A layer gathering a GATEConv and a GRU layer. First step in attentive_fp atomic embedding
       For intermolecular network
    """

    def __init__(self, in_channels: Tuple[int], out_channels: int, dropout: float,
                 edge_dim: int, heads: int = 1, add_self_loops: bool = False):
        """Construct a GATE_GRUConv layer

        Args:
            in_channels (Tuple[int]): The input channel sizes
            out_channels (int): The output channels size
            dropout (float): The dropout rate
            edge_dim (int): The edge dimension
            heads (int, optional): Number of heads for the GATEConv part. Defaults to 1.
            add_self_loops (bool, optional): Add self loops. Defaults to False.
        """
        super().__init__()
        self.dropout = dropout
        in_channels_src, in_channels_dst = in_channels
        self.lin1_src = Linear(in_channels_src, out_channels)
        self.lin1_dst = Linear(in_channels_dst, out_channels)
        self.gate_conv = GATEConv(out_channels, out_channels, dropout=dropout,
                                  edge_dim=edge_dim)
        self.gru = nn.GRUCell(out_channels, out_channels)

    def forward(self, x: Tuple[torch.Tensor], edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Process data graph through the layer

        Args:
            x (Tuple[torch.Tensor]): Nodes features
            edge_index (torch.Tensor): Edge index
            edge_attr (torch.Tensor): Edge attributes 

        Returns:
            torch.Tensor: The new nodes attributes
        """
        x_src, x_dst = x
        x_src = F.leaky_relu_(self.lin1_src(x_src))
        x_dst = F.leaky_relu_(self.lin1_dst(x_dst))
        h = F.elu_(self.gate_conv((x_src, x_dst), edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x_dst = self.gru(h, x_dst)
        return x_dst


class AFP_GATGRUConv_IntraMol(nn.Module):
    """A layer gathering a GATConv and a GRU layer. Following step in attentive_fp atomic embedding
       For intramolecular network
    """

    def __init__(self, in_channels: int, out_channels_gat: int,
                 out_channels_gru: int, dropout: float, edge_dim: int,
                 heads: int = 1, add_self_loops: bool = False):
        """Construct a GATGruConv layer

        Args:
            in_channels (int): Input channels size
            out_channels_gat (int): GATConv outpout channels size (GRY input size)
            out_channels_gru (int): output channels size
            dropout (float): Dropout rate
            edge_dim (int): Edge dimension
            heads (int, optional): NUmber of heads for the GATConv part. Defaults to 1.
            add_self_loops (bool, optional): Add self loops. Defaults to False.
        """
        super().__init__()
        self.dropout = dropout
        self.gat_conv = pyg.nn.GATConv(in_channels, out_channels_gat,
                                       dropout=dropout,
                                       edge_dim=edge_dim,
                                       add_self_loops=add_self_loops,
                                       heads=heads)
        self.gru = nn.GRUCell(out_channels_gru, out_channels_gru)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """Process data graph through the layer

        Args:
            x (torch.Tensor): Nodes features
            edge_index (torch.Tensor): Edge index 

        Returns:
            torch.Tensor: The new nodes attributes
        """
        h = F.elu_(self.gat_conv(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x)
        return x


class AFP_GATGRUConv_InterMol(nn.Module):
    """A layer gathering a GATConv and a GRU layer. Following step in attentive_fp atomic embedding
       For intermolecular network
    """

    def __init__(self, in_channels: Tuple[int], out_channels_gat: int,
                 out_channels_gru: int, dropout: float, edge_dim: int,
                 heads: int = 1, add_self_loops: bool = False):
        """Construct a GATGruConv layer

        Args:
            in_channels (Tuple[int]): Input channels sizes
            out_channels_gat (int): GATConv outpout channels size (GRY input size)
            out_channels_gru (int): output channels size
            dropout (float): Dropout rate
            edge_dim (int): Edge dimension
            heads (int, optional): NUmber of heads for the GATConv part. Defaults to 1.
            add_self_loops (bool, optional): Add self loops. Defaults to False.
        """
        super().__init__()
        self.dropout = dropout
        self.gat_conv = pyg.nn.GATConv(in_channels, out_channels_gat,
                                       dropout=dropout,
                                       edge_dim=edge_dim,
                                       add_self_loops=add_self_loops,
                                       heads=heads)
        self.gru = nn.GRUCell(out_channels_gru, out_channels_gru)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """Process data graph through the layer

        Args:
            x (torch.Tensor): Nodes features
            edge_index (torch.Tensor): Edge index 

        Returns:
            torch.Tensor: The new nodes attributes
        """
        x_src, x_dst = x
        h = F.elu_(self.gat_conv((x_src, x_dst), edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x_dst = self.gru(h, x_dst)
        return x_dst


class AFP_GATGRUConvMol(nn.Module):
    """A layer gathering GATConv and GRU, corresponding to the attentive_fp molecular embedding
    """

    def __init__(self, in_channels: int, out_channels_gat: int,
                 out_channels_gru: int, dropout: float, edge_dim: int,
                 heads: int = 1, add_self_loops: bool = False):
        """Construct a GATGruConv layer for molecular embedding

        Args:
            in_channels (int): Input channels size
            out_channels_gat (int): GATConv outpout channels size (GRY input size)
            out_channels_gru (int): output channels size
            dropout (float): Dropout rate
            edge_dim (int): Edge dimension
            heads (int, optional): NUmber of heads for the GATConv part. Defaults to 1.
            add_self_loops (bool, optional): Add self loops. Defaults to False.
        """
        super().__init__()
        self.dropout = dropout
        self.gat_conv = pyg.nn.GATConv(in_channels, out_channels_gat, dropout=dropout,
                                       edge_dim=edge_dim, add_self_loops=add_self_loops, heads=heads)
        self.gru = nn.GRUCell(out_channels_gru, out_channels_gru)

    def forward(self, x: torch.Tensor, out: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """Process data graph through the layer

        Args:
            x (torch.Tensor): The embedding of each atom, from previous layers
            out (torch.Tensor): The molecule atomic fingerprint from gloabl add pooling
            edge_index (torch.Tensor): The edge index

        Returns:
            torch.Tensor: The molecular fingerprint
        """
        h = F.elu_(self.gat_conv((x, out), edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.gru(h, out).relu_()
        return out


def molecular_embedding(x: torch.Tensor,
                        batch: torch.Tensor) -> Tuple[torch.Tensor,
                                                      torch.Tensor]:
    """From all atomic nodes embedding produce a molecular fingeprint, using a gloabl_add_pool

    Args:
        x (torch.Tensor): Atomic nodes embedding
        batch (torch.Tensor): Batch

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The molecular fingerprint, the edge_index
    """
    # Molecule Embedding:
    row = torch.arange(batch.size(0), device=batch.device)
    edge_index = torch.stack([row, batch], dim=0)
    out = pyg.nn.global_add_pool(x, batch).relu_()
    return out, edge_index


def molecular_pooling(x_dict: Dict, edge_index_dict: Dict,
                      batch: Dict) -> Tuple[Dict, Dict]:
    """Process the molecular embedding for both the ligand and the protein

    Args:
        x_dict (torch.Tensor): Dictionnary containing nodes features
        edge_index_dict (torch.Tensor): Dictionnary containing edge indexes
        batch (torch.Tensor): Batch

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Dictionnary with added molecular fingerprint
    """
    x, edge_index = molecular_embedding(
        x_dict['protein_atoms'], batch['protein_atoms'])
    x_dict['pa_embedding'] = x
    edge_index_dict['pa_embedding'] = edge_index

    x, edge_index = molecular_embedding(
        x_dict['ligand_atoms'], batch['ligand_atoms'])
    x_dict['la_embedding'] = x
    edge_index_dict['la_embedding'] = edge_index

    return x_dict, edge_index_dict
