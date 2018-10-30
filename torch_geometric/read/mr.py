import os.path as osp
import sys
from itertools import repeat

import torch
from torch_sparse import coalesce

from torch_geometric.data import Data
from torch_geometric.read import read_txt_array
from torch_geometric.utils import remove_self_loops

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_mr_data(folder, prefix):
    """Reads the mr data format.
    ind.{}.x
    """
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'adj', 'train.index', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, adj, train_index, test_index = items
    train_index = torch.arange(train_index.size(0), dtype=torch.long)
    print(train_index.size(0), x.size(0))
    val_index = torch.arange(train_index.size(0), 2 * train_index.size(0) - x.size(0), dtype=torch.long)
    sorted_test_index = test_index.sort()[0]

    x = torch.cat([allx, tx], dim=0)
    y = torch.cat([ally, ty], dim=0).max(dim=1)[1]

    x[test_index] = x[sorted_test_index]
    y[test_index] = y[sorted_test_index]

    train_mask = sample_mask(train_index, num_nodes=y.size(0))
    val_mask = sample_mask(val_index, num_nodes=y.size(0))
    test_mask = sample_mask(test_index, num_nodes=y.size(0))

    coo_adj = adj.tocoo()
    edge_index = torch.tensor([coo_adj.row, coo_adj.col], dtype=torch.long)
    edge_attr = torch.tensor(coo_adj.data).view(-1, 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))

    if name in ('test.index', 'train.index'):
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'adj':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = torch.Tensor(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    # NOTE: There are duplicated edges and self loops in the datasets. Other
    # implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index


def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes,), dtype=torch.uint8)
    mask[index] = 1
    return mask
