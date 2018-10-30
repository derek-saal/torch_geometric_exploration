import shutil
import os
import pickle
from pathlib import Path
import numpy as np
import glob
from torch_geometric.read import read_mr_data

import torch

from torch_geometric.data import (InMemoryDataset, Data)


class MR(InMemoryDataset):
    proj_dir = Path(__file__).parents[2]
    mr_data_dir = str(proj_dir / 'mr_data')

    def __init__(self, root, transform=None, pre_transform=None):
        self.name = 'mr'
        super(MR, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'adj', 'train.index', 'test.index']
        return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for filename in glob.glob(os.path.join(self.mr_data_dir, '*.*')):
            shutil.copy(filename, self.raw_dir)

    def process(self):
        data = read_mr_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
