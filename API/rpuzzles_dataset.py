import os
import numpy as np
from tqdm import tqdm
import _pickle as cPickle

import torch.utils.data as data
from .utils import cached_property


class RPuzzlesDataset(data.Dataset):
    def __init__(self, path='./'):
        self.path = path
        self.data = self.cache_data
    
    @cached_property
    def cache_data(self):
        alphabet_set = set(['A', 'U', 'C', 'G'])
        rna_puzzles_data = []
        if os.path.exists(self.path):
            data = cPickle.load(open(os.path.join(self.path), 'rb'))
            for entry in tqdm(data):
                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)
                bad_chars = set([s for s in entry['seq']]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    rna_puzzles_data.append(entry)
            return rna_puzzles_data
        else:
            raise "no such file:{} !!!".format(self.path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]