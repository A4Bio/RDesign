import os
import numpy as np
from tqdm import tqdm
import _pickle as cPickle

import torch.utils.data as data
from .utils import cached_property


class AugDataset(data.Dataset):
    def __init__(self, path='./',  mode='train', load_full_data= True):
        self.path = path
        self.mode = mode
        self.load_full_data = load_full_data
        self.data = self.cache_data[mode]
    
    @cached_property
    def cache_data(self):
        alphabet_set = set(['A', 'U', 'C', 'G'])
        if os.path.exists(self.path):
            data_dict = {'train': [], 'val': [], 'test': []}
            # val and test data
            if self.load_full_data:
                data_list = ['train', 'val', 'test']
            else:
                data_list = ['test']
            for split in data_list:
                data = cPickle.load(open(os.path.join(self.path, split + '_data.pt'), 'rb'))
                for entry in tqdm(data):
                    for key, val in entry['coords'].items():
                        entry['coords'][key] = np.asarray(val)
                    bad_chars = set([s for s in entry['seq']]).difference(alphabet_set)
                    if len(bad_chars) == 0:
                        data_dict[split].append(entry)
            return data_dict
        else:
            raise "no such file:{} !!!".format(self.path)

    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
