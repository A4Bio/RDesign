import torch


class DataLoader_GTrans(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, **kwargs):
        super(DataLoader_GTrans, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,**kwargs)
        self.featurizer = collate_fn