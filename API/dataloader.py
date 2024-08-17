import copy
from .aug_dataset import AugDataset
from .dataloader_gtrans import DataLoader_GTrans
from .featurizer import featurize_HC, featurize_HC_Aug


def load_data(batch_size, data_root, num_workers=8, load_full_data = True, **kwargs):
    if load_full_data:
        dataset = AugDataset(data_root, mode='train')
        train_set, valid_set, test_set = map(lambda x: copy.deepcopy(x), [dataset] * 3)
        valid_set.change_mode('val')
        test_set.change_mode('test')
        
        train_loader = DataLoader_GTrans(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=featurize_HC_Aug)
        valid_loader = DataLoader_GTrans(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=featurize_HC)
        test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=featurize_HC)
        return train_loader, valid_loader, test_loader
    else:
        dataset = AugDataset(data_root, mode='train', load_full_data = False)
        dataset.change_mode('test')
        test_set = dataset
        
        test_loader = DataLoader_GTrans(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=featurize_HC)
        return test_loader
