import logging
import pickle
import json
import torch
import os.path as osp
from parser import create_parser

import warnings
warnings.filterwarnings('ignore')

from methods import RDesign
from API import Recorder
from utils import *


class Exp:
    def __init__(self, args):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        print_log(output_namespace(self.args))
    
    def _acquire_device(self):
        if self.args.use_gpu: device = torch.device('cuda:0')
        else: device = torch.device('cpu')
        return device
    
    def _preparation(self):
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data, only using self._get_data() for training, otherwise just comment it
        self._get_data()
        # build the method
        self._build_method()

    # def _build_method(self):
    #     steps_per_epoch = 1
    #     # If training, uncomment next line
    #     # steps_per_epoch = len(self.train_loader)
    #     self.method = RDesign(self.args, self.device, steps_per_epoch)

    # def _get_data(self):
    #     self.train_loader, self.valid_loader, self.test_loader = get_dataset(self.config)
    def _build_method(self):
        if self.args.load_full_data:
            steps_per_epoch = len(self.train_loader)
        else:
            steps_per_epoch = 1
        self.method = RDesign(self.args, self.device, steps_per_epoch)

    def _get_data(self):
        if self.args.load_full_data:
            self.train_loader, self.valid_loader, self.test_loader = get_dataset(self.config)
        else:
            self.test_loader = get_dataset(self.config)

    def _save(self, name=''):
        torch.save(self.method.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))
        fw = open(osp.join(self.checkpoints_path, name + '.pkl'), 'wb')
        state = self.method.scheduler.state_dict()
        pickle.dump(state, fw)

    def _load(self, epoch):
        self.method.model.load_state_dict(torch.load(osp.join(self.checkpoints_path, str(epoch) + '.pth')))
        fw = open(osp.join(self.checkpoints_path, str(epoch) + '.pkl'), 'rb')
        state = pickle.load(fw)
        self.method.scheduler.load_state_dict(state)

    def test(self):
        test_perplexity, test_recovery = self.method.test_one_epoch(self.test_loader)
        print_log('Test Perp: {0:.4f}, Test Rec: {1:.4f}\n'.format(test_perplexity, test_recovery))
        return test_perplexity, test_recovery


if __name__ == '__main__':
    args = create_parser()
    config = args.__dict__

    exp = Exp(args)
    exp.method.model.load_state_dict(torch.load('checkpoints/checkpoint.pth'))
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> testing  <<<<<<<<<<<<<<<<<<<<<<<<<<')
    test_perp, test_rec = exp.test()
