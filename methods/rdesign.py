import torch
import numpy as np
from tqdm import tqdm
from .utils import cuda, loss_nll_flatten
from model import RDesign_Model
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support


alphabet = 'AUCG'
pre_base_pairs = {0: 1, 1: 0, 2: 3, 3: 2}
pre_great_pairs = ((0, 1), (1, 0), (2, 3), (3, 2))

class RDesign:
    def __init__(self, args, device, steps_per_epoch):
        self.args = args
        self.device = device
        self.config = args.__dict__

        self.model = self._build_model()

    def _build_model(self):
        return RDesign_Model(self.args).to(self.device)

    def _cal_recovery(self, dataset, featurizer):
        recovery = []
        S_preds, S_trues = [], []
        for sample in tqdm(dataset):
            sample = featurizer([sample])
            X, S, mask, lengths, clus, ss_pos, ss_pair, names = sample
            X, S, mask, ss_pos = cuda((X, S, mask, ss_pos), device=self.device)
            logits, gt_S = self.model.sample(X=X, S=S, mask=mask)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # secondary sharpen
            ss_pos = ss_pos[mask == 1].long()
            log_probs = log_probs.clone()
            log_probs[ss_pos] = log_probs[ss_pos] / self.args.ss_temp
            S_pred = torch.argmax(log_probs, dim=1)
            
            pos_log_probs = log_probs.softmax(-1)
            for pair in ss_pair[0]:
                s_pos_a, s_pos_b = pair
                if s_pos_a == None or s_pos_b == None or s_pos_b >= S_pred.shape[0]:
                    continue
                if (S_pred[s_pos_a].item(), S_pred[s_pos_b].item()) in pre_great_pairs:
                    continue
                
                if pos_log_probs[s_pos_a][S_pred[s_pos_a]] > pos_log_probs[s_pos_b][S_pred[s_pos_b]]:
                    S_pred[s_pos_b] = pre_base_pairs[S_pred[s_pos_a].item()]
                elif pos_log_probs[s_pos_a][S_pred[s_pos_a]] < pos_log_probs[s_pos_b][S_pred[s_pos_b]]:
                    S_pred[s_pos_a] = pre_base_pairs[S_pred[s_pos_b].item()]
                            
            cmp = S_pred.eq(gt_S)
            recovery_ = cmp.float().mean().cpu().numpy()
            S_preds += S_pred.cpu().numpy().tolist()
            S_trues += gt_S.cpu().numpy().tolist()
            if np.isnan(recovery_): recovery_ = 0.0
            recovery.append(recovery_)
        recovery = np.median(recovery)
        precision, recall, f1, _ = precision_recall_fscore_support(S_trues, S_preds, average=None)
        macro_f1 = f1.mean()
        print('macro f1', macro_f1)
        return recovery
    
    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        with torch.no_grad():
            valid_sum, valid_weights = 0., 0.
            valid_pbar = tqdm(valid_loader)
            for batch in valid_pbar:
                X, S, mask, lengths, clus, ss_pos, ss_pair, names = batch
                X, S, mask, lengths, clus, ss_pos = cuda((X, S, mask, lengths, clus, ss_pos), device=self.device)
                logits, S, _ = self.model(X, S, mask)
                
                log_probs = F.log_softmax(logits, dim=-1)
                loss, _ = loss_nll_flatten(S, log_probs)
                
                valid_sum += torch.sum(loss).cpu().data.numpy()
                valid_weights += len(loss)
                valid_pbar.set_description('valid loss: {:.4f}'.format(loss.mean().item()))
        
        valid_loss = valid_sum / valid_weights
        valid_perplexity = np.exp(valid_loss)        
        return valid_loss, valid_perplexity

    def test_one_epoch(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            test_sum, test_weights = 0., 0.
            test_pbar = tqdm(test_loader)
            for batch in test_pbar:
                X, S, mask, lengths, clus, ss_pos, ss_pair, names = batch
                X, S, mask, lengths, clus, ss_pos = cuda((X, S, mask, lengths, clus, ss_pos), device=self.device)
                logits, S, _ = self.model(X, S, mask)
                
                log_probs = F.log_softmax(logits, dim=-1)
                loss, _ = loss_nll_flatten(S, log_probs)
                
                test_sum += torch.sum(loss).cpu().data.numpy()
                test_weights += len(loss)
                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))

            test_recovery = self._cal_recovery(test_loader.dataset, test_loader.featurizer)
            
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
        return test_perplexity, test_recovery