import torch
import numpy as np
from collections.abc import Mapping, Sequence


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, *args, **kwargs)
    raise TypeError("Can't transfer object type `%s`" % type(obj))

def loss_nll_flatten(S, log_probs):
        """ Negative log probabilities """
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = criterion(log_probs, S)
        loss_av = loss.mean()
        return loss, loss_av