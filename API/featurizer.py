import torch
import numpy as np


def find_bracket_pairs(ss, seq):
    pairs = []
    stack = []
    for i, c in enumerate(ss):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                pairs.append((stack.pop(), i))
            else:
                pairs.append((None, i)) 
    if stack:
        pairs.extend(zip(stack[::-1], range(i, i - len(stack), -1)))
        
    npairs = []
    for pair in pairs:
        if None in pair:
            continue
        p_a, p_b = pair
        if (seq[p_a], seq[p_b]) in (('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C')):
            npairs.append(pair)
    return npairs

def shuffle_subset(n, p):
        n_shuffle = np.random.binomial(n, p)
        ix = np.arange(n)
        ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
        ix_subset_shuffled = np.copy(ix_subset)
        np.random.shuffle(ix_subset_shuffled)
        ix[ix_subset] = ix_subset_shuffled
        return ix

def featurize_HC(batch):
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    clus = np.zeros([B], dtype=np.int32)
    ss_pos = np.zeros([B, L_max], dtype=np.int32)
    
    ss_pair = []
    names = []

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']], 1)
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        ss_pos[i, :l] = np.asarray([1 if ss_val!='.' else 0 for ss_val in b['ss']], dtype=np.int32)
        ss_pair.append(find_bracket_pairs(b['ss'], b['seq']))
        names.append(b['name'])
        
        clus[i] = b['cluster']

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    clus = torch.from_numpy(clus).to(dtype=torch.long)
    return X, S, mask, lengths, clus, ss_pos, ss_pair, names

def featurize_HC_Aug(batch):
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    clus = np.zeros([B], dtype=np.int32)
    ss_pos = np.zeros([B, L_max], dtype=np.int32)
    
    
    aug_idxs = []
    aug_Xs = []
    aug_tms = []
    aug_rms = []
    ss_pair = []
    names = []

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']], 1)
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices
        ss_pos[i, :l] = np.asarray([1 if ss_val!='.' else 0 for ss_val in b['ss']], dtype=np.int32)
        ss_pair.append(find_bracket_pairs(b['ss'], b['seq']))
        names.append(b['name'])

        clus[i] = b['cluster']
        
        aug_Xs.append([])
        aug_tms.append([])
        aug_rms.append([])
        if len(batch[i]['augs']) > 0:
            aug_idxs.append(i)
            for aug_item in batch[i]['augs']:
                aug_x = np.stack([aug_item['coords'][c] for c in ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']], 1)
                aug_x_pad = np.pad(aug_x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
                aug_x_pad[np.isnan(aug_x_pad)] = 0.
                aug_Xs[i].append(torch.from_numpy(aug_x_pad).to(dtype=torch.float32))      
                aug_tms[i].append(aug_item['tm-score'])
                aug_rms[i].append(aug_item['rmsd'])
        

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    clus = torch.from_numpy(clus).to(dtype=torch.long)
    return X, aug_Xs, aug_idxs, aug_tms, aug_rms, S, mask, lengths, clus, ss_pos, names