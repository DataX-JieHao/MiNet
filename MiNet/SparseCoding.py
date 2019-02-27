import torch
import numpy as np
import math
import copy
from scipy.interpolate import interp1d

from Survival_CostFunc_CIndex import neg_par_log_likelihood

def dropout_mask(n_node, drop_p):
    '''Construct a binary matrix to randomly drop nodes in a layer.
    Input:
        n_node: number of nodes in the layer.
        drop_p: the probability that a node is to be dropped.
    Output:
        mask: a binary matrix, where 1 --> keep the node; 0 --> drop the node.
    '''
    keep_p = 1.0 - drop_p
    mask = torch.Tensor(np.random.binomial(1, keep_p, size=n_node))
    ###if gpu is being used
    if torch.cuda.is_available():
        mask = mask.cuda()
    ###
    return mask

def fixed_s_mask(w, idx):
    '''Force the connections based on a pre-defined sparse matrix.
    Input: 
        w: weight matrix.
        idx: the indices of having values (or connections).
    Output:
        returns the weight matrix that has been forced the connections.
    '''
    sp_w = torch.sparse_coo_tensor(idx, w[idx], w.size())
    return(sp_w.to_dense())

def get_threshold(w, m, sp):
    '''Obtain the weight value associated to sparsity
    Input: 
        w: weight matrix
        m: the bi-adjacency matrix to indicate the weights that have been updated
        sp: sparsity level
    Output:
        returns the cutoff value (th in soft_threshold())
    '''
    pos_param = torch.abs(torch.masked_select(w, m))
    ###obtain the kth number based on sparse_level
    top_k = math.ceil(pos_param.size(0) * (100 - sp) * 0.01)
    return(torch.topk(pos_param, top_k)[0][-1])

def soft_threshold(w, th):
    '''Soft-thresholding
    Input:
        w: weight matrix
        th: the cutoff value (output from get_threshold())
    Output:
        returns the shrinked weight matrix'''
    return torch.sign(w)*torch.clamp(abs(w) - th, min=0.0)

def get_sparse_weight(w, m, s):
    '''Generate the sparse weight matrix based on sparsity level'''
    epsilon = get_threshold(w, m, s)
    sp_w = soft_threshold(w, epsilon)
    return(sp_w)

def get_best_sparsity(sparse_set, loss_set):
    '''Estimate the best sparsity level by cubic interpolation'''
    interp_loss_set = interp1d(sparse_set, loss_set, kind='cubic')
    interp_sparse_set = torch.linspace(min(sparse_set), max(sparse_set), steps=100)
    interp_loss = interp_loss_set(interp_sparse_set)
    best_sp = interp_sparse_set[np.argmin(interp_loss)]
    return(best_sp)

def small_net_mask(w, m_in_nodes, m_out_nodes):
    '''Generate the masks in order to locate the trained weights in the selected small sub-network'''
    nonzero_idx_in = m_in_nodes.nonzero()
    nonzero_idx_out = m_out_nodes.nonzero()
    sparse_row_idx = nonzero_idx_out.repeat(nonzero_idx_in.size()).transpose(1,-2)
    sparse_col_idx = nonzero_idx_in.repeat(nonzero_idx_out.size()).transpose(1,-2)
    idx = torch.cat((sparse_row_idx, sparse_col_idx), 0)
    val = torch.ones(nonzero_idx_out.size(0)*nonzero_idx_in.size(0))
    sparse_bool_mask = torch.sparse_coo_tensor(idx, val, w.size())
    ##if gpu is being used
    if torch.cuda.is_available():
        sparse_bool_mask = sparse_bool_mask.cuda()
    ###
    mask = sparse_bool_mask.to_dense()
    return(mask.type(torch.uint8))    

def sparse_func(net, x_tr, age_tr, y_tr, delta_tr, Gene_Indices, Pathway_Indices, Dropout_Rate):
    '''Sparse coding phrase: optimize the connections between intermediate layers sequentially'''
    ###serializing net 
    net_state_dict = net.state_dict()
    ###make a copy for net, and then optimize sparsity level via copied net
    copy_net = copy.deepcopy(net)
    copy_state_dict = copy_net.state_dict()
    for name, param in net_state_dict.items():
        ###omit the param if it is not a weight matrix
        if not "weight" in name: continue
        if "omics" in name: continue
        if "gene" in name: continue
        if "hidden2" in name: continue
        if "bn1" in name: continue
        if "bn2" in name: continue
        if "bn3" in name: continue
        if "bn4" in name: continue
        if "pathway" in name:
            active_mask = small_net_mask(net.pathway.weight.data, net.do_m1, net.do_m2)
            copy_weight = copy.deepcopy(net.pathway.weight.data)
        if "hidden" in name:
            active_mask = small_net_mask(net.hidden.weight.data, net.do_m2, net.do_m3)
            copy_weight = copy.deepcopy(net.hidden.weight.data)
        S_set = torch.linspace(99, 0, 5) 
        S_loss = []
        for S in S_set:
            sp_param = get_sparse_weight(copy_weight, active_mask, S.item())
            copy_state_dict[name].copy_(sp_param)
            copy_net.train()
            y_tmp = copy_net(x_tr, age_tr, Gene_Indices, Pathway_Indices, Dropout_Rate)
            loss_tmp = neg_par_log_likelihood(y_tmp, y_tr, delta_tr)
            S_loss.append(loss_tmp)
        ###apply cubic interpolation
        best_S = get_best_sparsity(S_set, S_loss)
        best_epsilon = get_threshold(copy_weight, active_mask, best_S)
        optimal_sp_param = soft_threshold(copy_weight, best_epsilon)
        copy_weight[active_mask] = optimal_sp_param[active_mask]
        ###update weights in copied net
        copy_state_dict[name].copy_(copy_weight)
        ###update weights in net
        net_state_dict[name].copy_(copy_weight)
    return(net)
