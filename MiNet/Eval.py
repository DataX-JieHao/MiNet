import numpy as np
import pandas as pd
import copy
from Net import omics_net
from SparseCoding import dropout_mask, fixed_s_mask, sparse_func
from Survival_CostFunc_CIndex import R_set, neg_par_log_likelihood, c_index

import torch
import torch.nn as nn
import torch.optim as optim

def eval_omics_net(x_tr, age_tr, y_tr, delta_tr, \
            x_va, age_va, y_va, delta_va, \
            x_te, age_te, y_te, delta_te, \
            gene_indices, pathway_indices, \
            in_nodes, gene_nodes, pathway_nodes, hidden_nodes, \
            LR, L2, max_epochs, dropout_rate, step = 100, tolerance = 0.02, sparse_coding = False):

    net = omics_net(in_nodes, gene_nodes, pathway_nodes, hidden_nodes)
    ###if gpu is being used
    if torch.cuda.is_available():
        net = net.cuda()
    ###optimizer
    opt = optim.Adam(net.parameters(), lr=LR, weight_decay = L2)

    prev_sum = 0.0
    for epoch in range(max_epochs):
        net.train()
        ###reset gradients to zeros
        opt.zero_grad() 
        ###Randomize dropout masks
        net.do_m1 = dropout_mask(pathway_nodes, dropout_rate[0])
        net.do_m2 = dropout_mask(hidden_nodes[0], dropout_rate[1])
        ###Forward
        pred = net(x_tr, age_tr, gene_indices, pathway_indices, dropout_rate)
        ###calculate loss
        loss = neg_par_log_likelihood(pred, y_tr, delta_tr)
        ###calculate gradients
        loss.backward() 
        ###force the connections between omics layer and gene layer w.r.t. 'gene_mask'
        net.omics.weight.grad = fixed_s_mask(net.omics.weight.grad, gene_indices)
        ###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
        net.gene.weight.grad = fixed_s_mask(net.gene.weight.grad, pathway_indices)
        ###update weights and biases
        opt.step()
        if sparse_coding == True:
            net = sparse_func(net, x_tr, age_tr, y_tr, delta_tr, gene_indices, pathway_indices, dropout_rate)
        if epoch % step == step - 1:
            net.train()
            pred = net(x_tr, age_tr, gene_indices, pathway_indices, dropout_rate)
            train_cindex = c_index(pred.cpu(), y_tr.cpu(), delta_tr.cpu())
            net.eval()
            pred = net(x_va, age_va, gene_indices, pathway_indices, dropout_rate)
            eval_cindex = c_index(pred.cpu(), y_va.cpu(), delta_va.cpu())
            if ((eval_cindex.item() + train_cindex.item() + tolerance) < prev_sum): 
                print('Early stopping in [%d]' % (epoch + 1))
                print('[%d] Best CIndex in Train: %.3f' % (epoch + 1, opt_cidx_tr))
                print('[%d] Best CIndex in Valid: %.3f' % (epoch + 1, opt_cidx_va))
                opt_net.eval()
                pred = opt_net(x_te, age_te, gene_indices, pathway_indices, dropout_rate)
                eval_cindex = c_index(pred.cpu(), y_te.cpu(), delta_te.cpu())
                break
            else:
                prev_sum = eval_cindex.item() + train_cindex.item()
                opt_cidx_tr = train_cindex
                opt_cidx_va = eval_cindex
                opt_net = copy.deepcopy(net)
                print('[%d] CIndex in Train: %.3f' % (epoch + 1, train_cindex))
                print('[%d] CIndex in Valid: %.3f' % (epoch + 1, eval_cindex))

    return (opt_cidx_tr, opt_cidx_va, eval_cindex)
