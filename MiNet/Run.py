from DataLoader import load_sparse_indices, load_data
from Train import train_omics_net
from Eval import eval_omics_net

import torch
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(description="Train omics data")
parser.add_argument("REP_ID", type=int, default=1)
parser.add_argument("GPU_ID", type=int, default=0)

REPID = parser.parse_args().REP_ID
GPUID = parser.parse_args().GPU_ID

'''Set up '''
torch.cuda.set_device(GPUID)

##### Net
In_Nodes = 24803 ### number of omics
Gene_Nodes = 5481 ### number of genes
Pathway_Nodes = 507 ### number of pathways
Hidden_Nodes = [22, 5] ### number of hidden nodes
##### Initials
max_epochs = 10000 
Drop_Rate = [0.7, 0.5] ### dropout rates
''' load data '''
folder_path = "/home/NewUsersDir/jhao2/data/proposed/"
pathway_indices = load_sparse_indices(folder_path+"gbm_binary_pathway_mask.npz")
gene_indices = load_sparse_indices(folder_path+"gbm_binary_gene_mask.npz")

x_train, ytime_train, yevent_train, age_train = load_data(folder_path+"gbm_std_imputed_train_"+str(REPID)+".csv")
x_valid, ytime_valid, yevent_valid, age_valid = load_data(folder_path+"gbm_std_imputed_valid_"+str(REPID)+".csv")

###grid search the optimal hyperparameters using train and validation data
L2_Lambda = [0.01, 0.02, 0.04, 0.08, 0.10, 0.12]
Initial_Learning_Rate = [1e-2, 5e-3, 1e-3]

opt_cidx = 0.0
for lr in Initial_Learning_Rate:
    for l2 in L2_Lambda:
        print("L2: ", l2, "LR: ", lr)
        c_index_tr, c_index_va = train_omics_net(x_train, age_train, ytime_train, yevent_train, \
                                                x_valid, age_valid, ytime_valid, yevent_valid, \
                                                gene_indices, pathway_indices, \
                                                In_Nodes, Gene_Nodes, Pathway_Nodes, Hidden_Nodes, \
                                                lr, l2, max_epochs, Drop_Rate, step = 100, tolerance = 0.02, \
                                                sparse_coding = True)
        if (c_index_tr.item() > c_index_va.item()) and (c_index_va.item() > opt_cidx):
            opt_l2 = l2
            opt_lr = lr
            opt_cidx_tr = c_index_tr
            opt_cidx = c_index_va
print("Optimal l2: ", opt_l2, "Optimal lr: ", opt_lr)



x_test, ytime_test, yevent_test, age_test = load_data(folder_path+"gbm_std_imputed_test_"+str(REPID)+".csv")
train_cindex, valid_cindex, eval_cindex = eval_omics_net(x_train, age_train, ytime_train, yevent_train, \
                                                        x_valid, age_valid, ytime_valid, yevent_valid, \
                                                        x_test, age_test, ytime_test, yevent_test, \
                                                        gene_indices, pathway_indices, \
                                                        In_Nodes, Gene_Nodes, Pathway_Nodes, Hidden_Nodes, \
                                                        opt_lr, opt_l2, max_epochs, Drop_Rate, step = 100, tolerance = 0.02, \
                                                        sparse_coding = True)
print("C-index in Test: ", eval_cindex.item())


