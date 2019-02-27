from SparseCoding import fixed_s_mask
import torch
import torch.nn as nn

class omics_net(nn.Module):
    ''' Two hidden layers
        One clinical layer (age)
    '''
    def __init__(self, in_nodes, gene_nodes, pathway_nodes, hidden_nodes):
        super(omics_net, self).__init__()
        # activation function
        self.relu = nn.ReLU()
        ###omics layer --> gene layer
        self.omics = nn.Linear(in_nodes, gene_nodes)
        ###gene layer --> pathway layer
        self.gene = nn.Linear(gene_nodes, pathway_nodes)
        ###pathway layer --> hidden layer
        self.pathway = nn.Linear(pathway_nodes, hidden_nodes[0])
        ###hidden layer --> hidden 2 layer
        self.hidden = nn.Linear(hidden_nodes[0], hidden_nodes[1])
        ###hidden 2 layer + clinical layer (age) --> Cox layer
        self.hidden2 = nn.Linear(hidden_nodes[1]+1, 1, bias = False)
        ###batch normalization
        self.bn1 = nn.BatchNorm1d(gene_nodes)
        self.bn2 = nn.BatchNorm1d(pathway_nodes)
        self.bn3 = nn.BatchNorm1d(hidden_nodes[0])
        self.bn4 = nn.BatchNorm1d(hidden_nodes[1])
        ###randomly select a small sub-network
        self.do_m1 = torch.ones(pathway_nodes)
        self.do_m2 = torch.ones(hidden_nodes[0])
        self.do_m3 = torch.ones(hidden_nodes[1])
        ###if gpu is being used
        if torch.cuda.is_available():
            self.do_m1 = self.do_m1.cuda()
            self.do_m2 = self.do_m2.cuda()
            self.do_m3 = self.do_m3.cuda()

    def forward(self, x_1, x_2, gene_idx, pathway_idx, Drop_Rate):
        ###force the connections between omics layer and gene layer w.r.t. 'gene_mask'
        self.omics.weight.data = fixed_s_mask(self.omics.weight.data, gene_idx)
        ###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
        self.gene.weight.data = fixed_s_mask(self.gene.weight.data, pathway_idx)
        # bath norm beofre activation
        x_1 = self.relu(self.bn1(self.omics(x_1)))
        x_1 = self.relu(self.bn2(self.gene(x_1)))
        if self.training == True: 
            # inverted dropout
            x_1 = (1/(1-Drop_Rate[0])) * x_1.mul(self.do_m1)
        x_1 = self.relu(self.bn3(self.pathway(x_1)))
        if self.training == True: 
            x_1 = (1 / (1 - Drop_Rate[1])) * x_1.mul(self.do_m2)
        x_1 = self.relu(self.bn4(self.hidden(x_1)))
        ###add age 
        x_cat = torch.cat((x_1, x_2), 1)
        lin_pred = self.hidden2(x_cat)
        return lin_pred
