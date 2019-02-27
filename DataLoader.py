import numpy as np
import pandas as pd
from scipy import sparse
import torch

def load_sparse_indices(path):

    coo = sparse.load_npz(path)
    indices = np.vstack((coo.row, coo.col))

    return(indices)



def sort_data(path):

    data = pd.read_csv(path)
    
    data.sort_values("OS_MONTHS", ascending = False, inplace = True)
    
    x = data.drop(["OS_MONTHS", "OS_EVENT", "AGE"], axis = 1).values
    ytime = data.loc[:, ["OS_MONTHS"]].values
    yevent = data.loc[:, ["OS_EVENT"]].values
    age = data.loc[:, ["AGE"]].values

    return(x, ytime, yevent, age)



def load_data(path):

    x, ytime, yevent, age = sort_data(path)

    x = torch.from_numpy(x).to(dtype=torch.float).cuda()
    ytime = torch.from_numpy(ytime).to(dtype=torch.float).cuda()
    yevent = torch.from_numpy(yevent).to(dtype=torch.float).cuda()
    age = torch.from_numpy(age).to(dtype=torch.float).cuda()

    return(x, ytime, yevent, age)