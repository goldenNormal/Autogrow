from typing import Union
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.module import Module
from common.utils import get_model
from torch.optim import Adam
from models.DynamicEmb import DynamicEmbedding


class RetrainBaseModel(nn.Module):
    def __init__(self, args, backbone_model_name, unique_values, features, fea_dim_dict):
        super(RetrainBaseModel, self).__init__()
        # embedding table
        # self.embedding = nn.ParameterDict( {fea: nn.Embedding(unique_values[i],embedding_dim = fea_dim_dict[fea]) for i, fea in enumerate(fea_dim_dict.keys())})
        self.embedding = nn.ParameterDict({fea: DynamicEmbedding(unique_values[i],fea_dim_dict[fea], fea_dim_dict[fea] )  for i, fea in enumerate(features)})

        # for fea in features:
        #     torch.nn.init.normal_(self.embedding[fea].weight.data, mean=0, std=0.01)
        
        input_dim = sum(fea_dim_dict.values())
        self.bb = get_model(backbone_model_name, 'rec')(args, args.adapt_dim) # backbone model name
        
        self.lin = nn.Linear(input_dim, args.adapt_dim)
        self.args = args
        self.features = features
        self.opt = None
        self.fs = None

   
    def forward(self, x):
        X = []
        for i,fea in enumerate(self.features):
            
            X.append( self.embedding[fea](x[:,i]))
        
        x = torch.cat(X,dim=-1)
        x = self.lin(x)
        
        
        x = self.bb(x)
        return x,None
    
    
    def set_optimizer(self):
        optimizer_bb = torch.optim.Adam([params for name,params in self.named_parameters() if ('fs' not in name) or 'bb' in name], lr = self.args.learning_rate)

        if [params for name,params in self.named_parameters() if 'fs' in name] != []:
            optimizer_fs = torch.optim.Adam([params for name,params in self.named_parameters() if 'fs' in name and 'bb' not in name], lr = self.args.learning_rate)
        else:
            optimizer_fs = None

        self.opt = {'optimizer_bb': optimizer_bb, 'optimizer_fs': optimizer_fs}
        
        return self.opt
    