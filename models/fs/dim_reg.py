import torch
import torch.nn as nn
import numpy as np
import copy
import pickle

from models.DynamicEmb import DynamicEmbedding


class dim_reg(nn.Module):
    def __init__(self, args, features,  unique_values ):
        super(dim_reg, self).__init__()
        
        emb_dim = args.max_emb_dim
        self.embedding = nn.ParameterDict({fea: DynamicEmbedding(unique_values[i],emb_dim, emb_dim )  for i, fea in enumerate(features)})

        
        self.feature_num = len(features)
                
        self.features = features
        
        self.theta = nn.ParameterDict({fea:nn.Parameter(torch.full((emb_dim,),0.0)) for fea in features}).to(args.device) # 初始值gate为0.5
        
        self.weight = nn.ParameterDict({fea: nn.Parameter(torch.Tensor(emb_dim, args.adapt_dim))  for fea in features}).to(args.device)
        
        self.reset_parameters()
        
        self.temp = 5
        self.device = args.device
        self.args = args

        self.t = 1.7 # best param in their paper
        self.reg_weight = 0.1


    def reset_parameters(self):
        import math
        import torch.nn.init as init
        # 使用He初始化
        for fea in self.features:
            init.normal_(self.weight[fea], mean=0.0, std=0.01)
  

    def forward(self, inputs, **kwargs):

        X = []
        G = []
        W = []
        

        for i,fea in enumerate(self.features):

            fea_emb = self.embedding[fea](inputs[:,i])

            X.append(fea_emb)
            G.append(self.theta[fea])
            W.append(self.weight[fea])
        
        
        x = torch.concat(X,dim=-1)
        theta_ = torch.concat(G,dim=0)
        weight_ = torch.concat(W,dim=0)
        
        # shape (b,d)

        g = torch.sigmoid(theta_ * self.temp)
        
        
        x_ = x * g

        output = torch.matmul(x_, weight_)

        
        fs_loss = self.fs_loss(g)

        return output, fs_loss
    
    def fs_loss(self,g):

        l1_loss = torch.mean(g - torch.abs(g - torch.mean(g))) * self.reg_weight
        return l1_loss
    
    def get_dim_count(self,**kwargs):
        dims = []
        with torch.no_grad():
            for fea in self.features:
                dims.append((fea,(self.theta[fea] >0).sum().item()))
        return dims

    def save_search(self,**kwargs):
        from common.utils import save_fea_dim
        
        dims = []
        for fea in self.features:
            dims.append((self.theta[fea] >0).sum().item())
        
        return save_fea_dim(self.features, dims)