import torch
import torch.nn as nn
import numpy as np
import copy
import pickle

from models.DynamicEmb import DynamicEmbedding

def shuffle_batch(x):
    """
    对输入的 tensor 进行特征 shuffle，每个特征在 batch 维度上进行独立的 shuffle。
    """
    b, d = x.shape

    # 生成随机排列的索引
    indices = torch.rand(d, b, device=x.device).argsort(dim=1)  # (d, b)

    # 使用 torch.gather 进行 shuffle
    shuffled_x = torch.gather(x.T, dim=1, index=indices).T  # 先转置 (d, b)，然后恢复 (b, d)

    return shuffled_x

class prune_shuffle_dim(nn.Module):
    def __init__(self, args, features,  unique_values ):
        super(prune_shuffle_dim, self).__init__()
        
        init_dim = args.max_emb_dim # 1
        # init_dim = 1
        self.embedding = nn.ParameterDict({fea: DynamicEmbedding(unique_values[i],init_dim, args.max_emb_dim )  for i, fea in enumerate(features)})

        
        self.feature_num = len(features)
        self.max_emb_dim = args.max_emb_dim
        
        self.max_total_dim = self.feature_num * self.max_emb_dim
        
        self.features = features
        
        self.theta = nn.ParameterDict({fea:nn.Parameter(torch.full((self.max_emb_dim,),0.0)) for fea in features}).to(args.device) # 初始值gate为0.5
        
        # self.weight = nn.Parameter(torch.Tensor(self.max_total_dim, self.adpt_dim))
        self.weight = nn.ParameterDict({fea: nn.Parameter(torch.Tensor(self.max_emb_dim, args.adapt_dim))  for fea in features}).to(args.device)
        
        self.reset_parameters()
        
        self.temp = 5
        self.device = args.device
        self.args = args


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
        shuffle_x = shuffle_batch(x)

        g = torch.sigmoid(theta_ * self.temp)

        x_ = x * g + (1-g) * shuffle_x.detach()

        output = torch.matmul(x_, weight_)

        
        fs_loss = self.fs_loss()

        return output, fs_loss
    
    def fs_loss(self):
        G = []
        for fea in self.features:
            gate= torch.sigmoid(self.theta[fea] * self.temp).reshape(-1)
            G.append(gate * self.args.fs_weight)
            #  to do... dim descend

        g = torch.cat(G,dim=-1)

        l1_loss = torch.mean(g)
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
            dims.append((self.theta[fea] >0).sum())
        
        return save_fea_dim(self.features, dims)