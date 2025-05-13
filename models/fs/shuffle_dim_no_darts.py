import torch
import torch.nn as nn
import numpy as np
import copy
import pickle

from models.DynamicEmb import DynamicEmbedding
from models.AdaptWeight import AdaptWeight

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

class shuffle_dim_no_darts(nn.Module):
    def __init__(self, args, features,  unique_values ):
        super(shuffle_dim_no_darts, self).__init__()
        
        init_dim = 1
        self.embedding = nn.ParameterDict({fea: DynamicEmbedding(unique_values[i],init_dim, args.max_emb_dim )  for i, fea in enumerate(features)})

        
        self.feature_num = len(features)
        self.max_emb_dim = args.max_emb_dim
        
        self.max_total_dim = self.feature_num * self.max_emb_dim
        
        self.features = features
        
        self.theta = nn.ParameterDict({fea:torch.full((self.max_emb_dim,),0.0) for fea in features}).to(args.device) # 初始值gate为0.5
        self.dim_count = {fea:1 for fea in self.features}
        self.max_dim_count = {fea:1 for fea in self.features}

        self.current_dim = self.feature_num


        # self.weight = nn.Parameter(torch.Tensor(self.max_total_dim, self.adpt_dim))
        # self.weight = nn.ParameterDict({fea: torch.Tensor(init_dim, args.adapt_dim)  for fea in features}).to(args.device)
        self.weight = nn.ParameterDict({fea: AdaptWeight(args.adapt_dim,init_dim, args.max_emb_dim )  for fea in features})

        self.masks = {}

        self.update_threshold = (0.01,0.6)
        
        self.temp = 5
        self.device = args.device
        self.args = args
        self.add_dim = False



    def f_torch(self,x):
        return 1 / torch.arange(1, x + 1, dtype=torch.float,device=self.device)

    def update_dim_count(self):
        
        low,high = self.update_threshold

        self.add_dim = False
        
        for fea in self.features:
            gate= torch.sigmoid(self.theta[fea] * self.temp)
            dim = self.dim_count[fea]
            # print(f'in shuffle_dim, dim = {dim}')
            if (gate[:dim] > high).all() and self.args.budget > self.current_dim and dim < self.args.max_emb_dim:
                self.dim_count[fea] +=1
                self.current_dim +=1
     
                if self.dim_count[fea] > self.max_dim_count[fea]: # 需要扩dim
                    # print(f'extend {fea}')
                    self.max_dim_count[fea] = self.dim_count[fea]
                    self.add_dim = True
                    self.embedding[fea].expand_one_dim()
                    
                    # 给 weight 扩 dim
                    self.weight[fea].expand_one_dim()


                
            
            elif gate[:dim].min() < low and self.dim_count[fea] > 1:
                self.dim_count[fea] -= 1
                self.current_dim -=1
        
                # print(f'fea: {fea}, new dim: {self.dim_count[fea]} !')



                

    def forward(self, inputs,**kwargs):

        X = []
        G = []
        W = []
        
        # G_L1 = []
        for i,fea in enumerate(self.features):
            dim = self.dim_count[fea]


            fea_emb = self.embedding[fea](inputs[:,i])

            X.append(fea_emb[:, :dim])
            G.append(self.theta[fea][:dim])
            W.append(self.weight[fea]()[:dim])
        
        
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
            dim = self.dim_count[fea]

            dim_weight = self.f_torch(dim).reshape(-1)

            G.append(gate[:dim]  * dim_weight * self.args.fs_weight * 0.1)
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
        
        # dim_count = self.mask.sum(dim=1).detach().cpu().numpy()
        dims = []
        for fea in self.features:
            dims.append((self.theta[fea] >0).sum())
        
        return save_fea_dim(self.features, dims)


    