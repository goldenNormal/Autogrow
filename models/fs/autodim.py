import torch
import torch.nn as nn
import numpy as np
import copy
import pickle

from models.DynamicEmb import DynamicEmbedding

class autodim(nn.Module):
    def __init__(self, args, features,  unique_values ):
        super(autodim, self).__init__()
        
        self.emb_dim = args.max_emb_dim # supernet
        
        self.embedding = nn.ParameterDict({fea: DynamicEmbedding(unique_values[i],self.emb_dim, args.max_emb_dim )  for i, fea in enumerate(features)})
        
        self.feature_num = len(features)
        

        self.frequent = 10 # suggested by autodim paper
        
        self.features = features

        
        
        self.theta = nn.ParameterDict({fea:nn.Parameter(torch.full((self.emb_dim,),1.0), requires_grad=True) for fea in features}).to(args.device)
        
        self.out_emb_dim = self.emb_dim
        self.weight = nn.ParameterDict({fea: nn.Parameter(torch.Tensor(self.emb_dim, self.out_emb_dim), requires_grad=True)  for fea in features}).to(args.device)

        self.mid_dim = self.out_emb_dim * len(features)
        self.lin = nn.Linear(self.mid_dim, args.adapt_dim).to(args.device)

        self.bns = nn.ModuleDict({
                fea: nn.ModuleDict({
                    str(d): nn.BatchNorm1d(self.out_emb_dim)
                    for d in range(1, self.emb_dim + 1)
                })
                for fea in features
            }).to(args.device)
        
        self.device = args.device
        self.args = args
        
        self.step  = 0


    def reset_parameters(self):
        import math
        import torch.nn.init as init

        for fea in self.features:
            init.normal_(self.weight[fea], mean=0.0, std=0.01)

        init.kaiming_normal_(self.lin.weight, mode='fan_in', nonlinearity='relu')
        

    def forward(self, inputs ,**kwargs):

        
        temp = max(0.01, 1- 0.00005 * self.step)
        
        if self.training and temp > 0.01:
            
            self.step += 1

        X = []

        for i,fea in enumerate(self.features):

            fea_emb = self.embedding[fea](inputs[:,i])

            
            emb_per_part = []
            
            for d in range(1, self.embedding[fea].max_dim+1):
                x_ = fea_emb[:,:d]
                weight_ = self.weight[fea][:d]

                x_transformed = torch.matmul(x_, weight_)

                x_bn = self.bns[fea][str(d)](x_transformed)
                
                emb_per_part.append(x_bn)

            
            emb_per_part = torch.stack(emb_per_part,dim=-1) # shape (b,  emb_dim, emb_dim)
            
            # shape (dim)
            atten = torch.nn.functional.gumbel_softmax(self.theta[fea],tau=temp, hard=False, dim=-1).reshape(-1,1)

            final_x = torch.matmul(emb_per_part, atten) # agg by atten
            final_x = final_x.reshape(final_x.shape[0],-1) # shape (b, emb_dim)
            
            X.append(final_x)
            
        
        x = torch.concat(X,dim=-1) 
        # print(x)
        x = self.lin(x)
        return x, None
    
    def get_dim_count(self,**kwargs):
        dims = []
        with torch.no_grad():
            for fea in self.features:
                dims.append((fea,self.theta[fea].argmax().item() + 1))
        return dims
    
    def save_search(self,**kwargs):
        from common.utils import save_fea_dim
        
        dims = []
        for fea in self.features:
            dims.append((self.theta[fea].argmax().item() + 1))
        
        return save_fea_dim(self.features, dims)