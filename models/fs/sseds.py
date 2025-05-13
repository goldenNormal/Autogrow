import torch
import torch.nn as nn
import numpy as np
import copy
import pickle
import math

from models.DynamicEmb import DynamicEmbedding
import tqdm
    

class sseds(nn.Module):
    def __init__(self, args, features,  unique_values ):
        super(sseds, self).__init__()
        
        emb_dim = args.max_emb_dim
        self.embedding = nn.ParameterDict({fea: DynamicEmbedding(unique_values[i],emb_dim, emb_dim )  for i, fea in enumerate(features)})
        
        self.feature_num = len(features)
                
        self.features = features
        
        self.mask = nn.ParameterDict({fea:nn.Parameter(torch.full((emb_dim,),1.0),requires_grad=False) for fea in features}).to(args.device)
        # 刚开始训练，不更新mask
        
        self.weight = nn.ParameterDict({fea: nn.Parameter(torch.Tensor(emb_dim, args.adapt_dim))  for fea in features}).to(args.device)
        
        self.reset_parameters()
        
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
        W = []
        

        for i,fea in enumerate(self.features):

            fea_emb = self.embedding[fea](inputs[:,i])

            X.append(fea_emb * self.mask[fea])
            
            W.append(self.weight[fea])
        
        
        x = torch.concat(X,dim=-1)
        
        weight_ = torch.concat(W,dim=0)

        output = torch.matmul(x, weight_)

        return output, None
    

    def single_shot_save_search(self, data,model, prune_ratio = 0.9, **kwargs):
        from common.utils import save_fea_dim
        '''ratio: prune_ratio , the larger, the smaller model size'''
        for fea in self.features:
            self.mask[fea].requires_grad = True
        val_x,val_y = data
        random_perm = torch.randperm(val_x.shape[0])
        batch_size = 4096
        criterion = torch.nn.BCELoss()
        num_batch = val_x.shape[0]//batch_size
       
        average_grads = {fea: torch.zeros((self.args.max_emb_dim),device=self.device) for fea in self.features}

        for i in tqdm.tqdm(range(num_batch)):
            batch_idx = random_perm[i*batch_size: (i+1)*batch_size]
            (c_data, labels) = val_x[batch_idx], val_y[batch_idx]

            out,_ = model(c_data)
            loss = criterion(out, labels.float().unsqueeze(-1))
            
            model.zero_grad()
            loss.backward()

            for fea in self.features:
                average_grads[fea] = average_grads[fea] / (i+1) * i + torch.abs(self.mask[fea].grad) / (i+1)

        all_grads = []
        for fea in self.features:
            all_grads.append(average_grads[fea])
        all_grads = torch.concat(all_grads,dim=0)


        threshold = torch.sort(all_grads).values[math.ceil(all_grads.shape[0] * prune_ratio)]

        dims = []
        for fea in self.features:
            dim = ( average_grads[fea] > threshold ).sum()
            dims.append(dim)

        return save_fea_dim(self.features, dims)
        