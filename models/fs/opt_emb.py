import torch
import torch.nn as nn
import numpy as np
import copy
import pickle
import math
import random
from sklearn.metrics import roc_auc_score
from models.DynamicEmb import DynamicEmbedding
import tqdm

class opt_emb(nn.Module):
    def __init__(self, args, features,  unique_values ):
        super(opt_emb, self).__init__()
        
        emb_dim = args.max_emb_dim
        self.embedding = nn.ParameterDict({fea: DynamicEmbedding(unique_values[i],emb_dim, emb_dim )  for i, fea in enumerate(features)})
        
        self.feature_num = len(features)
                
        self.features = features
        
        self.weight = nn.ParameterDict({fea: nn.Parameter(torch.Tensor(emb_dim, args.adapt_dim))  for fea in features}).to(args.device)
        
        self.reset_parameters()
        
        self.device = args.device
        self.args = args


        ##### OptEmb parameters.
        self.m_prob = 0.1
        self.iterations = 30
        self.mutation_num = 10
        self.crossover_num = 10
        self.k = 15

        self.cands =[]


    def reset_parameters(self):
        import math
        import torch.nn.init as init
        # 使用He初始化
        for fea in self.features:
            init.normal_(self.weight[fea], mean=0.0, std=0.01)
  

    def forward(self, inputs, sample_dims = None, **kwargs):

        X = []
        W = []
        
        if sample_dims is None:
            sample_dims = np.random.randint(1,self.args.max_emb_dim, self.feature_num)

        for i,fea in enumerate(self.features):
            d = sample_dims[i]
            
            fea_emb = self.embedding[fea](inputs[:,i])[:, :d]

            X.append(fea_emb)
            
            W.append(self.weight[fea][:d])
        
        
        x = torch.concat(X,dim=-1)
        
        weight_ = torch.concat(W,dim=0)

        output = torch.matmul(x, weight_)

        return output, None
    
    
    def eval_val_data(self,val_data,model, cand):
        val_x,val_y = val_data

        model.eval()
        targets, predicts = list(), list()
        val_batch_size = 4096
        num_batches = val_x.shape[0] // val_batch_size
        with torch.no_grad():
            for i in range(num_batches):
                batch_i,batch_j = i * val_batch_size, (i+1)*val_batch_size
                x,y = val_x[batch_i:batch_j], val_y[batch_i:batch_j]
                y_pred, _ =  model(x,cand) # current_epoch=None means not in training mode
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())

        auc = roc_auc_score(np.asarray(targets),np.asarray(predicts))
        return auc

    def eval_all_parts(self,val_data,model):
        metrics = []
        for cand in self.cands:
            metrics.append(self.eval_val_data(val_data, model,cand))
        return metrics


    def get_random(self, num):
        print("Generating random embedding masks ...")
        self.cands = []
        for i in range(num):
            cand = torch.randint(low=1, high=self.args.max_emb_dim + 1, size=(self.feature_num,)).to(self.device)
            self.cands.append(cand)

    def sort_topk_cands(self, metrics):
        reverse = [1-i for i in metrics]
        indexlist = np.argsort(reverse)[:self.k]
        self.cands = [self.cands[i] for i in indexlist]

    def get_mutation(self, mutation_num, m_prob):
        mutation = []
        assert m_prob > 0

        for i in range(mutation_num):
            origin = self.cands[i]
            for i in range(self.feature_num):
                if random.random() < m_prob:
                    index = torch.tensor(i).to(self.device)
                    rand_value = torch.randint(low=1, high=self.args.max_emb_dim + 1, size=(1,)).to(self.device)
                    origin[index] = rand_value
            mutation.append(origin)
        return mutation

    def get_crossover(self, crossover_num):
        crossover = []

        def indexes_gen(m, n):
            seen = set()
            x, y = random.randint(m, n), random.randint(m, n)
            while True:
                seen.add((x,y))
                yield (x, y)
                x, y = random.randint(m, n), random.randint(m, n)
                while (x, y) in seen:
                    x, y = random.randint(m, n), random.randint(m, n)
        gen = indexes_gen(0, crossover_num)
        
        for i in range(crossover_num):
            point = random.randint(1, self.args.max_emb_dim)
            x, y = next(gen)
            origin_x, origin_y = self.cands[x].cpu().numpy(), self.cands[y].cpu().numpy()
            xy = np.concatenate((origin_x[:point], origin_y[point:]))
            crossover.append(torch.from_numpy(xy).to(self.device))   
        return crossover
    
    def single_shot_save_search(self, data,model, **kwargs):
        from common.utils import save_fea_dim
        
        acc_auc = 0.0
        
        print('Begin Searching ...')
        self.get_random(self.mutation_num + self.crossover_num)
        for  _ in tqdm.tqdm(range(self.iterations)):
            aucs = self.eval_all_parts(data, model)
            self.sort_topk_cands(aucs)

            if acc_auc < aucs[0]:
                acc_auc, acc_cand = aucs[0], self.cands[0]

            mutation = self.get_mutation(self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.crossover_num)
            self.cands = mutation + crossover
        
        final_acc_auc = self.eval_val_data(data,model, cand=acc_cand)
        print(f'End Search, auc is {final_acc_auc}')

        dims = []
        for i,fea in enumerate(self.features):
            d = acc_cand[i]
            dims.append(d)

        return save_fea_dim(self.features, dims)
    