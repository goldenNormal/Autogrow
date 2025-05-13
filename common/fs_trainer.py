import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import nni
import datetime as dt
from common.utils import EarlyStopper, compute_polar_metric
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import polars as pl

def safe_rebuild_optimizer(model, old_optimizer):
    new_optimizer = type(old_optimizer)(model.parameters(), **old_optimizer.defaults)

    old_state = old_optimizer.state_dict()
    new_state = new_optimizer.state_dict()

    # 尝试将旧的 state 迁移到新 optimizer
    for k, v in old_state['state'].items():
        if k in new_state['state']:
            for sub_key in v:
                old_tensor = v[sub_key]
                new_tensor = new_state['state'][k].get(sub_key, None)

                # 若 tensor 存在且形状一致，就拷贝过来
                if isinstance(old_tensor, torch.Tensor) and new_tensor is not None:
                    if old_tensor.shape == new_tensor.shape:
                        new_state['state'][k][sub_key].copy_(old_tensor)

    # 载入更新后的 state_dict
    new_optimizer.load_state_dict(new_state)
    return new_optimizer


def reset_adam_state(optimizer):
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            state = optimizer.state[p]
            if state:  # 如果该参数有状态（如 Adam 的 m 和 v）
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)


def get_batch_val(args,val_data,random_perm,i):
    val_x,val_y = val_data
    
    batch_size = args.batch_size
    i = i%(val_data[0].shape[0]//batch_size)
   
    batch_idx = random_perm[i * batch_size:(i+1) * batch_size]
    return val_x[batch_idx], val_y[batch_idx]

class modeltrainer():
    def __init__(self, args, model, model_name, device, epochs, retrain, early_stop=True):
        self.args = args
        self.model = model
        self.optimizers = model.set_optimizer() # dict of optimizers
        self.criterion = torch.nn.BCELoss()
        self.device = torch.device(device)

        self.model.to(self.device)
        self.n_epoch = epochs
        self.batch_size = self.args.batch_size
        # self.model_path = 'checkpoints/' + model_name + '_' + args.fs  + '_' + args.dataset + '/'
        if early_stop:
            self.early_stopper = EarlyStopper(args=args,patience=args.patience)
        else:
            self.early_stopper = None
        self.retrain = retrain


    def train_one_epoch(self, train_data,  epoch_i, val_data,val_dataloader):
        self.model.train()
        bce_loss_per_epoch = []
        fs_loss_per_epoch = []
        
        train_x, train_y = train_data
        assert train_x.device.type != 'cpu' # in gpu train
        
        random_perm = torch.randperm(train_x.shape[0])
        val_random_perm = torch.randperm(val_data[0].shape[0])
        # val_iter = iter(val_dataloader)
        # self.args.num_batch = 1
        for i in tqdm(range(self.args.num_batch)):
            
            batch_idx = random_perm[i*self.batch_size: (i+1)*self.batch_size]
            # print(train_x.shape[0], self.batch_size,max(batch_idx))
            (x, y) = train_x[batch_idx], train_y[batch_idx]
            
            y_pred, fs_loss = self.model(x)
            # print(y_pred,y)
            loss = self.criterion(y_pred, y.float().reshape(-1, 1))
            
            bce_loss_per_epoch.append(loss.item())

             #  fs loss
            if fs_loss is not None and not self.retrain:
                loss += fs_loss
                fs_loss_per_epoch.append(fs_loss.item())
            else:
                fs_loss_per_epoch.append(0)


                

            # optimization parameter
            self.model.zero_grad()
            loss.backward()
            self.optimizers['optimizer_bb'].step()
            
            
            if not self.retrain:
                
                self.optimizers['optimizer_fs_other'].step() # embbedding, adapt_layer
               

                def val_backward_loss():
                     # 在验证集上更新, 即论文中的darts
                    self.optimizers['optimizer_fs'].zero_grad()

                    x_,y_ = get_batch_val(self.args, val_data,val_random_perm, i)
                    
                    y_pred_, fs_loss = self.model(x_)
                    loss_ = self.criterion(y_pred_, y_.float().reshape(-1, 1))
                    if fs_loss is not None:
                        loss_ += fs_loss

                    loss_.backward()

                if self.args.fs in ['shuffle_dim','shuffle_dim_large', 'prune_shuffle_dim','dim_reg']:
   
                    val_backward_loss()
                    
                    self.optimizers['optimizer_fs'].step()

                elif self.args.fs in ['shuffle_dim_gradient_no_darts','shuffle_dim_no_darts']:
   
                    val_backward_loss()
                    
                    self.optimizers['optimizer_fs'].step()
                
                elif self.args.fs == 'autodim' and self.model.fs.step % self.model.fs.frequent == 0:
                    val_backward_loss()
                    self.optimizers['optimizer_fs'].step()

                with torch.no_grad():
                    if self.model.fs is not None and hasattr(self.model.fs, 'update_dim_count'): # shuffle_dim
                        self.model.fs.update_dim_count()

                        if self.model.fs.add_dim: # add new dim, so update the opt
                            opt = self.optimizers['optimizer_fs_other']
                            self.optimizers['optimizer_fs_other'] = safe_rebuild_optimizer(self.model.fs,opt)

                    
                     

        return bce_loss_per_epoch, fs_loss_per_epoch
            
    def fit(self, train_data, val_data):

        val_x, val_y = val_data
        val_dataloader = DataLoader(TensorDataset(val_x.cpu(), val_y.cpu()), batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

        BCE_loss = []
        FS_loss = []
       
        VAL_AUC = []

        all_start_time = dt.datetime.now()
        epoch_time_lis = []
        
        for epoch_i in range(self.n_epoch):
            
            epoch_start_time = dt.datetime.now()
            train_process_data = self.train_one_epoch(train_data, epoch_i,val_data,val_dataloader)
            epoch_end_time = dt.datetime.now()
            
            
            bce_loss_per_epoch, fs_loss_per_epoch = train_process_data

            BCE_loss.extend(bce_loss_per_epoch)
            FS_loss.extend(fs_loss_per_epoch)
            

            epoch_time_lis.append((epoch_end_time - epoch_start_time).total_seconds())
            print('epoch:', epoch_i,' train_loss: ', np.mean(bce_loss_per_epoch),' fs_loss :',np.mean(fs_loss_per_epoch))
            
            if self.model.fs is not None and self.args.fs == 'shuffle_gate' and self.model.fs.mode != 'retrain':
                print((torch.sigmoid(self.model.fs.theta * 5)>0.5).sum().item())

            
            if val_data:
            
                
                auc = self.validate_gpu(val_data,epoch_i )
                VAL_AUC.append(auc)
                print('epoch:', epoch_i, 'validation: auc:', auc)
                
                
                if self.early_stopper is not None and self.early_stopper.stop_training(auc, self.model):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    # self.model = self.early_stopper.best_model
                    
                    if self.args.mode == 'search':
                        self.model = torch.load(f'./ckpt/{self.args.model}_{self.args.fs}_{self.args.dataset}_{self.args.fs_seed}_best_model.pt', map_location=torch.device('cpu'))
                    else:
                        self.model.load_state_dict(self.early_stopper.best_weights)

                    
                    break

        all_end_time = dt.datetime.now()
        print('all training time: {} s'.format((all_end_time - all_start_time).total_seconds()))
        print('average epoch time: {} s'.format(sum(epoch_time_lis) / len(epoch_time_lis)))

        
        return VAL_AUC
        


        

    def validate_gpu(self,val_data, epoch_i):
        val_x,val_y = val_data
        assert val_x.device.type !='cpu'
        assert val_y.device.type !='cpu'

        self.model.eval()
        targets, predicts = list(), list()
        val_batch_size = 3000
        num_batches = val_x.shape[0] // val_batch_size
        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                batch_i,batch_j = i * val_batch_size, (i+1)*val_batch_size
                x,y = val_x[batch_i:batch_j], val_y[batch_i:batch_j]
                y_pred, _ = self.model(x) # current_epoch=None means not in training mode
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        
        if self.model.fs is not None:
            # print(self.model.fs.dim_count)
            if hasattr(self.model.fs, 'get_dim_count'):
                print(self.model.fs.get_dim_count())
        auc = roc_auc_score(np.asarray(targets),np.asarray(predicts))
        return auc


    def test_eval(self,test_data,load_model=False):
        test_x,test_y = test_data
        if load_model:
            self.model = torch.load(f'./ckpt/{self.args.model}_{self.args.fs}_{self.args.dataset}_{self.args.fs_seed}_best_model.pt', map_location= self.args.device)
        
        self.model.eval()
        self.model = self.model.to('cuda')
        targets, predicts = list(), list()
        test_batch_size = 3000
        num_batches =test_y.shape[0]//test_batch_size
        avg_loss = 0
        criterion = torch.nn.BCELoss()
        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                batch_i,batch_j = i * test_batch_size, (i+1)*test_batch_size
                x,y = test_x[batch_i:batch_j], test_y[batch_i:batch_j]
                x = x.to('cuda')
                y = y.to('cuda')
                y_pred,_ = self.model(x)
            
                bce_loss = criterion(y_pred, y.float().reshape(-1, 1))
                avg_loss += bce_loss.item()
            
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        
        auc = roc_auc_score(np.asarray(targets),np.asarray(predicts))
        avg_loss /= num_batches
        
        return auc, avg_loss

    def save_model(self,path):
        torch.save(self.model.state_dict(), path)


    