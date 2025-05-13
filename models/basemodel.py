from typing import Union
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.module import Module
from common.utils import get_model
from torch.optim import Adam



class BaseModel(nn.Module):
    def __init__(self, args, backbone_model_name, dim_search,  unique_values, features):
        super(BaseModel, self).__init__()

        self.bb = get_model(backbone_model_name, 'rec')(args, args.adapt_dim) # backbone model name
        self.fs = get_model(dim_search, 'fs')(args,  features, unique_values) # embedding dimension selection method
        self.args = args
        self.features = features
        self.opt = None


        self.bb.to(args.device)


    def forward(self, x, *args):
        
        
        x, loss = self.fs(x,*args) #
 
        x = self.bb(x)
        return x, loss
    

    def set_optimizer(self):
        # 所有参数名称和对象
        all_named_params = list(self.named_parameters())

        # 1. fs_theta 参数（假设都是 theta.xx）
        
        fs_theta_params = [p for name, p in all_named_params if name.__contains__('fs.theta.')]

        # 2. 其他 fs 参数（name 包含 fs 但不是 theta）
        fs_theta_param_ids = set(id(p) for p in fs_theta_params)
        fs_other_params = [
            p for name, p in all_named_params
            if 'fs' in name and not name.__contains__('fs.theta.') and id(p) not in fs_theta_param_ids
        ]


        # 3. backbone 参数（不包含 fs，或者包含 bb）
        fs_all_ids = fs_theta_param_ids.union(set(id(p) for p in fs_other_params))
        bb_params = [
            p for name, p in all_named_params
            if ('fs' not in name or 'bb' in name) and id(p) not in fs_all_ids
        ]

        # 创建优化器
        optimizer_bb = torch.optim.Adam(bb_params, lr=self.args.learning_rate)
        optimizer_fs = torch.optim.Adam(fs_theta_params, lr=self.args.learning_rate) if fs_theta_params else None
        optimizer_fs_other = torch.optim.Adam(fs_other_params, lr=self.args.learning_rate) if fs_other_params else None

        self.opt = {
            'optimizer_bb': optimizer_bb,
            'optimizer_fs': optimizer_fs,
            'optimizer_fs_other': optimizer_fs_other
        }

        return self.opt

    