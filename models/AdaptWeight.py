import torch
import torch.nn as nn


class AdaptWeight(nn.Module):
    def __init__(self, adapt_dim, init_dim, max_dim, device='cuda'):
        super().__init__()
        self.adapt_dim = adapt_dim
        self.current_dim = init_dim
        self.device = device
        self.max_dim = max_dim
        

        new_slice = torch.normal(
            mean=0, std=0.01, 
            size=(init_dim, adapt_dim),
            device=device 
        )

        # # GPU部分（初始维度）
        self.weight = nn.Parameter(new_slice, requires_grad=True).to(device)
        


    def expand_one_dim(self):
        """扩展1维（从CPU加载）"""

        if self.current_dim >= self.max_dim:
            print("Reached max dim.")
            return
        
        self.current_dim += 1

        new_slice = torch.normal(
            mean=0, std=0.01, 
            size=(1, self.adapt_dim),
            device=self.device 
        )

        new_weight = torch.cat([self.weight, new_slice], dim=0)
        
        self._parameters["weight"] = nn.Parameter(new_weight, requires_grad=True)



    def forward(self):
        return self.weight

