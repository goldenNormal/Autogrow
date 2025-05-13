import torch
import torch.nn as nn



class DynamicEmbedding(nn.Module):
    def __init__(self, num_embeddings, init_dim, max_dim, device='cuda'):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.current_dim = init_dim
        self.device = device
        self.max_dim = max_dim
        

        new_slice = torch.normal(
            mean=0, std=0.01, 
            size=(self.num_embeddings, init_dim),
            device=device 
        )

        # # GPU部分（初始维度）
        self.gpu_weight = nn.Parameter(new_slice, requires_grad=True).to(device)
        


    def expand_one_dim(self):
        """扩展1维（从CPU加载）"""

        if self.current_dim >= self.max_dim:
            print("Reached max dim.")
            return
        
        self.current_dim += 1

        # # 生成新维度
        new_slice = torch.normal(
            mean=0, std=0.01, 
            size=(self.num_embeddings, 1),
            device=self.gpu_weight.device 
        )

        new_weight = torch.cat([self.gpu_weight.data, new_slice], dim=1)
        
        self._parameters["gpu_weight"] = nn.Parameter(new_weight, requires_grad=True)



    def forward(self, input_ids):
        # print(input_ids.device, self.gpu_weight.device)
        return nn.functional.embedding(input_ids, self.gpu_weight)

