import random
import numpy as np
import torch
import os
import copy
import importlib
import datetime
import argparse
import polars as pl

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model(model_name: str, model_type: str):
    """
    Automatically select model class based on model name

    Args:
        model_name (str): model name
        model_type (str): rec, fs, es

    Returns:
        Recommender: model class
        Dict: model configuration dict
    """
    model_file_name = model_name.lower()
    model_module = None
    module_path = '.'.join(['models', model_type, model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)
    else:
        raise ValueError(f'`model_name` [{model_name}] is not the name of an existing model.')
    model_class = getattr(model_module, model_name)
    return model_class

@torch.no_grad()
def compute_polar_metric(w):
    return (torch.mean(torch.abs(w - torch.mean(w)))).item()

class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience.
        
    Args:
        patience (int): How long to wait after last time validation auc improved.
    """

    def __init__(self, args,patience):
        self.patience = patience
        self.trial_counter = 0
        self.best_auc = 0
        self.best_weight = None
        self.args = args

    def stop_training(self, val_auc, model):
        """whether to stop training.

        Args:
            val_auc (float): auc score in val data.
            weights (tensor): the weights of model
        """

        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.trial_counter = 0

            # self.best_model = copy.deepcopy(model)
            if self.args.mode == 'search':
                torch.save(model, f'{self.args.ckpt_path}/{self.args.model}_{self.args.fs}_{self.args.dataset}_{self.args.fs_seed}_best_model.pt')
            else:
                self.best_weights = copy.deepcopy(model.state_dict())

            return False
        
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True
        
        
def save_fea_dim(features, dim_count):
    # 存储格式， feature List，dim
    # 试验主要是做冷启的。
    
    return pl.DataFrame({'fea':features,'dim':dim_count}).sort('dim',descending=True)

    
def print_time(message):
    print(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S '), message)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    