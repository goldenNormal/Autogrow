import sys
import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 启用同步kernel启动
import traceback
import argparse
import yaml
import nni
import time
import datetime as dt
from tqdm import tqdm
import common.utils as utils
from common.fs_trainer import modeltrainer
from common.datasets import quick_read_dataset
from models.basemodel import BaseModel
from models.retrain_model import RetrainBaseModel
import polars as pl
import math
import gc

Mode = None

def search_stage_main(args,seed):
    args.mode = 'search'


    feature_dim_path = f'{args.save_path}/fea_dim/{args.fs}-{args.dataset}-{args.model}-{seed}.csv'
    
    if os.path.exists(feature_dim_path):
        print('hit... cache..')
        return
     
    utils.seed_everything(seed)
    features, _, unique_values, data_df = quick_read_dataset(args.dataset)
    train_x, train_y, val_x, val_y, test_x, test_y = data_df

    if args.fs == 'no_search': # max_dim or  math.floor(budget/num_feature)
        from common.utils import save_fea_dim
        dim_ = args.max_emb_dim
        
        save_fea_dim(features, [dim_ for _ in range(len(features))]).write_csv(feature_dim_path)
        return 

    import math
    args.num_batch =math.floor (train_x.shape[0] / args.batch_size)
    # args.num_batch = 1
    print(args.num_batch)
    print('-'*10)
    print('features and unique_values:')
    print(features)
    print(unique_values)
    print('-'* 10)

    
    

    # deep method
    model = BaseModel(args, args.model, args.fs, unique_values, features)

    # model.fs.mode = 'train'
    
    trainer = modeltrainer(args, model, args.model, args.device, epochs=args.epoch, retrain=False, early_stop=True)
    
    #### prepare data to device to speed up
    train_x, val_x, test_x = torch.tensor(train_x.values, dtype=torch.long), torch.tensor(val_x.values, dtype=torch.long), torch.tensor(test_x.values, dtype=torch.long)
    train_y, val_y, test_y = torch.tensor(train_y.values, dtype=torch.long), torch.tensor(val_y.values, dtype=torch.long), torch.tensor(test_y.values, dtype=torch.long)
    
    train_data = (train_x.to('cuda'), train_y.to('cuda'))
    val_data = (val_x.to('cuda'),val_y.to('cuda'))
    test_data = (test_x, test_y)
    ####
    save_model_pt = f'./ckpt/{args.model}_{args.fs}_{args.dataset}_{args.fs_seed}_best_model.pt'
    if not os.path.exists(save_model_pt):
        print('search .... stage ...')
        s = time.time()
        Val_AUC = trainer.fit(train_data,val_data)
        consume_time = time.time() - s

        test_auc, test_bce_loss = trainer.test_eval(test_data,load_model = True)
    else:
        test_auc, test_bce_loss = trainer.test_eval(test_data,load_model = True)
    
    
    model = trainer.model
    print(f'test_auc: {test_auc}, test_loss: {test_bce_loss}')

    print('write train process...')


    # train_metric_csv_path = f'{args.save_path}/train_metric/search-{args.fs}-{args.dataset}-{args.model}-{seed}.csv'
    # pl.DataFrame({'epoch':range(len(Val_AUC)+3),'metric': ['val_auc'] * len(Val_AUC) + ['test_auc','test_bce_loss','search_time'] ,
    #                'value':Val_AUC + [test_auc, test_bce_loss,consume_time]}).write_csv(train_metric_csv_path)
    
    
    if hasattr(model.fs, 'save_search'):
        
        res = model.fs.save_search()

        res.write_csv(feature_dim_path)

    if hasattr(model.fs, 'single_shot_save_search'):
        
        res = model.fs.single_shot_save_search(val_data,model, prune_ratio=0.9)

        res.write_csv(feature_dim_path)


def retrain_stage_main(args, seed, fs_seed):
    args.mode = 'retrain'
    utils.seed_everything(seed)

    # 1. Get the corresponding feature dimension...
        
    fea_dim_path = f'{args.save_path}/fea_dim/{args.fs}-{args.dataset}-{args.model}-{fs_seed}.csv'
    fea_df = pl.read_csv(fea_dim_path).sort(by='fea')

    fea_dict = {}
    selected_features= []
    for item in fea_df.to_dicts():
        if item['dim']>0:
            fea_dict[str(item['fea'])] = item['dim']
            selected_features.append(item['fea'])

    fea_dim_str = str(fea_dict)
    
    print(f'select: {fea_dim_str}')
    
    # 2. Go to record_auc to find out if there is already a corresponding auc. 
    # If so, reuse it. If not, retrain the model and add a new auc data.
    record_path = f'{args.save_path}/retrain_result.csv'
    metric_df = pl.read_csv(record_path)
    
    filter_df = metric_df.filter(
        pl.col('dataset') == args.dataset,
        pl.col('fea_dim_pair') == fea_dim_str,
        pl.col('model') == args.model,
        pl.col('seed') == seed
    )
    match_before = filter_df.shape[0]>0
    if match_before:
        test_auc,test_bce_loss = filter_df['auc'][0], filter_df['bce_loss'][0]
        print(f'match cache... test_auc is {test_auc}, test_bce_loss: {test_bce_loss}')
        
    else:
        
        features, _, unique_values, data_df = quick_read_dataset(args.dataset, selected_features=selected_features)
        train_x, train_y, val_x, val_y, test_x, test_y = data_df

        import math
        args.num_batch =math.floor (train_x.shape[0] / args.batch_size)
        

        print('-'*10)
        print('features and unique_values:')
        print(features)
        print(unique_values)
        print('-'* 10)

        utils.print_time('start retrain...')
        print(args.fs,args.dataset)
        model = RetrainBaseModel(args, args.model,  unique_values, features, fea_dim_dict=fea_dict)

        
        trainer = modeltrainer(args, model, args.model, args.device, epochs=args.epoch, retrain=True)
        #### prepare data to device to speed up
        train_x, val_x, test_x = torch.tensor(train_x.values, dtype=torch.long), torch.tensor(val_x.values, dtype=torch.long), torch.tensor(test_x.values, dtype=torch.long)
        train_y, val_y, test_y = torch.tensor(train_y.values, dtype=torch.long), torch.tensor(val_y.values, dtype=torch.long), torch.tensor(test_y.values, dtype=torch.long)
        
        train_data = (train_x.to('cuda'), train_y.to('cuda'))
        val_data = (val_x.to('cuda'),val_y.to('cuda'))
        test_data = (test_x, test_y)
        ####

        
        Val_AUC = trainer.fit(train_data, val_data)
        test_auc, test_bce_loss = trainer.test_eval(test_data)

        print(f'retrain finished...\n test_auc is {test_auc}, test_bce_loss: {test_bce_loss}')

        # 写入 csv 数据
        new_record = pl.DataFrame({
                    'dataset':args.dataset,
                    'fea_dim_pair':fea_dim_str,
                    'model': args.model,
                    'seed': seed,
                    'auc': test_auc,
                    'bce_loss':test_bce_loss})
        
        pl.concat([metric_df,new_record]).write_csv(record_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='widedeep', help='mlp, widedeep...')
    # parser.add_argument('--dataset', type=str, default='avazu', help='avazu, criteo, movielens-1m, aliccp')  
    # parser.add_argument('--fs', type=str, default='no_selection', help='feature selection methods: \
                        # no_selection, shuffle_gate, autofield, gbdt, lasso, lpfs,  \
                        #  rf, sfs, shark, xgb...')
    
    parser.add_argument('--device',type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu, cuda')
    parser.add_argument('--data_path', type=str, default='quick_data/', help='data path')
    
    parser.add_argument('--max_emb_dim', type=int, default=16, help='embedding dimension')
    parser.add_argument('--adapt_dim', type=int, default=64, help='')
    
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--budget', type=int, default=1000, help='dim budget')
    
    parser.add_argument('--patience', type=int, default=2, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=32, help='num_workers')
    
    args = parser.parse_args()

    with open('models/config.yaml', 'r') as file:
        data = yaml.safe_load(file)
    args.__dict__.update(data)
    
    
    args.timestr = str(time.time())
    fs_methods = [
        # 'no_search',
        #  'shuffle_dim',
        #  'shuffle_dim_large',
        # 'prune_shuffle_dim',
        'autodim',
        # 'dim_reg',
        # 'sseds',
        # 'opt_emb',
                  ]
    
    # fs_methods =['shuffle_dim_gradient_no_darts','shuffle_dim_no_darts']
    data_list = ['criteo']
    # data_list = ['aliccp','avazu','criteo']

    # copy hp from shuffle_gate
    fs_weight_map = {'movielens-1m':0.1,'aliccp':0.00125,'avazu':0.005,'criteo':0.02}


    model_list = ['widedeep']
    
    args.save_path = './exp_save/'

    fs_seed_list = [200,201]
    # fs_seed_list = [900,901] # 测试用
    retrain_seed_list = [0,1]

    # 配置日志
    
    logging.basicConfig(
        filename=f'error_{datetime.now().strftime("%Y%m%d")}.log',
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    for dataset in data_list:
        args.dataset = dataset
        if args.dataset == 'movielens-1m':
            args.batch_size = 256
            args.epoch = 100

        else:
            args.batch_size = 2048
        
        args.fs_weight = fs_weight_map[dataset]

        for fs in fs_methods:
            args.fs = fs

            # print args
            
            print(f'======== {args.fs} on {args.dataset} ========')
            print(f'========Print Args========')
            for key in args.__dict__:
                if key not in ['fs_config','rec_config']:
                    print(key, ':', args.__dict__[key])
                else:
                    print(key, ':')
                    for key2 in args.__dict__[key]:
                        if key2 in [args.model, args.fs]:
                            print('\t', key2, ':', args.__dict__[key][key2])
            print('')
            ######

            try:
                for fs_seed in fs_seed_list:
                    args.fs_seed = fs_seed
                    print(f'search seed {fs_seed}')
                    search_stage_main(args, fs_seed)

                    torch.cuda.empty_cache()  # 清空 CUDA 显存
                    print('here reach..')

                for fs_seed in fs_seed_list:
                    args.fs_seed = fs_seed      
                    for seed in retrain_seed_list:
                        args.seed = seed
                        retrain_stage_main(args,seed, fs_seed )
                        
                        torch.cuda.empty_cache()  # 清空 CUDA 显存
                
            except Exception as e:
                import subprocess
                subprocess.run('~/code/send_msg.sh "autodim任务失败"',shell=True,capture_output=False)
                # 记录异常信息到日志
                
                print(f"{args.fs}方法在数据集{args.dataset}上 发生异常: {e}")
                print(traceback.format_exc())  # 记录完整的异常堆栈

            #      # 清空 PyTorch 显存
            #     if torch.cuda.is_available():
            #         torch.cuda.empty_cache()  # 清空 CUDA 显存
            #         gc.collect()  # 强制垃圾回收
            #         print("已清空 PyTorch 显存")
    import subprocess
    subprocess.run('~/code/send_msg.sh "任务完成"',shell=True,capture_output=False)
