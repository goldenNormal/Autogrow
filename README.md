# 论文标题


## Getting Started

### Dataset
The preprocessed DRS dataset is available on Huggingface:
- Dataset: [DRS-dataset](https://huggingface.co/datasets/yihong-1101/DRS-dataset)
- Download the dataset to the `quick_data` directory

Note: `utils/datasets.py` describes how we transform the original ERASE dataset [ERASE_Dataset](https://huggingface.co/datasets/Jia-py/ERASE_Dataset) to our dataset, in order to achieve a smaller storage and much more efficient read files speed.

### Prerequisites
All required packages are listed in `requirements.txt`.

## Running Experiments

The experimental pipeline consists of two stages:

### 1. Search and Retrain Stage
Run all feature selection methods:
```bash
python search_and_retrain.py
```
Results will be saved in the `exp_save` directory.

### 2. Results Analysis
Analyze the experimental results using:
```bash
jupyter notebook agg_results.ipynb
```

## 目录介绍
* `ckpt`: 存储训练好的模型文件目录
* `common`: fs_trainer涉及到训练的过程，有一些细节可以看看，例如验证集上更新门控参数，以及加载新优化器，继承老优化器参数等等...
* `exp_save`: 实验结果保存在此文件中。重训模型的时候，如果有一模一样的特征维度选择结果，重训的时候会复用`retrain_result.csv`里的数据。
* `models.fs`: 包含了emb dim search的方法，都是手动复习的论文，或者把论文的公开代码拷贝核心部分过来。
* `search_and_retrain.py`: 入口函数
* `agg_results.ipynb`: 用于分析实验结果的脚本。