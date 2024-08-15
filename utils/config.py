#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:project
@file: config
@platform: PyCharm
@time: 2023/8/10
"""

import argparse
from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'test_batch_size':(2, 'batch_size for testing'), # 2 # 2027 # 4054
        'learning_rate': (5e-4, 'learning rate'),
        'cuda': ('0', 'which cuda device to use (-1 for cpu training)'),
        'epoch_num': (400, 'maximum number of epochs to train for'),
        'weight_decay': (1e-7, 'l2 regularization strength'),
        'lr_reduce_freq': (30, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'eval_freq': (1, 'how often to compute val metrics (in epochs)'),
        'seed': (1111, 'seed for training'),
    },
    'model_config': {
        'model': ('FairDGE', 'FairDGE'),
        'enc': ('GCN', 'encoder of our model, can be any of [GCN, GAT, GraphSage]'),
        'dim': (64, 'embedding dimension'),
        'ablation': ('dynamic', 'dynamic or static or no_CL or no_degree or no_classification_loss'),
        'if_two_layer': (False, 'if there is cnn to embed degree'),
        'seq_layer': (3, 'number of sequence layers'),
        'n_heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'dropout': (0.1, 'dropout'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double_precision': (0, '0 or 1 to set double precision of model'),
        'l1_lambda': (1e-7, 'l1 loss hyperparameter'),
        'l2_lambda': (1e-7, 'l2 loss hyperparameter'),
        'CE_loss_weight': (0.25, 'weight of downstream task loss'),
        'Class_loss_weight': (0.25, 'weight of classification loss'),
        'CL_loss_weight': (0.25, 'weight of contrastive learning loss'),
        'Fair_loss_weight': (0.25, 'weight of fair loss'),
        'head_threshold':(0.1, 'head threshold'),
    },
    'data_config': {
        'data_name': ('MovieLens', 'dataset'),
        'test_neg_num': (500, 'negative sampling number for test'),
        'train_neg_num': (1, 'negative sampling number for train'),
        'split_t_num': (15, 'split how many snapshots'),
        'k_list': ([1,5,10,15,20,40,60,80,100], 'evaluation K for hr, ndcg and prec.'),
        'use_feats': (0, 'whether to use initialized features'),
        'normalize_feats': (0, 'whether to normalize input node features'),
        'normalize_adj': (0, 'whether to row-normalize the adjacency matrix'),
        'init_gru': (False, 'if init gru'),
        'best_flag': ('hit_avg', 'hit_avg or ndcg_avg or prec_avg'),
        'best_k_flag': (20, 'one of the number in k_list'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
