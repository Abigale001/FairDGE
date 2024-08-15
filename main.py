#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:project
@file: main
@platform: PyCharm
@time: 2023/7/10
"""

import sys,os
sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import pandas as pd
from utils.split_t import split_df
from model.train import train

from utils.config import parser
import pickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    args = parser.parse_args()
    print(args)
    args.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    processed_train_csv = args.ROOT_DIR + '/data/'+args.data_name+'/user_item_train.relation'
    processed_predict_csv = args.ROOT_DIR + '/data/'+args.data_name+'/user_item_predict.relation'

    user_item_df = pd.read_csv(processed_train_csv, header=None, sep=',')
    user_item_predict_df = pd.read_csv(processed_predict_csv, header=None, sep=',')

    ui_dict_file = args.ROOT_DIR + '/data/'+args.data_name+'/all_ui_dict.pkl'
    all_ui_dict = pickle.load(open(ui_dict_file, 'rb'))
    print(f'load all_ui_dict file:{ui_dict_file} finish')

    args.user_num = max(user_item_df[0])+1
    args.item_num = max(user_item_df[1])+1-args.user_num
    args.node_num = args.user_num + args.item_num
    args.all_u_set = set(range(args.user_num))
    args.all_i_set = set(range(args.user_num, args.user_num+args.item_num))


    # split n snapshots
    train_user_item_t, t_list = split_df(user_item_df, args)

    train(train_user_item_t, user_item_predict_df, args, all_ui_dict)

