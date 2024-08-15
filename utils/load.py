#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:project
@file: load
@platform: PyCharm
@time: 2023/7/12
"""
import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch
from random import choices
import os.path
import pickle


def load_data_lp(args, df):
    ui_df = df[[0, 1]]
    idx_counter = 0
    object_to_idx = {}
    edges = []
    for rowid, row in ui_df.iterrows():
        n1, n2 = row[0], row[1]
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))

        adj = np.zeros((args.node_num, args.node_num), dtype='uint8')
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1

    features = np.random.rand(args.node_num, args.dim)
    data = {'adj_train': adj, 'features': features}
    return data, ui_df

def load_test_data_lp(args, df):
    ui_df = df[[0, 1]]
    idx_counter = 0
    object_to_idx = {}
    edges = []
    for rowid, row in ui_df.iterrows():
        n1, n2 = row[0], row[1]
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))

    adj_predict = np.zeros((args.node_num, args.node_num), dtype='uint8')
    for i, j in edges:
        adj_predict[i, j] = 1  
        adj_predict[j, i] = 1

    data = {'adj_predict': adj_predict}
    return data, ui_df


def load_label(args):
    label_dict_file = args.ROOT_DIR + '/data/'+args.data_name+'/label/data_label.pkl'
    label_dict = pickle.load(open(label_dict_file, 'rb'))
    item_list = range(args.user_num, args.user_num+args.item_num)
    label_list = []
    for item_id in item_list:
        v = label_dict[item_id]
        if v == 'T': # tail-to-head
            label_list.append(0)
        elif v == 'F': # fluctuation-at-tail
            label_list.append(1)
        elif v == 'N': # start-from-head
            label_list.append(2)
    label_tensor = torch.LongTensor(label_list)
    return label_tensor


def load_data(args, snapshot_id, train_user_item_df, all_ui_dict):
    data, ui_df = load_data_lp(args, train_user_item_df)
    adj_train, train_edges, train_edges_false = mask_edges(args, data['adj_train'], ui_df, snapshot_id, all_ui_dict)
    data['adj_train'] = adj_train
    data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
    data['adj_train_norm'], data['features'] = process(
        data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    return data

def load_test_data(args, user_item_df, all_ui_dict):
    predict_data, ui_df = load_test_data_lp(args, user_item_df)
    predict_edges, predict_edges_false = mask_test_edges(args, ui_df, all_ui_dict)
    predict_data['predict_edges'], predict_data['predict_edges_false'] = predict_edges, predict_edges_false
    return predict_data


def mask_edges(args, adj, ui_df, snapshot_id, all_ui_dict):
    seed = 2
    np.random.seed(seed)
    pos_edges = ui_df.to_numpy()
    np.random.shuffle(pos_edges)

    train_edges_false = negative_sampling(args, all_ui_dict, ui_df, snapshot_id)

    adj_train = sp.csr_matrix((np.ones(pos_edges.shape[0]), (pos_edges[:, 0], pos_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(pos_edges), torch.LongTensor(train_edges_false.to_numpy())

def mask_test_edges(args, ui_df, all_ui_dict):
    predict_pos_edges = ui_df.to_numpy()
    predict_pos_edges_ = predict_pos_edges.reshape((args.test_batch_size, -1, 2))

    predict_edges_false = negative_sampling(args, all_ui_dict, ui_df, -1, 'predict').to_numpy()
    predict_edges_false_ = predict_edges_false.reshape((args.test_batch_size, -1, 2))

    return torch.LongTensor(predict_pos_edges_), torch.LongTensor(predict_edges_false_)

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def negative_sampling(args, all_ui_dict, pos_df, snapshot_id=-1, data_status='train'):
    neg_folder = args.ROOT_DIR + '/data/' + args.data_name + '/neg/'

    if snapshot_id == -1:
        neg_sampling_num = args.test_neg_num
        if data_status == 'predict':
            neg_csv = neg_folder + 'user_item_predict_neg.relation'

    elif snapshot_id != -1:
        neg_sampling_num = args.train_neg_num
        neg_folder = args.ROOT_DIR + '/data/' + args.data_name + '/neg/'+ str(args.split_t_num) + '/'
        neg_csv = neg_folder + 'user_item_train_neg.' + str(snapshot_id) + '.relation'

    if not os.path.exists(neg_folder):
        os.mkdir(neg_folder)

    if os.path.isfile(neg_csv):
        neg_df = pd.read_csv(neg_csv, header=None, sep='\t')
        print(f'load train neg file finish: {neg_csv}')
        return neg_df
    # This step may take long time for sampling.
    negative_list = []
    for idx, row in pos_df.iterrows():
        uid, pos_iid = row
        remain = list(args.all_i_set.difference(set(all_ui_dict[uid])))
        negative = [[uid, item] for item in choices(remain, k=neg_sampling_num)]
        negative_list += negative

    neg_df = pd.DataFrame(negative_list)

    neg_df.to_csv(neg_csv, header=None, index=None, sep='\t')
    print(f'dump {data_status} neg file finish: {neg_csv} ')

    return neg_df

