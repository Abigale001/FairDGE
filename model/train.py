#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:project
@file: train
@platform: PyCharm
@time: 2023/7/12
"""
import time
from utils.load import load_data, load_test_data, load_label
from model.downstream import BaseModel
from torch.optim import Adam
import torch
import numpy as np
import json


def train(train_user_item_t, user_item_test_df, args, all_ui_dict):

    print(f'learning rate:{args.learning_rate}')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    gcn_encode = BaseModel(args)
    optimizer = Adam(params=gcn_encode.parameters(), lr=args.learning_rate,
                     weight_decay=args.weight_decay, eps=1e-8, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    predict_data = load_test_data(args, user_item_test_df, all_ui_dict)
    snapshot_data = []
    for snapshot_id, train_user_item in enumerate(train_user_item_t):
        data = load_data(args, snapshot_id, train_user_item, all_ui_dict)
        snapshot_data.append(data)

    data_label = load_label(args)

    all_data_features_list = []
    all_data_adj_norm_list = []
    all_data_adj_list = []
    all_train_edges_list = []
    all_train_edges_false_list = []

    for data in snapshot_data:
        all_data_features_list.append(data['features'])
        all_data_adj_norm_list.append(data['adj_train_norm'])
        all_data_adj_list.append(data['adj_train'])
        all_train_edges_list.append(data['train_edges'])
        all_train_edges_false_list.append(data['train_edges_false'])

    all_train_edges = torch.concat(all_train_edges_list, axis=0)
    all_train_edges_false = torch.concat(all_train_edges_false_list, axis=0)
    all_data_features = torch.stack(all_data_features_list, axis=0)
    all_data_adj_norm = torch.stack(all_data_adj_norm_list, axis=0)
    degree_np_list = []
    for adj_matrix in all_data_adj_list:
        D = np.array(np.sum(adj_matrix, axis=0))[0]
        degree_np_list.append(D)
    all_snapshot_deg_h = torch.tensor(np.array(degree_np_list)).unsqueeze(-1).expand(-1, -1, args.dim)

    if args.cuda is not None and int(args.cuda) >= 0:
        args.device = 'cuda:' + str(args.cuda)
        print(args.device)
        gcn_encode = gcn_encode.to(args.device)
        all_data_features = all_data_features.to(args.device)
        all_data_adj_norm = all_data_adj_norm.to(args.device)
        all_snapshot_deg_h = all_snapshot_deg_h.to(args.device)
        data_label = data_label.to(args.device)
        for x, val in predict_data.items():
            if torch.is_tensor(predict_data[x]):
                predict_data[x] = predict_data[x].to(args.device)

    best_metric = {}
    best_metric["hit_avg"] = {}
    best_metric["ndcg_avg"] = {}
    best_metric["prec_avg"] = {}
    for k in args.k_list:
        best_metric["hit_avg"][k] = 0.
        best_metric["ndcg_avg"][k] = 0.
        best_metric["prec_avg"][k] = 0.
    best_metric["rHR"] = 0.
    best_metric["rND"] = 0.
    best_fair_metric = {}
    best_fair_metric["rHR"] = 10000.
    best_fair_metric["rND"] = 10000.

    best_loss = 10
    count = 0
    train_loss_list = []
    training_time = 0.
    for epoch in range(args.epoch_num):

        gcn_encode.train()
        optimizer.zero_grad()
        start_time = time.time()
        embeddings, degree_embedding, pattern_embedding, pattern_prob = gcn_encode.encode(all_data_features, all_data_adj_norm, all_snapshot_deg_h)
        train_metrics = gcn_encode.compute_metrics(embeddings, pattern_prob, all_train_edges, all_train_edges_false, data_label)
        train_metrics['loss'].backward(retain_graph=True)
        optimizer.step()
        lr_scheduler.step()
        test_start_time = time.time()
        train_epoch_time = test_start_time - start_time

        train_loss_list.append(train_metrics["loss"].cpu().detach().numpy())

        if (epoch + 1) % args.eval_freq == 0:
            gcn_encode.eval()

            embeddings, degree_embedding, pattern_embedding, pattern_prob = gcn_encode.encode(all_data_features, all_data_adj_norm, all_snapshot_deg_h)
            val_metrics = gcn_encode.compute_metrics_test(embeddings, predict_data, data_label, 'predict')
            gcn_encode.train()
            print(
                f'Epoch {epoch + 1}: train loss:{train_metrics["loss"]:.4f}, train recall:{train_metrics["roc"]:.4f}, train prec:{train_metrics["ap"]:.4f}; '
                f'test loss:{val_metrics["loss"]:.4f}, hr@1:{val_metrics["hit_avg"][1]:.4f}, hr@5:{val_metrics["hit_avg"][5]:.4f}, '
                f'hr@10:{val_metrics["hit_avg"][10]:.4f}, hr@15:{val_metrics["hit_avg"][15]:.4f}, '
                f'hr@20:{val_metrics["hit_avg"][20]:.4f}, hr@40:{val_metrics["hit_avg"][40]:.4f}, '
                f'hr@60:{val_metrics["hit_avg"][60]:.4f}, hr@80:{val_metrics["hit_avg"][80]:.4f}, '
                f'hr@100:{val_metrics["hit_avg"][100]:.4f}, '
                f'ndcg@1:{val_metrics["ndcg_avg"][1]:.4f}, ndcg@5:{val_metrics["ndcg_avg"][5]:.4f}, '
                f'ndcg@10:{val_metrics["ndcg_avg"][10]:.4f}, ndcg@15:{val_metrics["ndcg_avg"][15]:.4f}, '
                f'ndcg@20:{val_metrics["ndcg_avg"][20]:.4f}, ndcg@40:{val_metrics["ndcg_avg"][40]:.4f}, '
                f'ndcg@60:{val_metrics["ndcg_avg"][60]:.4f}, ndcg@80:{val_metrics["ndcg_avg"][80]:.4f}, '
                f'ndcg@100:{val_metrics["ndcg_avg"][100]:.4f}, '
                f'prec@1:{val_metrics["prec_avg"][1]:.4f}, prec@5:{val_metrics["prec_avg"][5]:.4f}, '
                f'prec@10:{val_metrics["prec_avg"][10]:.4f}, prec@15:{val_metrics["prec_avg"][15]:.4f}, '
                f'prec@20:{val_metrics["prec_avg"][20]:.4f}, prec@40:{val_metrics["prec_avg"][40]:.4f}, '
                f'prec@60:{val_metrics["prec_avg"][60]:.4f}, prec@80:{val_metrics["prec_avg"][80]:.4f}, '
                f'prec@100:{val_metrics["prec_avg"][100]:.4f}, '
                f'rHR:{val_metrics["rHR"]:.4f}, rND:{val_metrics["rND"]:.4f},'
                f'train time: {train_epoch_time:.4f}, test time: {time.time() - test_start_time:.4f}, all_training_time: {training_time:.4f}')
            training_time += train_epoch_time

            if val_metrics[args.best_flag][args.best_k_flag] > best_metric[args.best_flag][args.best_k_flag]:
                # counter = 0
                for metric in ['hit_avg', 'ndcg_avg', 'prec_avg']:
                    for k in args.k_list:
                        best_metric[metric][k] = val_metrics[metric][k]
                best_metric["rHR"] = val_metrics["rHR"]
                best_metric["rND"] = val_metrics["rND"]
                best_metric['epoch'] = epoch + 1
                best_embedding = embeddings

            if val_metrics["rHR"] < best_fair_metric["rHR"]:
                best_fair_metric["rHR"] = val_metrics["rHR"]
                best_fair_metric["rND"] = val_metrics["rND"]

        torch.cuda.empty_cache()

    print(
        f'Epoch {best_metric["epoch"]}: '
        f'hr@1:{best_metric["hit_avg"][1]:.4f}, hr@5:{best_metric["hit_avg"][5]:.4f}, '
        f'hr@10:{best_metric["hit_avg"][10]:.4f}, hr@15:{best_metric["hit_avg"][15]:.4f}, '
        f'hr@20:{best_metric["hit_avg"][20]:.4f}, hr@40:{best_metric["hit_avg"][40]:.4f}, '
        f'hr@60:{best_metric["hit_avg"][60]:.4f}, hr@80:{best_metric["hit_avg"][80]:.4f}, '
        f'hr@100:{best_metric["hit_avg"][100]:.4f}, '
        f'ndcg@1:{best_metric["ndcg_avg"][1]:.4f}, ndcg@5:{best_metric["ndcg_avg"][5]:.4f}, '
        f'ndcg@10:{best_metric["ndcg_avg"][10]:.4f}, ndcg@15:{best_metric["ndcg_avg"][15]:.4f}, '
        f'ndcg@20:{best_metric["ndcg_avg"][20]:.4f}, ndcg@40:{best_metric["ndcg_avg"][40]:.4f}, '
        f'ndcg@60:{best_metric["ndcg_avg"][60]:.4f}, ndcg@80:{best_metric["ndcg_avg"][80]:.4f}, '
        f'ndcg@100:{best_metric["ndcg_avg"][100]:.4f}, '
        f'prec@1:{best_metric["prec_avg"][1]:.4f}, prec@5:{best_metric["prec_avg"][5]:.4f}, '
        f'prec@10:{best_metric["prec_avg"][10]:.4f}, prec@15:{best_metric["prec_avg"][15]:.4f}, '
        f'prec@20:{best_metric["prec_avg"][20]:.4f}, prec@40:{best_metric["prec_avg"][40]:.4f}, '
        f'prec@60:{best_metric["prec_avg"][60]:.4f}, prec@80:{best_metric["prec_avg"][80]:.4f}, '
        f'prec@100:{best_metric["prec_avg"][100]:.4f}, '
        f'rHR:{best_metric["rHR"]:.4f}, rND:{best_metric["rND"]:.4f},'
        f'best_rHR:{best_fair_metric["rHR"]:.4f}, best_rND:{best_fair_metric["rND"]:.4f}')
