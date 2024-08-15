#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:project
@file: downstream
@platform: PyCharm
@time: 2023/7/12
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.FairDGE as encoders
import math
import numpy as np
import utils.utils_math as umath
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import eva
from collections import defaultdict
import os
import pickle
from utils.eva import rHR, rND


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.nnodes = args.node_num
        if self.args.data_name == 'MovieLens':
            self.encoder = getattr(encoders, args.model)(args)
        else:
            print('wrong data name')
            exit(0)
        self.classification_loss = torch.nn.CrossEntropyLoss()

    def deg_encode(self, snapshot_data):
        adj_snapshots = []
        for data in snapshot_data:
            adj = data['adj_train']
            adj_snapshots.append(adj)
        degree_np_list = []
        for adj in adj_snapshots:
            D = np.array(np.sum(adj, axis=0))[0]
            degree_np_list.append(D)
        all_snapshot_deg = torch.tensor(np.array(degree_np_list)).unsqueeze(-1).expand(-1,-1,128).to('cuda:'+self.args.cuda)
        all_snapshot_deg = all_snapshot_deg.type(torch.FloatTensor).to('cuda:'+self.args.cuda)
        de_h = self.degree_encoder.encode(all_snapshot_deg).to('cuda:'+self.args.cuda)
        return de_h

    def encode(self, x_t, adj_t, deg_t):
        h, degree_h, pattern_emb, pattern_prob = self.encoder.forward(x_t, adj_t, deg_t)
        return h, degree_h, pattern_emb, pattern_prob

    def compute_metrics(self, embeddings, pattern_prob, pos_edges, edges_false, data_label):

        pos_scores = self.decode(embeddings.clone(), pos_edges)
        neg_scores = self.decode(embeddings.clone(), edges_false)
        loss_pos = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss = loss_pos + F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))

        ########### classification loss ###########
        pattern_prob_ = pattern_prob[self.args.user_num:]
        c_loss = self.classification_loss(pattern_prob_, data_label)

        ########### contrastive learning loss ###########
        item_embedding = embeddings[self.args.user_num:]
        sampled_item_neg_1, sampled_item_neg_2 = self.sample_neg(item_embedding, data_label)
        contras_loss = self.cal_constras_loss(sampled_item_neg_1, sampled_item_neg_2)

        ########### fairness loss ###########
        T2H_id = (data_label == 0).nonzero().squeeze().cpu().detach().numpy() + self.args.user_num
        FaT_id = (data_label == 1).nonzero().squeeze().cpu().detach().numpy() + self.args.user_num
        SfH_id = (data_label == 2).nonzero().squeeze().cpu().detach().numpy() + self.args.user_num

        T2H_bias_pos_edges = pos_edges[[x[0].item() in T2H_id or x[1].item() in T2H_id for x in pos_edges]]
        T2H_bias_false_edges = edges_false[[x[0].item() in T2H_id or x[1].item() in T2H_id for x in pos_edges]]

        SfH_bias_pos_edges = pos_edges[[x[0].item() in SfH_id or x[1].item() in SfH_id for x in pos_edges]]
        SfH_bias_false_edges = edges_false[[x[0].item() in SfH_id or x[1].item() in SfH_id for x in pos_edges]]

        T2H_bias_pos_scores = self.decode(embeddings.clone(), T2H_bias_pos_edges)
        T2H_bias_neg_scores = self.decode(embeddings.clone(), T2H_bias_false_edges)
        loss_T2H_bias = F.binary_cross_entropy(T2H_bias_pos_scores, torch.ones_like(T2H_bias_pos_scores)) + F.binary_cross_entropy(T2H_bias_neg_scores, torch.zeros_like(T2H_bias_neg_scores))

        SfH_bias_pos_scores = self.decode(embeddings.clone(), SfH_bias_pos_edges)
        SfH_bias_neg_scores = self.decode(embeddings.clone(), SfH_bias_false_edges)
        loss_SfH_bias = F.binary_cross_entropy(SfH_bias_pos_scores, torch.ones_like(SfH_bias_pos_scores)) + F.binary_cross_entropy(SfH_bias_neg_scores, torch.zeros_like(SfH_bias_neg_scores))
        loss_fair = torch.linalg.norm(loss_T2H_bias - loss_SfH_bias)


        l1_norm = sum(abs(p.pow(1.0)).sum() for p in self.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())

        total_loss = self.args.CE_loss_weight*loss + \
                     self.args.Class_loss_weight*c_loss + \
                     self.args.CL_loss_weight*contras_loss + \
                     self.args.Fair_loss_weight*loss_fair + \
                     self.args.l1_lambda * l1_norm + self.args.l2_lambda * l2_norm

        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': total_loss, 'roc': roc, 'ap': ap}
        return metrics

    def compute_metrics_test(self, embeddings, data, data_label, split):

        edges_false = data[f'{split}_edges_false']
        edges = data[f'{split}_edges']
        assert edges.shape[0] == edges_false.shape[0]

        all_pos_num = edges.shape[0] * edges.shape[1]
        batch_num = edges.shape[0]
        all_hits = defaultdict(float)
        all_ndcg = defaultdict(float)
        all_prec = defaultdict(float)
        all_recall = defaultdict(float)

        all_T2H_hits = defaultdict(float)
        all_FaT_hits = defaultdict(float)
        all_SfH_hits = defaultdict(float)

        all_rHR = 0.
        all_rND = 0.
        hit_avg = defaultdict(float)
        ndcg_avg = defaultdict(float)
        prec_avg = defaultdict(float)
        recall_avg = defaultdict(float)

        T2H_hit_avg = defaultdict(float)
        FaT_hit_avg = defaultdict(float)
        SfH_hit_avg = defaultdict(float)

        rHR_avg = 0.
        rND_avg = 0.
        metric = defaultdict()
        metric['loss'] = 0.
        d_loss = 0.
        T2H_id = (data_label == 0).nonzero().squeeze().cpu().detach().numpy() + self.args.user_num
        FaT_id = (data_label == 1).nonzero().squeeze().cpu().detach().numpy() + self.args.user_num
        SfH_id = (data_label == 2).nonzero().squeeze().cpu().detach().numpy() + self.args.user_num
        T2H_id_num = len(T2H_id)
        FaT_id_num = len(FaT_id)
        SfH_id_num = len(SfH_id)
        max_rHR=0.
        max_rND=0.
        T2H_scores_dict_list = defaultdict(list)
        FaT_scores_dict_list = defaultdict(list)
        SfH_scores_dict_list = defaultdict(list)
        for k in self.args.k_list:
            max_rHR += 1-T2H_id_num/self.args.item_num
            max_rND += 1/(math.log2(k)+1e-6)-T2H_id_num/self.args.item_num

        for batch_id in range(batch_num):
            indices = [batch_id]
            this_batch_edges = edges[indices, :, :].squeeze()
            this_batch_edges_false = edges_false[indices, :, :].squeeze()
            this_batch_pos_scores = self.decode(embeddings, this_batch_edges)
            this_batch_neg_scores = self.decode(embeddings, this_batch_edges_false)

            T2H_bias_pos_edges_id = [x[0].item() in T2H_id or x[1].item() in T2H_id for x in this_batch_edges]
            FaT_bias_pos_edges_id = [x[0].item() in FaT_id or x[1].item() in FaT_id for x in this_batch_edges]
            SfH_bias_pos_edges_id = [x[0].item() in SfH_id or x[1].item() in SfH_id for x in this_batch_edges]

            if this_batch_pos_scores.is_cuda:
                this_batch_pos_scores = this_batch_pos_scores.cpu()
                this_batch_neg_scores = this_batch_neg_scores.cpu()
            loss = F.binary_cross_entropy(this_batch_pos_scores, torch.zeros_like(this_batch_pos_scores))
            loss += F.binary_cross_entropy(this_batch_neg_scores, torch.zeros_like(this_batch_neg_scores))

            this_batch_pos_scores = this_batch_pos_scores.detach().numpy()
            this_batch_neg_scores = this_batch_neg_scores.detach().numpy()
            pos_num = this_batch_pos_scores.shape[0]
            for pos_id in range(pos_num):
                pos_score = np.take(this_batch_pos_scores, pos_id) # np
                neg_score = np.take(this_batch_neg_scores, range(pos_id*self.args.test_neg_num,(pos_id+1)*self.args.test_neg_num))

                neg_edge = this_batch_edges_false.index_select(0,torch.tensor(range(pos_id*self.args.test_neg_num,(pos_id+1)*self.args.test_neg_num)).to(self.args.device)).cpu().detach().numpy()[:,-1]
                pos_neg_item_ids = np.insert(neg_edge, 0, pos_id)

                this_pos_score = np.concatenate((pos_score, neg_score), axis=None)
                sorted_s = np.argsort(-this_pos_score)
                hit_s = (sorted_s == 0).astype(int)
                ground_truth_id = [np.argwhere(hit_s == 1).item()]

                sorted_pos_neg_item_ids = pos_neg_item_ids[sorted_s]
                for k in self.args.k_list:
                    this_hits = eva.hit_at_k(hit_s, k)
                    this_ndcg = eva.ndcg_at_k(hit_s, k, ground_truth_id, 0)
                    this_prec = eva.precision_at_k(hit_s, k)
                    this_recall = eva.recall_at_k(hit_s, k, 1.0)
                    all_hits[k] += this_hits
                    all_ndcg[k] += this_ndcg
                    all_prec[k] += this_prec
                    all_recall[k] += this_recall

                    if T2H_bias_pos_edges_id[pos_id]:
                        all_T2H_hits[k] += this_hits
                        T2H_scores_dict_list[k].append(this_hits)
                    if FaT_bias_pos_edges_id[pos_id]:
                        all_FaT_hits[k] += this_hits
                        FaT_scores_dict_list[k].append(this_hits)
                    if SfH_bias_pos_edges_id[pos_id]:
                        all_SfH_hits[k] += this_hits
                        SfH_scores_dict_list[k].append(this_hits)
                    all_rHR += rHR(sorted_pos_neg_item_ids, k, T2H_id, self.args.item_num)
                    all_rND += rND(sorted_pos_neg_item_ids, k, T2H_id, self.args.item_num)
            d_loss += loss
        d_loss /= batch_num
        metric['loss'] = d_loss

        for k in self.args.k_list:
            hit_avg[k] = all_hits[k] / float(all_pos_num)
            ndcg_avg[k] = all_ndcg[k] / float(all_pos_num)
            prec_avg[k] = all_prec[k] / float(all_pos_num)
            recall_avg[k] = all_recall[k] / float(all_pos_num)

            T2H_hit_avg[k] = all_T2H_hits[k] / float(T2H_id_num)
            FaT_hit_avg[k] = all_FaT_hits[k] / float(FaT_id_num)
            SfH_hit_avg[k] = all_SfH_hits[k] / float(SfH_id_num)

        rHR_avg = all_rHR/float(all_pos_num)/max_rHR
        rND_avg = all_rND/float(all_pos_num)/max_rND

        metric['hit_avg'] = hit_avg
        metric['ndcg_avg'] = ndcg_avg
        metric['prec_avg'] = prec_avg
        metric['recall_avg'] = recall_avg

        metric['rHR'] = rHR_avg
        metric['rND'] = rND_avg

        metric['T2H_hit_avg'] = T2H_hit_avg
        metric['FaT_hit_avg'] = FaT_hit_avg
        metric['SfH_hit_avg'] = SfH_hit_avg

        metric['T2H_hit_dict'] = T2H_scores_dict_list
        metric['FaT_hit_dict'] = FaT_scores_dict_list
        metric['SfH_hit_dict'] = SfH_scores_dict_list
        return metric


    def sample_neg(self, item_degree_embedding, data_label):
        item_emb_np = item_degree_embedding.cpu().detach().numpy()
        sampled_item_neg_1_list = []
        sampled_item_neg_2_list = []

        sampled_ids_file = self.args.ROOT_DIR + '/data/'+self.args.data_name+'/contras_sampled_ids.pkl'
        if os.path.isfile(sampled_ids_file):
            sampled_items_ids = pickle.load(open(sampled_ids_file, 'rb'))
            for item_id, item_emb in enumerate(item_emb_np):
                sampled_item_neg_1_list.append(item_degree_embedding[sampled_items_ids[item_id][0], :])
                sampled_item_neg_2_list.append(item_degree_embedding[sampled_items_ids[item_id][1], :])
        else:
            sampled_items_ids = defaultdict()
            T2H_id = (data_label == 0).nonzero().squeeze().cpu().detach().numpy()
            FaT_id = (data_label == 1).nonzero().squeeze().cpu().detach().numpy()
            SfH_id = (data_label == 2).nonzero().squeeze().cpu().detach().numpy()

            data_label_np = data_label.cpu().detach().numpy()

            for item_id, item_emb in enumerate(item_emb_np):
                this_item_label = data_label_np[item_id]
                if this_item_label == 0:
                    sampled_id_1 = np.random.choice(a=FaT_id, size=1, replace=False)
                    sampled_id_2 = np.random.choice(a=SfH_id, size=1, replace=False)
                    sampled_item_neg_1_list.append(item_degree_embedding[sampled_id_1,:])
                    sampled_item_neg_2_list.append(item_degree_embedding[sampled_id_2,:])
                elif this_item_label == 1:
                    sampled_id_1 = np.random.choice(a=T2H_id, size=1, replace=False)
                    sampled_id_2 = np.random.choice(a=SfH_id, size=1, replace=False)
                    sampled_item_neg_1_list.append(item_degree_embedding[sampled_id_1,:])
                    sampled_item_neg_2_list.append(item_degree_embedding[sampled_id_2,:])
                if this_item_label == 2:
                    sampled_id_1 = np.random.choice(a=T2H_id, size=1, replace=False)
                    sampled_id_2 = np.random.choice(a=FaT_id, size=1, replace=False)
                    sampled_item_neg_1_list.append(item_degree_embedding[sampled_id_1,:])
                    sampled_item_neg_2_list.append(item_degree_embedding[sampled_id_2,:])
                sampled_items_ids[item_id] = [sampled_id_1,sampled_id_2]
            pickle.dump(sampled_items_ids, open(sampled_ids_file, 'wb'))
        sampled_item_neg_1 = torch.squeeze(torch.stack(sampled_item_neg_1_list))
        sampled_item_neg_2 = torch.squeeze(torch.stack(sampled_item_neg_2_list))

        return sampled_item_neg_1, sampled_item_neg_2

    def cal_constras_loss(self, pos, neg):
        pos_neg_embeds = torch.cat([pos, neg], dim=1)
        contrastive_scores = self.sim_score(pos_neg_embeds, pos_neg_embeds)
        contras_loss = self.contrastive_loss(contrastive_scores)
        return contras_loss

    def sim_score(self, a, b, tau=1.0, eps=1e-8):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1)) / tau
        return sim_mt

    def contrastive_loss(self, scores):
        l = scores.shape[0]
        mask = torch.eye(l, device=scores.device).float() * -1e8
        scores = scores + mask
        label = torch.zeros_like(scores)
        row_idx = [i for i in range(l)]
        column_idx = [(i+l/2) % l for i in range(l)]
        label[row_idx, column_idx] = 1
        loss = -F.log_softmax(scores, dim=-1)
        loss = torch.sum((loss * label).flatten().unsqueeze(-1))
        return loss*1e-5


    def decode(self, h, idx):
        h = umath.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = umath.sqdist(emb_in, emb_out)
        probs = 1. / (torch.exp((sqdist - self.args.r) / self.args.t) + 1.0)
        return probs




