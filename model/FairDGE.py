#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@project:project
@file: gcn
@platform: PyCharm
@time: 2023/7/12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import math


def get_dim_act(args):
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.dim] + ([args.dim] * (args.num_layers - 1))
    dims += [args.dim]
    acts += [act]
    return dims, acts

class FairDGE(Module):
    def __init__(self, args):
        super(FairDGE, self).__init__()
        self.args = args
        self.dropout = args.dropout
        in_features = args.dim
        out_features = args.dim
        use_bias = args.bias
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.n_classes = 2
        if args.enc == 'GCN':
            self.gnn_1 = GraphConvolution(args, args.dim, args.dim, args.dropout, args.act, args.bias)
            self.gnn_2 = GraphConvolution(args, args.dim, args.dim, args.dropout, args.act, args.bias)
        elif args.enc == 'GAT':
            self.gnn_1 = GraphAttentionLayer(args, args.dim, args.dim, args.dropout, args.act, args.alpha, args.n_heads, concat=False)
            self.gnn_2 = GraphAttentionLayer(args, args.dim, args.dim, args.dropout, args.act, args.alpha, args.n_heads, concat=False)
        elif args.enc == 'GraphSage':
            self.gnn_1 = GraphSage(args.dim, args.dim)
            self.gnn_2 = GraphSage(args.dim, args.dim)

        self.seq_module = torch.nn.GRU(self.args.dim, self.args.dim, self.args.seq_layer, batch_first=False)
        if self.args.init_gru:
            for layer in range(self.args.seq_layer):
                for weight in self.seq_module._all_weights[layer]:
                    if "weight" in weight:
                        nn.init.xavier_uniform_(getattr(self.seq_module, weight))
                    if "bias" in weight:
                        nn.init.uniform_(getattr(self.seq_module, weight))
        if self.args.if_two_layer == True:
            self.deg_cnn = torch.nn.Conv1d(in_channels=self.args.dim, out_channels=self.args.dim, kernel_size=3, padding=1)
        self.deg_seq = torch.nn.GRU(self.args.dim, self.args.dim, self.args.seq_layer, batch_first=False)
        if self.args.init_gru:
            for layer in range(self.args.seq_layer):
                for weight in self.deg_seq._all_weights[layer]:
                    if "weight" in weight:
                        nn.init.xavier_uniform_(getattr(self.deg_seq, weight))
                    if "bias" in weight:
                        nn.init.uniform_(getattr(self.deg_seq, weight))
        self.fusion = torch.nn.Linear(in_features=args.dim*2, out_features=args.dim)

        self.pattern_classification_1 = torch.nn.Linear(in_features=args.dim, out_features=args.dim)
        self.pattern_classification_2 = torch.nn.Linear(in_features=args.dim, out_features=3)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, all_features, all_adj, all_deg):
        if self.args.enc == 'GCN':
            input_gnn_1 = (all_features, all_adj)
            output_gnn_1 = self.gnn_1(input_gnn_1)
            all_features_, all_adjs = self.gnn_2(output_gnn_1)
        elif self.args.enc == 'GAT':
            all_features_list = []
            for time_id in range(self.args.split_t_num):
                t_feature = all_features[time_id, :, :]
                t_adj = all_adj.cpu().detach().to_dense()[time_id,:,:].to_sparse().to(self.args.device)
                input_gnn_1 = (t_feature, t_adj)
                output_gnn_1 = self.gnn_1(input_gnn_1)
                t_features_, t_adj_ = self.gnn_2(output_gnn_1)
                all_features_list.append(t_features_)
            all_features_ = torch.stack(all_features_list, dim=0)
        elif self.args.enc == 'GraphSage':
            all_features_list = []
            for time_id in range(self.args.split_t_num):
                t_feature = all_features[time_id, :, :]
                t_adj = all_adj.cpu().detach().to_dense()[time_id,:,:].to_sparse().to(self.args.device)
                output_gnn_1 = self.gnn_1(t_feature, t_adj)
                t_features_ = self.gnn_2(output_gnn_1, t_adj)
                all_features_list.append(t_features_)
            all_features_ = torch.stack(all_features_list, dim=0)

        if self.args.double_precision:
            all_deg = all_deg.type(torch.DoubleTensor).to('cuda:' + self.args.cuda)  # 5,21705,64
        else:
            all_deg = all_deg.type(torch.FloatTensor).to('cuda:' + self.args.cuda)  # 5,21705,64
        if self.args.if_two_layer == True:
            seq_features, _ = self.seq_module(all_features_)
            all_deg_p = all_deg.permute(1,2,0)
            de_h_conv = self.deg_cnn(all_deg_p).permute(2,0,1)
            de_h, _ = self.deg_seq(all_deg)
            degree_h = torch.stack((de_h_conv, de_h), dim=0).mean(dim=0)
            deg_structure = torch.concat([seq_features, degree_h], axis=-1)
            fused_deg_structure = self.fusion(deg_structure)[-1]
        else:
            seq_features, _ = self.seq_module(all_features_)
            _, de_h = self.deg_seq(all_deg)
            degree_h = de_h[-1]
            deg_structure = torch.concat([seq_features[-1,:,:], degree_h], axis=1)
            fused_deg_structure = self.fusion(deg_structure)
        hidden_ = self.pattern_classification_1(fused_deg_structure)
        pattern_emb = self.pattern_classification_2(hidden_)
        hid_relu = self.relu(pattern_emb)
        pattern_prob = self.softmax(hid_relu).squeeze()
        return fused_deg_structure, degree_h, pattern_emb, pattern_prob


class GraphConvolution(Module):
    """
    Simple GCN layer.
    """

    def __init__(self, args, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.args = args
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = getattr(F, args.act)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input):
        x, adj = input
        x = x.to(self.args.device)
        adj = adj.to(self.args.device)
        hidden = self.linear.forward(x)
        hidden_ = F.dropout(hidden, self.dropout, training=self.training)
        support = torch.bmm(adj, hidden_)
        output = self.act(support), adj
        return output


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self):
        super(Encoder, self).__init__()

    def encode(self, x_t, adj_t):
        input = (x_t, adj_t)
        output = self.layers.forward(input)
        return output


class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, args):
        super(GCN, self).__init__()
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        self.gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            self.gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*self.gc_layers)
        self.encode_graph = True


class GraphSage(nn.Module):
    def __init__(self, infeat, outfeat):
        super(GraphSage, self).__init__()
        self.infeat = infeat
        self.model_name = 'Graphsage'
        self.W = nn.Parameter(torch.zeros(size=(2 * infeat, outfeat)))
        self.bias = nn.Parameter(torch.zeros(outfeat))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        h1 = torch.spmm(adj, input).to_dense()
        degree = adj.to_dense().sum(axis=1).repeat(self.infeat, 1).T
        degree_ = degree+1e-6
        h1 = h1/degree_
        h1 = torch.cat([input, h1], dim=1)
        h1 = torch.mm(h1, self.W)
        return h1

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, args, in_features, out_features, dropout, alpha, activation):
        super(SpGraphAttentionLayer, self).__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.act = activation

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj._indices()

        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.to(self.args.device)
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), ones)

        if torch.count_nonzero(e_rowsum).item() > 0:
            e_rowsum = e_rowsum + 1e-5

        edge_e = self.dropout(edge_e)

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        return self.act(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    def __init__(self, args, input_dim, output_dim, dropout, activation, alpha, nheads, concat):
        """Sparse version of GAT."""
        super(GraphAttentionLayer, self).__init__()
        self.args = args
        self.dropout = dropout
        self.output_dim = output_dim
        self.act = getattr(F, activation)
        self.attentions = [SpGraphAttentionLayer(args, input_dim,
                                                 output_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 activation=self.act) for _ in range(nheads)]
        self.concat = concat
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, input):
        x, adj = input
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:
            h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            h_cat = torch.cat([att(x, adj).view((-1, self.output_dim, 1)) for att in self.attentions], dim=2)
            h = torch.mean(h_cat, dim=2)
        h = F.dropout(h, self.dropout, training=self.training)
        return (h, adj)
