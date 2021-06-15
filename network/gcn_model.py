import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from non_local import _GraphNonLocal
import pdb

import torch.nn as nn

class SemGCN(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(5, 3), num_layers=4, non_local=False, p_dropout=None, learn_mask=False):
        super(SemGCN, self).__init__()
        self.num_jts = len(adj)
        self.coords_dim = coords_dim
        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout, learn_mask=learn_mask)]
        _gconv_layers = []

        if not non_local:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout, learn_mask=learn_mask))
        else:
            grouped_order = torch.arange(len(adj))
            restored_order = torch.arange(len(adj))
            group_size = 1
            _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
                _gconv_layers.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, coords_dim[1], adj, learn_mask=learn_mask)

    def set_bn_momentum(self, momentum):
        pass

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_jts, self.coords_dim[0])
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout, learn_mask=False):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout, learn_mask=learn_mask)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout, learn_mask=learn_mask)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None, learn_mask=False):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj, learn_mask=learn_mask)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True, learn_mask=False):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.learn_mask = learn_mask
        if learn_mask:
            self.m = nn.Parameter(torch.ones_like(self.adj))
            self.init_mask()
        else:
            # self.m = self.adj > 0
            self.m = self.adj.cuda()
        # self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        # nn.init.constant_(self.e.data, 1)
        # self.e = nn.Parameter(self.adj.clone())
        self.e = nn.Parameter(torch.ones_like(self.adj))
        # self.e = nn.Parameter(torch.ones(1, len(self.m.nonzero())))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        # self.binarize = Binarize.apply
        self.binarize = BinaryStraightThroughFunction.apply

    def init_mask(self):
        init_type = '1nn'
        if init_type == 'constant':
            nn.init.constant_(self.m, 1)
        elif init_type == '1nn':
            with torch.no_grad():
                self.m.data = self.adj.clone()
                self.m.data[self.m.data == 0] = -1
        elif init_type == 'random':
            # random init
            nn.init.kaiming_uniform_(self.m, a=math.sqrt(5))
            # self.m.diag set to 1
            with torch.no_grad():
                for i in range(len(self.m)):
                    self.m[i, i] = 1

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        # adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        # adj[self.m] = self.e
        if self.learn_mask:
            # self.m.data.clamp(-1, 1)
            # mask = Binarize.apply(self.m)
            mask = self.binarize(self.m)
            eye = torch.eye(len(self.adj)).to(mask.device)
            mask = mask*(1-eye) + eye       # keep diag elment = 1
        else:
            mask = self.m
        adj = mask * self.e + (1-mask) * (-9e15)

        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask):
        ctx.save_for_backward(mask)
        prob = hard_sigmoid(mask)
        binary_mask = (torch.rand(mask.shape).to(mask.device) <= prob).float()
        return binary_mask
    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors
        grad_m = grad_output.clone()
        grad_m = grad_m * where(torch.abs(input[0]) <= 1, 1, 0)
        return grad_m

def where(cond, x1, x2):
    return cond.float() * x1 + (1 - cond.float()) * x2

class BinaryStraightThroughFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = where(input>=0, 1, 0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        grad_input = grad_output.clone()
        grad_input = grad_input * where(torch.abs(input[0]) <= 1, 1, 0)
        return grad_input

def hard_sigmoid(x):
    x = (x+1.) / 2
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))

def freeze_param(model, frozen_key):
    if isinstance(model, SemGraphConv):
        model.m.requires_grad = 'm' not in frozen_key
        model.e.requires_grad = 'e' not in frozen_key
    else:
        if len(model._modules) > 0:
            for submodel in model._modules.values():
                freeze_param(submodel, frozen_key)

def adj_prior_loss(model):
    if isinstance(model, SemGraphConv):
        loss = torch.abs(Binarize.apply(model.m).sum() - model.adj.sum())
        num = 1
    else:
        loss = 0
        num = 0
        if len(model._modules) > 0:
            for submodel in model._modules.values():
                subloss, subnum = adj_prior_loss(submodel)
                if subnum > 0:
                    loss += subloss
                    num += subnum

    return loss, num
