import torch.nn as nn
import torch

ReLU = nn.ReLU

class LinearBlock(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5, biased=True):
        super(LinearBlock, self).__init__()
        self.l_size = linear_size

        self.relu = ReLU()
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size, bias=biased)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size, bias=biased)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def set_bn_momentum(self, momentum):
        self.batch_norm1.momentum = momentum
        self.batch_norm2.momentum = momentum

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class LinearModel(nn.Module):
    def __init__(self,
                 hidden_size=1024,
                 num_block=2,
                 num_jts=17,
                 out_num_jts=17,
                 p_dropout=0.25,
                 input_dim=2,
                 output_dim = 3):
        super(LinearModel, self).__init__()

        self.linear_size = hidden_size
        self.p_dropout = p_dropout
        self.num_stage = num_block
        self.num_jts = num_jts
        self.out_num_jts = num_jts
        self.out_dim = output_dim

        # 2d joints
        self.input_size = num_jts * input_dim
        # 3d joints
        self.output_size = out_num_jts * output_dim

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_block):
            self.linear_stages.append(LinearBlock(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = ReLU()

        self.dropout = nn.Dropout(self.p_dropout)

    def set_bn_momentum(self, momentum):
        self.batch_norm1.momentum = momentum
        for block in self.linear_stages:
            block.set_bn_momentum(momentum)

    def forward(self, x, interm_feat=None, output_feature = False):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        if self.out_dim > 3:
            y = y.reshape(batch_size, -1, 3)
        else:
            y = y.reshape(batch_size, -1, self.out_dim)

        return y

class LinearModelwithSigmoid(nn.Module):
    def __init__(self,
                 hidden_size=1024,
                 num_block=2,
                 num_jts=17,
                 p_dropout=0.5,
                 input_dim=2,
                 output_dim = 3):
        super(LinearModelwithSigmoid, self).__init__()

        self.linear_size = hidden_size
        self.p_dropout = p_dropout
        self.num_stage = num_block
        self.num_jts = num_jts
        self.out_dim = output_dim

        # 2d joints
        self.input_size = num_jts * input_dim
        # 3d joints
        self.output_size = num_jts * output_dim

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_block):
            self.linear_stages.append(LinearBlock(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = ReLU()

        self.dropout = nn.Dropout(self.p_dropout)
        self.sigmoid = nn.Sigmoid()

    def set_bn_momentum(self, momentum):
        self.batch_norm1.momentum = momentum
        for block in self.linear_stages:
            block.set_bn_momentum(momentum)

    def forward(self, x, interm_feat=None, output_feature = False):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        y = self.sigmoid(y)

        y = y.reshape(batch_size, 17, -1)

        return y

class LinearModel_xy(nn.Module):
    def __init__(self, hidsize=512, inp_dim=2, out_dim=2, num_jts=17, p_dropout=0.25):
        super(LinearModel_xy, self).__init__()
        self.hidsize = hidsize
        self.num_jts = num_jts
        self.out_dim = out_dim

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(inp_dim*num_jts, self.hidsize)
        self.batch_norm1 = nn.BatchNorm1d(self.hidsize)
        self.w2 = nn.Linear(self.hidsize, 2*num_jts)
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.activation(y)
        y = self.dropout(y)

        out = self.w2(y)
        out = out.reshape(batch_size, self.num_jts, 2)
        out = torch.cat((out, torch.ones(batch_size, self.num_jts, 1).cuda()), -1)
        return out