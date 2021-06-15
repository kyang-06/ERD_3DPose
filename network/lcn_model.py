import torch.nn as nn
import torch
import torch.nn.init as init
import math
import torch.nn.functional as F

from base_modules import get_knn_mat

# ReLU = nn.ReLU
ReLU = nn.PReLU

class LCNLinear(nn.Module):
    def __init__(self, spa_in, spa_out, chn_in, chn_out, mask):
        super(LCNLinear, self).__init__()
        input_size = spa_in * chn_in
        output_size = spa_out * chn_out
        self.spa_in = spa_in
        self.spa_out = spa_out
        self.chn_in = chn_in
        self.chn_out = chn_out

        self.mask = torch.Tensor(mask).cuda().repeat(chn_out, chn_in)

        self.weight = nn.Parameter(torch.zeros(len(self.mask.nonzero())))
        self.bias = nn.Parameter(torch.zeros(spa_out * chn_out))


        self.reset_parameters()

    def reset_parameters(self):
        weight = self.weight.reshape(-1, self.chn_out, self.chn_in)
        init.kaiming_uniform_(weight, a=math.sqrt(3))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size = x.shape[0]
        a = torch.zeros_like(self.mask)
        a[self.mask > 0] = self.weight
        x = x.reshape(batch_size, -1)
        y = F.linear(x, a, self.bias)
        return y


class LCNBlock(nn.Module):
    def __init__(self, spat_size, inp_chn_size, out_chn_size, mask, stage=2, p_dropout=0.5, biased=True):
        super(LCNBlock, self).__init__()
        self.Kin = inp_chn_size
        self.Kout = out_chn_size
        self.J = spat_size
        self.stage = stage
        self.w1 = LCNLinear(self.J, self.J, self.Kin, self.Kout, mask)
        self.batch_norm1 = nn.BatchNorm1d(self.J)  # yunpeng's version
        # self.batch_norm1 = nn.BatchNorm1d(self.J*self.Kout)

        self.w2 = LCNLinear(self.J, self.J, self.Kout, self.Kout, mask)
        self.batch_norm2 = nn.BatchNorm1d(self.J)  # yunpeng's version
        # self.batch_norm2 = nn.BatchNorm1d(self.J*self.Kout)

        self.relu = ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def set_bn_momentum(self, momentum):
        self.batch_norm1.momentum = momentum
        self.batch_norm2.momentum = momentum

    def forward(self, x):
        batch_size = x.shape[0]
        y = self.w1(x)
        y = self.batch_norm1(y.reshape(batch_size, self.J, self.Kout)).reshape(batch_size, -1)  # yunpeng's version
        # y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y.reshape(batch_size, self.J, self.Kout)).reshape(batch_size, -1)  # yunpeng's version
        # y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LCNModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_size=64, num_jts=17, num_block=2, p_dropout=0.5, knn=3):
        super(LCNModel, self).__init__()
        self.K = hidden_size
        self.J = num_jts
        self.stage = num_block
        self.p_dropout = p_dropout

        self.input_dim = input_dim
        self.output_dim = output_dim

        mask = get_knn_mat(knn, num_jts=num_jts)

        self.w1 = LCNLinear(self.J, self.J, self.input_dim, self.K, mask=mask)
        self.batch_norm1 = nn.BatchNorm1d(self.J)  # yunpeng's version
        # self.batch_norm1 = nn.BatchNorm1d(self.J*self.K)
        self.relu = ReLU()
        self.dropout = nn.Dropout(p_dropout)

        self.linear_stages = []
        self.num_stage = num_block
        for i in range(num_block):
            self.linear_stages.append(
                LCNBlock(self.J, hidden_size, hidden_size, mask=mask, p_dropout=p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.w2 = LCNLinear(self.J, self.J, hidden_size, self.output_dim, mask=mask)

    def set_bn_momentum(self, momentum):
        self.batch_norm1.momentum = momentum
        for block in self.linear_stages:
            block.set_bn_momentum(momentum)

    def forward(self, x):
        batch_size = x.shape[0]
        y = self.w1(x)
        y = self.batch_norm1(y.reshape(batch_size, self.J, self.K)).reshape(batch_size, -1)
        # y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        y = y.reshape(batch_size, self.J, self.output_dim)

        return y