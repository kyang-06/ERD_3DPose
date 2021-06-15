import torch
import torch.nn as nn
from network.linear_model import LinearBlock, LinearModel

class RegressDeltaModule(nn.Module):
    def __init__(self, num_jts=17, input_dim=2, hidden_size=1024, output_dim=3, blocks=1, p_dropout=0.5):
        super(RegressDeltaModule, self).__init__()
        self.num_jts = num_jts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_dim * num_jts
        self.output_size = output_dim * num_jts
        self.w1 = nn.Sequential(nn.Linear(self.input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True), nn.Dropout(p_dropout))

        self.increment = nn.ModuleList()
        self.blocks = blocks
        self.hidden_size = hidden_size
        for i in range(self.blocks):
            self.increment.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(p_dropout)))
        self.w2 = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        y = self.w1(x)
        for i in range(self.blocks):
            res = y[:]
            y = self.increment[i](y)
            y = y + res
        y = self.w2(y)
        y = y.reshape(batch_size, -1, 3)
        return y


class ResidualRegressor(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_size=1024, num_block=1, num_jts=17, inc_num=2, p_dropout=0.5):
        super(ResidualRegressor, self).__init__()
        self.model = []
        self.cas_num = inc_num
        for i in range(inc_num):
            self.model.append(RegressDeltaModule(input_dim=input_dim, output_dim=output_dim, num_jts=num_jts,
                                                 hidden_size=hidden_size, blocks=num_block, p_dropout=p_dropout))
        self.model = nn.ModuleList(self.model)
    def forward(self, x):
        return self.model(x)



