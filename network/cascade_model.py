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
        # if input_dim == 4:
        #     self.trans_net = nn.Linear(2 * num_jts, 2 * num_jts)

    def forward(self, x):
        batch_size = x.shape[0]
        # if self.input_dim == 4:
        #     pose2d = x[:, :, :2]
        #     delta = x[:, :, 2:]
        #     pose2d_feat = self.trans_net(pose2d.reshape(batch_size, -1)).reshape(batch_size, self.num_jts, -1)
        #     x = torch.cat((pose2d_feat, delta), -1)
        #     if self.training:
        #         # store for computing loss
        #         self.input_feat = x
        x = x.reshape(batch_size, -1)
        y = self.w1(x)
        for i in range(self.blocks):
            res = y[:]
            y = self.increment[i](y)
            y = y + res
        y = self.w2(y)
        # y = y.reshape(batch_size, self.num_jts, self.output_dim)
        y = y.reshape(batch_size, -1, 3)
        return y
    # def loss(self, predict, target, criterion=nn.MSELoss()):
    #     # only for concat input(pose2d + delta)
    #     batch_size = predict.shape[0]
    #     loss_sup = criterion(predict, target)
    #     pose2d_feat = self.input_feat[:, :, :2].reshape(batch_size, -1)
    #     delta_feat = self.input_feat[:, :, 2:].reshape(batch_size, -1)
    #     pose2d_stat = torch.cat((pose2d_feat.mean(-1, keepdim=True), pose2d_feat.std(-1, keepdim=True)), -1)
    #     delta_stat = torch.cat((delta_feat.mean(-1, keepdim=True), delta_feat.std(-1, keepdim=True)), -1)
    #     loss_trans = criterion(pose2d_stat, delta_stat)
    #     loss = loss_sup + loss_trans
    #     return loss


class CascadeNet(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_size=1024, num_block=1, num_jts=17, cas_num=2, p_dropout=0.5):
        super(CascadeNet, self).__init__()
        self.model = []
        self.cas_num = cas_num
        for i in range(cas_num):
            self.model.append(RegressDeltaModule(input_dim=input_dim, output_dim=output_dim, num_jts=num_jts,
                                                 hidden_size=hidden_size, blocks=num_block, p_dropout=p_dropout))
        self.model = nn.ModuleList(self.model)
    def forward(self, x):
        return self.model(x)


class RegressPelvisModel(nn.Module):
    def __init__(self, num_jts=17, input_dim=2, output_dim=3, num_blocks=1, hidden_size=1024, p_dropout=0.5):
        super(RegressPelvisModel, self).__init__()
        self.num_jts = num_jts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        self.w1 = nn.Sequential(
            nn.Linear(self.num_jts * self.input_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout)
        )
        self.blocks = nn.ModuleList([LinearBlock(self.hidden_size, p_dropout=p_dropout) for _ in range(num_blocks)])
        self.w2 = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        y = self.w1(x)
        for i in range(self.num_blocks):
            y = self.blocks[i](y)
        y = self.w2(y)
        y = y.reshape(batch_size, 1, self.output_dim)
        return y

class NoiseModel(nn.Module):
    def __init__(self, hidsize=512, inp_dim=2, out_dim=2, num_jts=17, p_dropout=0.25):
        super(NoiseModel, self).__init__()
        self.hidsize = hidsize
        self.num_jts = num_jts
        self.out_dim = out_dim

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(inp_dim*num_jts, self.hidsize)
        self.batch_norm1 = nn.BatchNorm1d(self.hidsize)
        self.w2 = nn.Linear(self.hidsize, 2*num_jts)
        # nn.init.constant_(self.w1.weight, 0)
        # nn.init.constant_(self.w1.bias, 0)
        # nn.init.constant_(self.w2.weight, 0)
        # nn.init.constant_(self.w2.bias, 0)
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

class Cascade_LiftingNetwork(nn.Module):
    def __init__(self, model_func, cas_num, **args):
        super(Cascade_LiftingNetwork, self).__init__()
        self.model = nn.ModuleList(model_func(**args) for i in range(cas_num))
    def forward(self, x):
        return self.model(x)
