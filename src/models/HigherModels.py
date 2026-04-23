import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)

class DropAttention(nn.Module):
    """Implement DropAttention as described in the paper with post-dropout normalization"""
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        self.eps = 1e-7


    def forward(self, att_weights):
        if not self.training or self.p == 0:
            return att_weights
        
        # 1. Generate binary mask (0s indicate dropped positions)
        mask = torch.rand(att_weights.size(0), 1, att_weights.size(2),
                        device = att_weights.device) > self.p
        
        masked_att = att_weights * mask.float()
        
        normalized_att = masked_att / (masked_att.sum(dim=2, keepdim = True) + self.eps)

        return normalized_att

        



class Attention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation, dropatt_rate = 0.2):
        super(Attention, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

        # Add DropAttention
        self.dropatt = DropAttention(dropatt_rate) if dropatt_rate > 0 else nn.Identity()


    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return torch.sigmoid(x)

        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        norm_att = self.dropatt(norm_att)
        
        x = torch.sum(norm_att * cla, dim=2)

        return x, norm_att

class MeanPooling(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(MeanPooling, self).__init__()

        self.cla_activation = cla_activation

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.cla)

    def activate(self, x, activation):
        return torch.sigmoid(x)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        x = torch.mean(cla, dim=2)

        return x, []

class MHeadAttention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation, head_num = 4, dropatt_rate = 0.2):
        super(MHeadAttention, self).__init__()

        self.head_num = head_num
        #self.dropout_rate = dropout_rate
        self.dropatt_rate  = dropatt_rate

        # Predefine activation functions for efficiency
        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.ModuleList([])
        self.cla = nn.ModuleList([])
        for i in range(self.head_num):
            self.att.append(nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))
            self.cla.append(nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))

        self.head_weight = nn.Parameter(torch.tensor([1.0/self.head_num] * self.head_num))
        # Add dropout layer for attention weights
        #self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropatt = DropAttention(dropatt_rate) if dropatt_rate > 0 else nn.Identity()


    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        """
        input: (samples_num, freq_bins, time_steps, 1)
        """
        x_out = []
        for i in range(self.head_num):
            att = self.att[i](x)
            att = self.activate(att, self.att_activation)

            cla = self.cla[i](x)
            cla = self.activate(cla, self.cla_activation)

            att = att[:, :, :, 0]  # (samples_num, classes_num, time_steps)
            cla = cla[:, :, :, 0]  # (samples_num, classes_num, time_steps)

            epsilon = 1e-7
            att = torch.clamp(att, epsilon, 1. - epsilon)

            # Initial normalization (pre - dropout)
            #norm_att = att / torch.sum(att, dim=2)[:, :, None]
            norm_att = att/ (torch.sum(att, dim = 2, keepdim=True) + 1e-7)
            
            # Apply DropAttention (dropout + renormalization)
            #norm_att = self.dropout(norm_att)
            norm_att = self.dropatt(norm_att)

            # Weigthed summation
            weighted = torch.sum(norm_att * cla, dim = 2)
            x_out.append(weighted * self.head_weight[i])

        x = (torch.stack(x_out, dim=0)).sum(dim=0)

        return x, []