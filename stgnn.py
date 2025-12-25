import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *
from mtcl import graph_constructor, graph_global

class MixProp1(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(MixProp1, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0), device=x.device)
        d = adj.sum(1)
        d = torch.diag(d)
        d = torch.inverse(d)
        h = x
        out = [h]
        a = adj @ d
        for i in range(self.gdep):
            h = self.alpha * x + self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        ho = torch.sigmoid(ho)
        return ho

class Gate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gate, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W_x = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.W_h = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.zeros_(self.W_x.bias)
        nn.init.zeros_(self.W_h.bias)

    def forward(self, x, h):
        Wx = self.W_x(x)
        Wh = self.W_h(h)
        gate = torch.sigmoid(Wx + Wh)
        z = gate * Wx + (1 - gate) * Wh
        return z

class GAT(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.lin = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        batch_size, num_nodes, feature_dim, seq_len = x.size()
        assert feature_dim == self.in_features, f"Expected feature_dim {self.in_features}, got {feature_dim}"
        adj_matrices = []
        for t in range(seq_len):
            x_t = x[:, :, :, t]  # [batch_size, num_nodes, feature_dim]
            z_t = torch.tanh(self.lin(x_t))  # [batch_size, num_nodes, out_features]
            norm = torch.norm(z_t, dim=2, keepdim=True) + 1e-6
            cos_sim = torch.bmm(z_t, z_t.transpose(1, 2)) / (norm @ norm.transpose(1, 2))
            diff = z_t.unsqueeze(2) - z_t.unsqueeze(1)
            diff_norm = torch.norm(diff, dim=3)
            sum_norm = norm.squeeze(2).unsqueeze(2) + norm.squeeze(2).unsqueeze(1) + 1e-6
            feat_dev = diff_norm / sum_norm
            dynamic_term = cos_sim - feat_dev
            attention = torch.exp(dynamic_term)
            attention_sum = torch.sum(attention, dim=2, keepdim=True) + 1e-6
            attention = attention / attention_sum
            A_d_t = attention.mean(dim=0)  # [num_nodes, num_nodes]
            adj_matrices.append(A_d_t)
        adj_matrices = torch.stack(adj_matrices, dim=0)  # [seq_len, num_nodes, num_nodes]
        return adj_matrices

class stgnn(nn.Module):
    def __init__(self, gcn_true=True, buildA_true=True, gcn_depth=2, num_nodes=127, device='cuda',
                 predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=30, node_dim=40,
                 dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64,
                 end_channels=128, seq_length=5, in_dim=100, out_dim=12, layers=3, propalpha=0.05,
                 tanhalpha=3, layer_norm_affline=True):
        super(stgnn, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.device = device
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.gconv3 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
        self.Gates = nn.ModuleList([Gate(residual_channels, residual_channels) for _ in range(layers)])
        self.GAT = GAT(in_features=residual_channels, out_features=residual_channels, dropout=dropout, alpha=propalpha)
        self.seq_length = seq_length
        kernel_size = 7
        self.receptive_field = layers * (kernel_size - 1) + 1
        for i in range(1):
            rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                rf_size_j = rf_size_i + j * (kernel_size - 1)
                self.filter_convs.append(
                    nn.Conv2d(in_channels=residual_channels,
                              out_channels=conv_channels,
                              kernel_size=(1, kernel_size),
                              dilation=(1, new_dilation)))
                self.gate_convs.append(
                    nn.Conv2d(in_channels=residual_channels,
                              out_channels=conv_channels,
                              kernel_size=(1, kernel_size),
                              dilation=(1, new_dilation)))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))
                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv3.append(MixProp1(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                new_dilation *= dilation_exponential
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, idx=None):
        seq_len = input.size(3)
        input_nodes = input.size(2)
        assert seq_len == self.seq_length, f'input sequence length {seq_len} not equal to preset sequence length {self.seq_length}'
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))
        if self.gcn_true and self.buildA_true:
            if idx is None or idx.size(0) != input_nodes:
                idx = torch.arange(input_nodes).to(self.device)
            adp = self.gc(idx)
        else:
            adp = self.predefined_A
        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            x = x + residual[:, :, :, -x.size(3):]
            if self.gcn_true:
                x_s = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
                x_gat = x.permute(0, 2, 1, 3)  # [batch_size, num_nodes, conv_channels, seq_len]
                adj = self.GAT(x_gat)  # [seq_len, num_nodes, num_nodes]
                x_d_list = []
                for t, x_t_a in enumerate(adj):
                    x_d = self.gconv3[i](x, x_t_a)
                    x = self.Gates[i](x_s, x_d)
                    x_d_list.append(x)
                x = sum(x_d_list)
            else:
                x = self.residual_convs[i](x)
            x = self.norm[i](x, idx)
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        if self.training:
            return x
        else:
            return x, adp