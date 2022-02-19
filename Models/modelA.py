"""
ResNet Architecture for computing the representations.
"""
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        #mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inp)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)

       

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
        
class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out

class DualGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out
 
 class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert(self.d_model % self.num_heads == 0)

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        '''Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        '''
        N, L, D = x.shape
        x = x.view(N, L, self.num_heads, -1)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v, mask):
        '''
        Attention(Q,K,V) = softmax(QK/sqrt(d)) * V
        The mask is multiplied with -1e9 (close to negative infinity). 
        This is done because the mask is summed with the scaled 
        matrix multiplication of Q and K and is applied immediately before a softmax. 
        The goal is to zero out these cells, and large negative inputs to softmax are near zero in the output.        
        Args:
          q: query shape (N,num_heads,Lq,depth)
          k: key shape (N,num_heads,Lk,depth)
          v: value shape (N,num_heads,Lv,depth)
          mask: mask shape (N,num_heads,Lq,Lk)
        Returns:
          out: sized [N,M,Lq,depth]
          attention_weights: sized [N,M,Lq,Lk]
        '''
        N, M, Lq, D = q.shape
        q = q.reshape(N*M, -1, D)
        k = q.reshape(N*M, -1, D)
        v = q.reshape(N*M, -1, D)

        qk = q @ k.transpose(1, 2)  # [N*M,Lq,Lk]
        scaled_qk = qk * (D ** -0.5)

        # Add mask to scaled qk.
        if mask is not None:
            mask = mask.view(N*M, Lq, -1)
            scaled_qk += (mask * -1e9)

        attention_weights = F.softmax(scaled_qk, dim=-1)  # [N*M,Lq,Lk]
        out = attention_weights @ v  # [N*M,Lq,depth]
        return out.view(N, M, Lq, D), attention_weights.view(N, M, Lq, -1)

    def forward(self, q, k, v, mask):
        N = q.size(0)

        q = self.wq(q)  # [N,Lq,D]
        k = self.wk(k)  # [N,Lk,D]
        v = self.wv(v)  # [N,Lv,D]

        # Lq = Lk = Lv = L
        q = self.split_heads(q)  # [N,num_heads,Lq,depth]
        k = self.split_heads(k)  # [N,num_heads,Lk,depth]
        v = self.split_heads(v)  # [N,num_heads,Lv,depth]

        # scaled_attention:  [N,num_heads,Lq,depth]
        # attention_weights: [N,num_heads,Lq,Lk]
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        # [N,Lq,num_heads,depth]
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        # [N,Lq,d_model]
        scaled_attention = scaled_attention.reshape(N, -1, self.d_model)

        out = self.linear(scaled_attention)
        return out, attention_weights
        
 
 class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_out, _ = self.mha(x, x, x, mask)  # [N,L,D]
        attn_out = F.dropout(attn_out, p=self.dropout)
        attn_out = self.norm1(x + attn_out)    # [N,L,D]

        ffn_out = self.ffn(attn_out)           # [N,L,D]
        ffn_out = F.dropout(ffn_out, p=self.dropout)
        out = self.norm2(attn_out + ffn_out)   # [N,L,D]
        return out


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x       
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
        
class LearnedPosEncoding(nn.Module):
    def __init__(self, num_pos, d_model, dropout=0.1):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_pos, d_model))
        self.dropout = dropout

    def forward(self, x):
        out = x + self.pos_encoding
        out = F.dropout(out, p=self.dropout)
        return out
        
        
class VGGStem(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.layers = self.make_layers()

    def make_layers(self):
        cfg = [64, 64, 'M', 128, 128, 'M', self.d_model, 'M']
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNetStem(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self.make_layer(PreActBlock, 64, 1, stride=1)
        self.layer2 = self.make_layer(PreActBlock, 128, 1, stride=2)
        self.bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, d_model, kernel_size=3,
                               stride=2, padding=1, bias=False)

    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.conv2(F.relu(self.bn(out)))
        return out
 
class ConViT(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.stem = ResNetStem(d_model)
        self.pos_encoding = LearnedPosEncoding(8*8+1, d_model, dropout)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dropout)
        self.linear = nn.Linear(d_model, 10)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x):
        N = x.size(0)
        out = self.stem(x)
        out = out.reshape(N, self.d_model, -1)  # [N,D,L]
        out = out.permute(0, 2, 1)  # [N,L,D]
        cls_tokens = self.cls_token.expand(N, -1, -1)
        out = torch.cat([cls_tokens, out], dim=1)
        out = self.pos_encoding(out)
        out = self.encoder(out, None)
        out = out[:, 0, :]
        out = self.linear(out.view(N, -1))
        return out


class modelA(nn.Module):
    def __init__(self, **param):
        super().__init__()
        self.number_features = param['number_features']
        self.embed_dim = param['embed_dim']
        self.number_labels = param['number_labels']
        self.patch_size = param['patch_size']

        self.bn1 = nn.BatchNorm2d(self.number_features)
        self.conv1 = nn.Conv2d(self.number_features, 128, kernel_size=1, stride=1, bias=False)
        self.cord1 = CoordAtt(128, 128)
        self.graph1 = DualGCN(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, bias=False)
        self.cord2 = CoordAtt(64, 64)
        self.graph2 = DualGCN(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.transformer1 = ConViT(num_layers=2, d_model=64, num_heads=4, dff=1024)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False)
        sz = 2*self.patch_size + 1 - 2*(3-1)

        self.bn4 = nn.BatchNorm1d(32*sz*sz)
        self.fc = nn.Linear(32*sz*sz, self.embed_dim)

    def forward_rep(self, x):
        out = F.dropout(F.relu(self.graph1(self.cord1(self.conv1(self.bn1(x))))),p=0.3)
        out = F.relu(self.graph2(self.cord2(self.conv2(self.bn2(out)))))
        out = F.relu(self.conv3(self.transformer1(self.bn3(out))))
        #out = F.relu(self.cord1(out))
        out = torch.reshape(out, (len(out), -1))
        out = self.fc(self.bn4(out))
        return out

    def forward(self, x):
        out = self.forward_rep(x)
        return out
