# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from .head import DynamicConv

__all__ = ["MyFastRCNNConvFCHead", "build_box_head", "ROI_BOX_HEAD_REGISTRY"]

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias = False),
                nn.ReLU(),
                nn.Linear(channels // reduction, channels, bias = False),
                nn.Sigmoid())
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = y.expand_as(x)
        
        return y

class SpatialAttention(nn.Module):
    def __init__(self, out_length, mul = 4):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(out_length, out_length * mul, bias = False),
                nn.ReLU(),
                nn.Linear(out_length * mul, out_length, bias = False),
                nn.Sigmoid())

    def forward(self, x):
        shape = x.shape
        x = torch.sum(x, dim = 1).squeeze()
        x = x.view(shape[0], -1)
        out = self.fc(x)
        out = out.view(shape[0], 1, shape[2], shape[3])
        out = out.repeat(1, shape[1], 1, 1)
        
        return out

class SeqLayers(nn.Sequential):
    def __init__(self, input_shape: ShapeSpec, conv_dims: List[int], fc_dims: List[int], conv_norm=""):
        super().__init__()
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                    self._output_size[0],
                    conv_dim,
                    kernel_size=3,
                    padding=1,
                    bias=not conv_norm,
                    norm=get_norm(conv_norm, conv_dim),
                    activation=nn.ReLU(),
                    )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        
        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim
        
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self:
            x = layer(x)

        return x

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


@ROI_BOX_HEAD_REGISTRY.register()
class MyFastRCNNConvFCHead(nn.Module):

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm=""):
        """
        NOTE: this interface is experimental.
        
        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        self.layers = SeqLayers(input_shape, conv_dims, fc_dims, conv_norm)
        self.output_shape = self.layers.output_shape

        #self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        #self.conv_norm_relus = []
        #for k, conv_dim in enumerate(conv_dims):
        #    conv = Conv2d(
        #            self._output_size[0],
        #            conv_dim,
        #            kernel_size=3,
        #            padding=1,
        #            bias=not conv_norm,
        #            norm=get_norm(conv_norm, conv_dim),
        #            activation=nn.ReLU(),
        #            )
        #    self.add_module("conv{}".format(k + 1), conv)
        #    self.conv_norm_relus.append(conv)
        #    self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        #
        #self.fcs = []
        #for k, fc_dim in enumerate(fc_dims):
        #    if k == 0:
        #        self.add_module("flatten", nn.Flatten())
        #    fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
        #    self.add_module("fc{}".format(k + 1), fc)
        #    self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
        #    self.fcs.append(fc)
        #    self._output_size = fc_dim
        #
        #for layer in self.conv_norm_relus:
        #    weight_init.c2_msra_fill(layer)
        #for layer in self.fcs:
        #    weight_init.c2_xavier_fill(layer)

        #nhead = 8
        #dropout = 0.1
        #d_model = 1024
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        ##self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.inst_interact = DynamicConv()
        #self.dropout1 = nn.Dropout(dropout)
        #self.norm1 = nn.LayerNorm(d_model)
        #self.dropout2 = nn.Dropout(dropout)
        #self.norm2 = nn.LayerNorm(d_model)
        #self.dropout3 = nn.Dropout(dropout)
        #
        #dim_feedforward = 256
        #self.linear1 = nn.Linear(d_model, dim_feedforward)
        #self.dropout = nn.Dropout(dropout)
        #self.linear2 = nn.Linear(dim_feedforward, d_model)
        #self.activation = nn.ReLU()
        #self.norm3 = nn.LayerNorm(d_model)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {"input_shape": input_shape,
                "conv_dims": [conv_dim] * num_conv,
                "fc_dims": [fc_dim] * num_fc,
                "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,}

    def forward(self, x):
    #def forward(self, x, proposal_features = None):
        #proposal_features shape: 1 * 1024 * 1024
        #x.shape before layers: 1024, 256, 7, 7
        #x shape after layers: 1024* 1024
        roi_features = x
        for layer in self.layers:
            x = layer(x)

        return x
        #shape = proposal_features.shape
        #proposal_features = proposal_features.view(-1, 512, 1024).permute(1, 0, 2)
        #proposal_features2 = self.self_attn(proposal_features, proposal_features, value=proposal_features)[0]
        #proposal_features = proposal_features + self.dropout1(proposal_features2)
        #pro_features = self.norm1(proposal_features)

        ##x_reshaped = x.view(-1, 512, 1024).permute(1, 0, 2)
        ##proposal_features2 = self.cross_attn(x_reshaped, proposal_features, value=proposal_features)[0]
        ##proposal_features2 = self.cross_attn(key = x_reshaped, query = proposal_features, value=x_reshaped)[0]
        ##proposal_features = proposal_features + self.dropout2(proposal_features2)
        ##obj_features = self.norm2(proposal_features)

        #pro_features = pro_features.view(512, -1, 1024).permute(1, 0, 2).reshape(1, -1, 1024)
        #roi_features = roi_features.view(roi_features.shape[0], 1024, -1).permute(2, 0, 1)
        ##pro_features = pro_features.view(512, -1, 1024).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        #pro_features2 = self.inst_interact(pro_features, roi_features)
        #pro_features = pro_features + self.dropout2(pro_features2)
        #obj_features = self.norm2(pro_features)
        #
        #obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        #obj_features = obj_features + self.dropout3(obj_features2)
        #obj_features = self.norm3(obj_features)

        #return x, obj_features.view(shape[0], shape[1], shape[2])
    
    #@property
    #@torch.jit.unused
    #def output_shape(self):
    #    """
    #    Returns:
    #        ShapeSpec: the output feature shape
    #    """
    #    o = self._output_size
    #    if isinstance(o, int):
    #        return ShapeSpec(channels=o)
    #    else:
    #        return ShapeSpec(channels=o[0], height=o[1], width=o[2])

def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
