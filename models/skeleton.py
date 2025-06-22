import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group

class MultiLayerAttention(nn.Module):
    def __init__(self, feat_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            SA_Layer(feat_dim) for _ in range(num_layers)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * num_layers, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def execute(self, x: jt.Var, pos: jt.Var):
        outputs = []
        for layer in self.layers:
            att_out = layer(x, pos)
            x = x + att_out
            xx = x.mean(dim=2)
            outputs.append(xx)
        
        multi_scale_features = jt.concat(outputs, dim=-1)
        return self.fusion(multi_scale_features)

class SimpleSkeletonModel(nn.Module):
    
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = Point_Transformer2(output_channels=feat_dim)
        self.multi_attention = MultiLayerAttention(feat_dim, num_layers=3)
        self.skip_connection = nn.Linear(feat_dim, feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        N = vertices.shape[2]
        att_x = self.multi_attention(x.unsqueeze(-1).repeat(1, 1, N), vertices)
        x = x + self.skip_connection(x)
        return self.mlp(x + att_x)

# Factory function to create models
def create_model(model_name='pct', output_channels=66, **kwargs):
    if model_name == "pct":
        return SimpleSkeletonModel(feat_dim=256, output_channels=output_channels)
    raise NotImplementedError()
