import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys
from math import sqrt
import math

# Import the PCT model components
#from PCT.networks.cls.original_pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group

#相对位置编码
class RelativePositionEncoder(nn.Module):
    def __init__(self, feat_dim, max_dist=10.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.max_dist = max_dist
        
        # 距离编码网络
        self.dist_encoder = nn.Sequential(
            nn.Linear(1, feat_dim//4),
            nn.ReLU(),
            nn.Linear(feat_dim//4, feat_dim//2)
        )
        
        # 方向编码网络
        self.dir_encoder = nn.Sequential(
            nn.Linear(3, feat_dim//4),
            nn.ReLU(),
            nn.Linear(feat_dim//4, feat_dim//2)
        )
        
        # 合并后的投影
        self.combine = nn.Linear(feat_dim, feat_dim)

    def execute(self, vertices, joints):
        B, N, _ = vertices.shape
        M = joints.shape[1]
        
        # 计算相对位置向量 [B,N,M,3]
        rel_pos = vertices.unsqueeze(2) - joints.unsqueeze(1)
        
        # 距离特征 [B,N,M,1]->[B,N,M,d/2]
        dist = jt.norm(rel_pos, p=2, dim=-1, keepdim=True).clamp(0, self.max_dist)
        dist_feat = self.dist_encoder(dist/self.max_dist)
        
        # 方向特征 [B,N,M,d/2]
        dir_feat = self.dir_encoder(rel_pos/(dist + 1e-6))
        
        # 合并特征 [B,N,M,d]
        pos_enc = self.combine(jt.concat([dist_feat, dir_feat], dim=-1))
        return pos_enc


#修改后注意力
class PositionAwareAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 位置编码器
        self.pos_encoder = RelativePositionEncoder(feat_dim)
        
        # 特征投影
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)
        
        # 位置到注意力的投影
        self.pos_proj = nn.Linear(feat_dim, feat_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(feat_dim, feat_dim)

    def execute(self, vertices, joints):
        B, N, _ = vertices.shape
        M = joints.shape[1]
        
        # 计算相对位置编码 [B,N,M,feat_dim]
        pos_enc = self.pos_encoder(vertices[..., :3], joints[..., :3])
        
        # 投影查询/键/值 [B,N,feat_dim]
        q = self.query(vertices)  # [B,N,d]
        k = self.key(joints)      # [B,M,d]
        v = self.value(joints)    # [B,M,d]
        
        # 计算注意力分数 [B,N,M]
        attn_logits = (q @ k.transpose(-2,-1)) / sqrt(self.feat_dim)
        
        # 加入位置信息 [B,N,M,d] -> [B,N,M]
        pos_effect = (self.pos_proj(pos_enc) * q.unsqueeze(2)).sum(-1)
        attn_logits = attn_logits + pos_effect
        
        # 计算注意力权重
        attn_weights = nn.softmax(attn_logits, dim=-1)
        
        # 应用注意力 [B,N,d]
        attended = attn_weights @ v
        
        return self.out_proj(attended)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2), # 新加的
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    
    def execute(self, x):
        B, N, D = x.shape  # 获取输入形状
        assert D == self.input_dim, f"Input dimension mismatch: expected {self.input_dim}, got {D}"
        
        # 先reshape为(B*N, input_dim)进行MLP处理
        x = x.reshape(B*N, self.input_dim)
        x = self.encoder(x)  # [B*N, output_dim]
        return x.reshape(B, N, self.output_dim)  # [B, N, output_dim]

class SimpleSkinModel(nn.Module):

    def __init__(self, feat_dim: int, num_joints: int):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

        # self.pct = Point_Transformer2(output_channels=feat_dim)

        # 增强的点云特征提取
        self.pct1 = Point_Transformer2(output_channels=feat_dim)
        self.pct2 = Point_Transformer2(output_channels=feat_dim)


        # 位置感知注意力
        self.attention = PositionAwareAttention(feat_dim)
        
        # 绝对位置编码
        self.abs_pos_encoder = nn.Sequential(
            nn.Linear(3, feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2, feat_dim)
        )
        
        # 注意：拼接了 vertices(3) + shape_latent(feat_dim*2) + global_feature(feat_dim) = 3 + 3*feat_dim
        self.joint_mlp = MLP(3 + feat_dim*4, feat_dim)#没有相对位置编码时是*3
        self.vertex_mlp = MLP(3 + feat_dim*4, feat_dim)

        # 全局特征提取
        self.global_feat = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )

        # # 注意力机制的温度参数
        # self.temperature = jt.array([10.0])

        self.relu = nn.ReLU()
    
    def execute(self, vertices: jt.Var, joints: jt.Var):
        # # (B, latents)
        # shape_latent = self.relu(self.pct(vertices.permute(0, 2, 1)))

        # 多尺度特征提取
        shape_latent1 = self.relu(self.pct1(vertices.permute(0, 2, 1)))
        
        # 减少点数以获取不同尺度的表示
        B, N, _ = vertices.shape
        idx = jt.randperm(N)[:N//2]
        vertices_sub = vertices[:, idx, :]
        shape_latent2 = self.relu(self.pct2(vertices_sub.permute(0, 2, 1)))
        
        # 融合不同尺度特征
        shape_latent = jt.concat([shape_latent1, shape_latent2], dim=1)

        # # (B, N, latents)
        # vertices_latent = (
        #     self.vertex_mlp(concat([vertices, shape_latent.unsqueeze(1).repeat(1, vertices.shape[1], 1)], dim=-1))
        # )

        # 提取全局特征
        global_feature = self.global_feat(shape_latent)

        # 顶点特征 - 包含形状特征和全局特征
        vertex_pos_enc = self.abs_pos_encoder(vertices)
        vertices_latent = self.vertex_mlp(
            concat([
                vertices,
                shape_latent.unsqueeze(1).expand(-1, N, -1),
                global_feature.unsqueeze(1).expand(-1, N, -1),
                vertex_pos_enc
            ], dim=-1)
        )

        # # (B, num_joints, latents)
        # joints_latent = (
        #     self.joint_mlp(concat([joints, shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1)], dim=-1))
        # )

        # 关节特征 - 包含形状特征和全局特征
        joint_pos_enc = self.abs_pos_encoder(joints)
        joints_latent = self.joint_mlp(
            concat([joints, 
                   shape_latent.unsqueeze(1).repeat(1, self.num_joints, 1),
                   global_feature.unsqueeze(1).repeat(1, self.num_joints, 1),
                   joint_pos_enc], 
                  dim=-1)
        )

        # # (B, N, num_joints)
        # res = nn.softmax(vertices_latent @ joints_latent.permute(0, 2, 1) / sqrt(self.feat_dim), dim=-1)

        # # # 带可学习温度的注意力机制
        # # res = nn.softmax(
        # #     vertices_latent @ joints_latent.permute(0, 2, 1) / jt.abs(self.temperature), 
        # #     dim=-1
        # # )

        # assert not jt.isnan(res).any()

        # return res
        
        #引入位置后# 位置感知注意力
        vertices_attended = self.attention(vertices_latent, joints_latent)
        
        # 计算蒙皮权重
        weights = nn.softmax(
            vertices_attended @ joints_latent.transpose(-2, -1) / sqrt(self.feat_dim),
            dim=-1
        )
        
        return weights


# Factory function to create models
def create_model(model_name='pct', feat_dim=256, **kwargs):
    if model_name == "pct":
        return SimpleSkinModel(feat_dim=feat_dim, num_joints=22)
    raise NotImplementedError()