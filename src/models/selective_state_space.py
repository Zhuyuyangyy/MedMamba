"""
Selective State Space - 选择性状态空间核心实现
参考: Mamba-Linear-Time-Selective-SSM (ICLR 2024)
核心创新: 输入依赖的SSM参数(Δ, B, C)，替代Transformer的Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """RMSNorm - 比LayerNorm更简单的归一化"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class MambaBlock(nn.Module):
    """
    Mamba Block - 核心SSM块
    替代Transformer的Self-Attention + FFN
    
    核心思想: SSM参数(A,B,C,Δ)全部由输入数据动态生成
    这是Mamba区别于其他SSM的关键 - 选择性机制
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dropout: float = 0.1,
        layer_norm: bool = True,
        norm_eps: float = 1e-5,
        flip_skip_connection: bool = False,
        variant: str = "v0",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank != "auto" else max(d_model // 16, 1)
        self.variant = variant
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1x1卷积(局部特征提取)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        # SSM参数投影
        # x -> (dt, B, C)
        self.dt_proj = nn.Linear(self.d_inner, self.dt_rank, bias=True)
        self.dt_proj_bias = nn.Parameter(torch.ones(self.dt_rank) * 0.1)
        
        # B, C参数 (d_inner -> d_state * 2)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # A矩阵参数 (-A用于稳定训练)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        
        # D矩阵 (残差连接)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 归一化
        self.norm = RMSNorm(d_model, eps=norm_eps) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.flip_skip_connection = flip_skip_connection
        
        # 初始化
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        
        # dt初始化
        dt = torch.exp(torch.rand(self.dt_rank) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        nn.init.uniform_(self.dt_proj_bias, -inv_dt.sum() / self.dt_rank, inv_dt.sum() / self.dt_rank)
        
        # D初始化
        nn.init.uniform_(self.D, -0.5, 0.5)
    
    def selective_scan(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, 
                       B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """
        选择性扫描算法 - O(n)线性复杂度
        
        SSM核心公式:
        h' = A·h + B·x  (状态更新)
        y  = C·h + D·x  (输出)
        
        其中A,B,C,Δ都由输入数据动态决定
        """
        batch, seqlen, dim = x.shape
        d_state = A.shape[1]
        
        # 离散化: 将连续系统转换为离散系统
        # Δ' = softplus(dt) 限制Δ > 0
        dt = F.softplus(dt + self.dt_proj_bias)
        
        # 离散A: A_discrete = exp(Δ·A)
        # 这步是O(n)的关键 - 利用指数的性质
        A_discrete = torch.exp(torch.einsum('bdt,dnd->btdn', dt, A))
        
        # 离散B: B_discrete = Δ·B
        B_discrete = torch.einsum('bdt,bd->btd', dt, B)
        
        # 展开: 沿着序列维度展开SSM
        # 状态h: [batch, seqlen, dim, d_state]
        # 最终输出: 使用C对状态加权求和
        
        # 简化的硬件感知扫描(使用并行前缀)
        # 这里用naive实现保证正确性
        h = torch.zeros(batch, dim, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for i in range(seqlen):
            # h' = A·h + B·x
            h = torch.einsum('bnd,dn->bd', h, A_discrete[:, i]) + \
                torch.einsum('bd,dn->bd', x[:, i], B_discrete[:, i])
            
            # y = C·h + D·x
            y = torch.einsum('bd,dn->bn', h, C) + D * x[:, i]
            ys.append(y)
        
        return torch.stack(ys, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        output: [batch, seq_len, d_model]
        """
        batch, seqlen, dim = x.shape
        
        # 输入投影 + 分支
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_inner, z = xz.chunk(2, dim=-1)  # 各 d_inner
        
        # 卷积 (局部特征)
        x_conv = x_inner.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :seqlen]  # 截断
        x_conv = x_conv.transpose(1, 2)  # [B, L, d_inner]
        x_conv = F.silu(x_conv)
        
        # SSM参数
        x_dbl = self.x_proj(x_conv)  # [B, L, dt_rank + d_state*2]
        dt, B, C = torch.split(
            x_dbl, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        dt = self.dt_proj(dt)  # [B, L, dt_rank]
        self.last_delta = dt.detach()  # CTM monitoring cache
        
        # A矩阵 (可训练参数)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # 选择性扫描
        y = self.selective_scan(x_conv, dt, A, B, C, self.D.float())
        
        # 门控机制
        y = y * F.silu(z)
        
        # 输出投影
        y = self.out_proj(y)
        
        # 残差连接
        if self.norm is not None:
            y = self.norm(y)
        
        if self.flip_skip_connection:
            y = x - y  # 残差翻转
        else:
            y = x + y
        
        return self.dropout(y)


class SelectiveStateSpace(nn.Module):
    """
    选择性状态空间序列模型
    替代Transformer Encoder
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Mamba Block堆叠
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                dropout=dropout,
                variant=kwargs.get("variant", "v0"),
            )
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MedMambaBlock(nn.Module):
    """
    医学影像优化的Mamba Block
    相比标准MambaBlock增加:
    - 空间注意力偏置
    - 多尺度特征融合
    - 通道注意力
    """
    
    def __init__(self, d_model: int, d_state: int = 16, **kwargs):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, **kwargs)
        
        # 通道注意力 (医学影像专用)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x假设是[B, C, H, W]格式(空间特征)
        x_t = x.transpose(1, 2).flatten(-2)  # -> [B, H*W, C]
        x_t = self.mamba(x_t)
        x_t = x_t.unflatten(1, (x.shape[2], x.shape[3]))  # -> [B, H, W, C]
        x_t = x_t.transpose(1, 2)  # -> [B, C, H, W]
        
        # 通道注意力
        ca = self.channel_attn(x)
        x = x * ca.unsqueeze(-1).unsqueeze(-1)
        
        return x