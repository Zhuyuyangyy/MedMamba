"""
HoME-MoE: 分层软专家混合 (Hierarchical Mixture of Experts)

参考: HoME - Hierarchical Experts for 3D Medical Image Segmentation (NeurIPS 2025)

核心思想:
1. 分层MoE: 在不同层级(patch/block/stage)使用不同的专家选择策略
2. 软专家混合: 不使用hard routing，而是加权组合所有专家
3. 轻量级门控: 避免巨型门控网络，分布式路由

适用场景:
- 3D医学影像分割 (CT/MRI体积数据)
- 多模态医学影像
- 病灶检测与分割

作者: MedMamba Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict, List
import math


# =============================================================================
# 基础模块
# =============================================================================

class RMSNorm(nn.Module):
    """RMSNorm - Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        if x.dim() == 4:
            norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return x * norm * self.weight.view(1, -1, 1, 1)
        else:
            norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return x * norm * self.weight


# =============================================================================
# 专家网络 (Expert Networks)
# =============================================================================

class ExpertBlock(nn.Module):
    """
    专家块 - 单个专家网络
    
    每个专家负责处理特定类型的特征模式:
    - 专家0: 边缘/轮廓特征
    - 专家1: 纹理/局部模式
    - 专家2: 语义/全局上下文
    - 专家3: 病变更/异常模式
    """
    
    def __init__(
        self,
        d_model: int,
        d_ffn: int = None,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__()
        d_ffn = d_ffn or d_model * 4
        
        # FFN结构
        self.fc1 = nn.Linear(d_model, d_ffn)
        self.act = nn.SiLU(inplace=True) if activation == "silu" else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ffn, d_model)
        
        self.norm = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D] 或 [B, C, H, W]
        """
        residual = x
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.norm(x + residual)


class ExpertRouter(nn.Module):
    """
    专家路由器 - 决定每个token/区域使用哪些专家
    
    软路由: 输出每个专家的权重 (不使用sparse routing)
    
    设计: 使用轻量级网络，避免巨大门控开销
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        routing_type: str = "soft",  # soft/hard/moe
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.routing_type = routing_type
        self.top_k = top_k
        
        # 轻量级门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(inplace=True),
            nn.Linear(d_model // 4, num_experts, bias=False),
        )
        
        # 专家偏好向量 (可学习)
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, D] 或 [B, C, H, W]
        returns: (weights, indices)
            weights: [B, num_experts] - 每个专家的权重
            indices: [B, top_k] - 选中的专家索引
        """
        # 展平以便处理
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            spatial = True
        else:
            B, L, C = x.shape
            x_flat = x
            spatial = False
        
        # 全局池化得到句子级别表示
        x_g = x_flat.mean(dim=1)  # [B, C]
        
        # 门控分数
        gate_logits = self.gate_net(x_g)  # [B, num_experts]
        gate_logits = gate_logits + self.expert_bias
        
        # Softmax得到权重
        weights = F.softmax(gate_logits, dim=-1)  # [B, num_experts]
        
        # Top-k 索引 (用于分析)
        if self.top_k < self.num_experts:
            _, indices = torch.topk(weights, k=self.top_k, dim=-1)
        else:
            indices = torch.arange(self.num_experts, device=x.device).unsqueeze(0).expand(B, -1)
        
        return weights, indices


# =============================================================================
# 分层MoE (Hierarchical MoE)
# =============================================================================

class HierarchicalMoE(nn.Module):
    """
    分层软专家混合 - HoME核心
    
    三层路由:
    1. Patch层级: 决定每个patch使用哪些专家
    2. Block层级: 聚合patch级决策
    3. Stage层级: 全局语义路由
    
    软混合: 所有专家的输出加权求和 (非稀疏)
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        num_layers: int = 1,
        routing_type: str = "soft",
        top_k: int = 2,
        expert_dropout: float = 0.1,
        use_scale: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.use_scale = use_scale
        self.use_residual = use_residual
        
        # 专家网络
        self.experts = nn.ModuleList([
            ExpertBlock(d_model, dropout=expert_dropout)
            for _ in range(num_experts)
        ])
        
        # 路由器
        self.router = ExpertRouter(
            d_model=d_model,
            num_experts=num_experts,
            routing_type=routing_type,
            top_k=top_k,
        )
        
        # 输出归一化
        self.norm = RMSNorm(d_model)
        
        # 可学习的输出缩放
        if use_scale:
            self.output_scale = nn.Parameter(torch.ones(num_experts))
        else:
            self.output_scale = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        x: [B, C, H, W] 或 [B, L, D]
        returns: (output, routing_info)
        """
        B = x.shape[0]
        
        # 获取路由权重
        weights, topk_indices = self.router(x)  # [B, num_experts], [B, top_k]
        
        # 保存路由信息
        routing_info = {
            "weights": weights,
            "topk_indices": topk_indices,
            "entropy": self._calc_entropy(weights),
        }
        
        # 展平以便处理
        if x.dim() == 4:
            original_shape = x.shape
            x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            spatial_shape = (original_shape[2], original_shape[3])
            need_reshape = True
        else:
            x_flat = x
            spatial_shape = None
            need_reshape = False
        
        # 专家输出
        expert_outputs = []
        for expert in self.experts:
            out = expert(x_flat)  # [B, L, D]
            expert_outputs.append(out)
        
        # 加权求和
        weights_expanded = weights.unsqueeze(1)  # [B, 1, num_experts]
        stacked = torch.stack(expert_outputs, dim=-1)  # [B, L, D, num_experts]
        
        # 应用输出缩放
        if self.output_scale is not None:
            scale = torch.softmax(self.output_scale, dim=0)
            scale_expanded = scale.view(1, 1, 1, self.num_experts)
            stacked = stacked * scale_expanded
        
        # 软混合
        output = torch.einsum('bldn,bn->bld', stacked, weights)  # [B, L, D]
        
        # 残差连接
        if self.use_residual:
            output = x_flat + output
        
        output = self.norm(output)
        
        # 恢复形状
        if need_reshape:
            output = output.transpose(1, 2).reshape(B, self.d_model, *spatial_shape)
        
        return output, routing_info
    
    def _calc_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """计算路由分布的熵 (衡量路由不确定性)"""
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy


# =============================================================================
# 3D医学影像专用的分层MoE
# =============================================================================

class HoMEMoE3D(nn.Module):
    """
    HoME-MoE for 3D Medical Imaging
    
    专门为3D医学影像(CT/MRI体积)设计的分层MoE:
    - 在D/H/W三个维度上进行分层处理
    - 考虑空间连贯性
    - 支持多模态融合
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        depth: int = 3,  # 3D block堆叠数
        routing_type: str = "soft",
        spatial_group: int = 4,  # 空间分组数
        modalities: int = 1,    # 输入模态数
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.depth = depth
        self.spatial_group = spatial_group
        
        # 3D卷积用于局部特征提取
        self.input_conv = nn.Conv3d(
            modalities, d_model,
            kernel_size=3, padding=1,
        )
        
        # 分层MoE块
        self.moe_blocks = nn.ModuleList([
            HierarchicalMoEBlock(
                d_model=d_model,
                num_experts=num_experts,
                routing_type=routing_type,
                spatial_group=spatial_group,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        # 输出投影
        self.output_norm = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        x: [B, C, D, H, W] - 3D体积
        returns: (output, routing_infos)
        """
        B, C, D, H, W = x.shape
        
        # 初始卷积
        x = self.input_conv(x)  # [B, d_model, D, H, W]
        
        # 通过MoE块
        routing_infos = []
        for block in self.moe_blocks:
            x, info = block(x)
            routing_infos.append(info)
        
        # 最终归一化
        x = self.output_norm(x)
        
        return x, routing_infos


class HierarchicalMoEBlock(nn.Module):
    """
    单个分层MoE块 - 处理3D特征的切片
    
    策略:
    1. 在D维度上切片，每个slice独立路由
    2. 使用空间注意力聚合相邻slice
    3. 跨slice的一致性正则化
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        routing_type: str = "soft",
        spatial_group: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_group = spatial_group
        
        # 空间注意力 (在H,W上)
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(d_model, d_model // 4, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(d_model // 4, d_model, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # 深度方向的MoE
        self.depth_moe = HierarchicalMoE(
            d_model=d_model,
            num_experts=num_experts,
            routing_type=routing_type,
            expert_dropout=dropout,
        )
        
        # 3D归一化
        self.norm = RMSNorm(d_model)
        
        # 上采样 (可选)
        self.upscale = nn.Conv3d(d_model, d_model, kernel_size=1) if spatial_group > 1 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        x: [B, C, D, H, W]
        """
        residual = x
        
        # 空间注意力
        attn_weight = self.spatial_attn(x)
        x = x * attn_weight
        
        # 深度方向处理 (展平D,H,W作为序列)
        B, C, D, H, W = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B, C, D * H * W)  # [B, C, D*H*W]
        x_flat = x_flat.permute(0, 2, 1)  # [B, D*H*W, C]
        
        # MoE处理
        x_out, routing_info = self.depth_moe(x_flat)
        
        # 恢复形状
        x_out = x_out.permute(0, 2, 1).reshape(B, C, D, H, W)
        
        # 残差
        x_out = residual + x_out
        
        return self.norm(x_out), routing_info


# =============================================================================
# 2D医学影像专用的分层MoE (轻量版)
# =============================================================================

class HoMEMoE2D(nn.Module):
    """
    HoME-MoE for 2D Medical Imaging
    
    轻量版本，适用于2D医学影像(病理图像、X光等)
    同样使用分层软专家混合
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        depth: int = 3,
        routing_type: str = "soft",
        top_k: int = 2,
        dropout: float = 0.1,
        use_ctm: bool = False,  # 是否集成CTM分析
    ):
        super().__init__()
        self.d_model = d_model
        self.use_ctm = use_ctm
        
        # 分层MoE块
        self.moe_blocks = nn.ModuleList([
            HierarchicalMoE(
                d_model=d_model,
                num_experts=num_experts,
                num_layers=1,
                routing_type=routing_type,
                top_k=top_k,
                expert_dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        # 可选: CTM分析器用于幻觉检测
        if use_ctm:
            self.ctm_analyzer = CTMTrajectoryAnalyzer(d_model)
        else:
            self.ctm_analyzer = None
        
        # 输出归一化
        self.output_norm = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        x: [B, C, H, W]
        returns: (output, routing_infos)
        """
        B, C, H, W = x.shape
        
        routing_infos = []
        
        for block in self.moe_blocks:
            x, info = block(x)
            routing_infos.append(info)
        
        x = self.output_norm(x)
        
        # CTM分析
        ctm_scores = None
        if self.ctm_analyzer is not None:
            x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            x_seq = x_flat.unsqueeze(1)  # 作为时间序列
            ctm_scores = self.ctm_analyzer.hallucination_score(x_seq)
        
        return x, routing_infos


# =============================================================================
# CTM轨迹分析器 (用于幻觉检测)
# =============================================================================

class CTMTrajectoryAnalyzer(nn.Module):
    """
    CTM动力学轨迹分析 - 监控MoE专家的决策稳定性
    
    用于检测:
    - 专家选择是否稳定
    - 路由分布是否振荡
    - 是否存在冲突的专家共识
    """
    
    def __init__(self, d_model: int, num_ticks: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_ticks = num_ticks
        
        # 轨迹→指标投影
        self.stability_proj = nn.Linear(d_model, 1)
        self.oscillation_proj = nn.Linear(d_model, 1)
        self.conflict_proj = nn.Linear(d_model, 1)
    
    def forward(self, trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        trajectory: [B, T, D] - MoE状态的序列 (T=层序号)
        """
        T = trajectory.shape[1]
        # Need at least 2 ticks to compute diff-based oscillation.
        k = max(2, min(self.num_ticks, T))
        last_k = trajectory[:, -k:, :]  # [B, k, D]

        # 稳定性: 最后k步范数均值的倒数
        last_norms = torch.norm(last_k, dim=-1)
        stability = 1.0 / (1.0 + last_norms.mean(dim=-1))

        # 振荡: 相邻时间步差的方差 (k >= 2 guaranteed above)
        deltas = torch.norm(torch.diff(last_k, dim=1), dim=-1)
        oscillation = deltas.std(dim=-1)
        
        # 吸引子margin: top1 - top2
        final = trajectory[:, -1, :]
        sorted_final, _ = final.sort(dim=-1, descending=True)
        attractor_margin = (sorted_final[:, 0] - sorted_final[:, 1]).abs()
        
        return {
            "stability": stability,
            "oscillation": oscillation,
            "attractor_margin": attractor_margin,
        }
    
    def hallucination_score(self, trajectory: torch.Tensor) -> torch.Tensor:
        """综合幻觉风险分数 0=可信, 1=高风险"""
        m = self.forward(trajectory)
        return (
            (1 - m["stability"]) * 0.4 +
            m["oscillation"] * 0.3 +
            (1 - torch.sigmoid(m["attractor_margin"])) * 0.3
        ).clamp(0, 1)


# =============================================================================
# 专家多样性损失 (用于训练)
# =============================================================================

class ExpertDiversityLoss(nn.Module):
    """
    专家多样性损失 - 鼓励不同的专家被使用
    
    公式: L_div = -sum_i(p_i * log(p_i)) / log(num_experts)
    最大化熵 = 最大化多样性
    """
    
    def __init__(self, target_entropy: float = 0.9):
        super().__init__()
        self.target_entropy = target_entropy
    
    def forward(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        routing_weights: [B, num_experts]
        """
        num_experts = routing_weights.shape[-1]
        
        # 计算熵
        entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1)
        max_entropy = math.log(num_experts)
        
        # 归一化熵
        normalized_entropy = entropy / max_entropy
        
        # 损失: 偏离目标熵
        loss = F.mse_loss(normalized_entropy, torch.ones_like(normalized_entropy) * self.target_entropy)
        
        return loss


# =============================================================================
# MoE集成工具
# =============================================================================

class MoEEnsemble(nn.Module):
    """
    MoE集成 - 组合多个MoE专家的输出
    
    用途:
    - 医学影像多任务学习
    - 集成不同专家的预测
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        ensemble_method: str = "weighted",  # weighted/attention
    ):
        super().__init__()
        self.ensemble_method = ensemble_method
        
        # 专家输出变换
        self.expert_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_experts)
        ])
        
        # 集成权重
        if ensemble_method == "weighted":
            self.ensemble_weights = nn.Parameter(torch.ones(num_experts) / num_experts)
        elif ensemble_method == "attention":
            self.attention = nn.MultiheadAttention(
                d_model, num_heads=4, batch_first=True
            )
    
    def forward(
        self,
        expert_outputs: List[torch.Tensor],
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        expert_outputs: list of [B, L, D]
        routing_weights: [B, num_experts] 可选
        """
        # 变换每个专家输出
        transformed = [et(e) for et, e in zip(self.expert_transforms, expert_outputs)]
        
        # 加权集成
        if self.ensemble_method == "weighted":
            weights = torch.softmax(self.ensemble_weights, dim=0)
            output = sum(w * t for w, t in zip(weights, transformed))
        else:
            # Attention集成
            stacked = torch.stack(transformed, dim=1)  # [B, num_experts, L, D]
            B, n_exp, L, D = stacked.shape
            stacked_flat = stacked.reshape(B * n_exp, L, D)

            # Self-attention
            output, _ = self.attention(stacked_flat, stacked_flat, stacked_flat)
            output = output.reshape(B, n_exp, L, D).mean(dim=1)  # [B, L, D]
        
        return output


__all__ = [
    'ExpertBlock',
    'ExpertRouter',
    'HierarchicalMoE',
    'HoMEMoE3D',
    'HoMEMoE2D',
    'HierarchicalMoEBlock',
    'CTMTrajectoryAnalyzer',
    'ExpertDiversityLoss',
    'MoEEnsemble',
    'RMSNorm',
]