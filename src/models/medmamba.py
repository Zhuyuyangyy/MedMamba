"""
MedMamba V2 - 医学影像分类/分割统一模型
融合最新研究: U-Mamba + VMamba + TransMamba + CTM动力学分析 + HoME-MoE

V2改进:
1. Cross-Scan: 四方向扫描使SSM适应2D空间 (VMamba)
2. CNN-SSM双分支: CNN局部 + SSM全局 (U-Mamba启发)
3. 特征融合: CNN特征与SSM隐藏态交叉融合 (TransMamba)
4. CTM动力学幻觉检测: 实时监控SSM轨迹稳定性
5. HoME-MoE: 分层软专家混合 (NeurIPS 2025)
6. 可切换分类/分割头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict, Any, List
import math
from functools import partial

# 导入新模块
from .vmamba_blocks import (
    CrossScan, CrossMerge, SS2D, VSSBlock2D,
    CrossAttentionFusion, VMamba2D, SimplifiedCrossScan,
    SimplifiedCrossMerge, channel_shuffle,
)
from .home_moe import (
    ExpertBlock, ExpertRouter, HierarchicalMoE,
    HoMEMoE3D, HoMEMoE2D, CTMTrajectoryAnalyzer,
    ExpertDiversityLoss, MoEEnsemble,
)


# =============================================================================
# 归一化
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


# =============================================================================
# Cross-Scan (VMamba核心) - 四方向扫描让1D SSM适应2D图像
# =============================================================================

class CrossScan(nn.Module):
    """
    四方向扫描 - 将2D图像展分为4条1D序列
    每个像素从四个方向汇聚信息，实现全局感受野
    """
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: [B, C, H, W]
        returns: 4条扫描序列, 各 [B, C, H*W]
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)  #  raster: 左上→右下
        
        return [
            x_flat,  # 方向1: 左上→右下
            torch.flip(x_flat, dims=[-1]),  # 方向2: 右下→左上
            x.transpose(2, 3).reshape(B, C, H * W),  # 方向3: 右上→左下
            torch.flip(x.transpose(2, 3).reshape(B, C, H * W), dims=[-1]),  # 方向4: 左下→右上
        ]


class CrossMerge(nn.Module):
    """四方向扫描结果合并"""
    
    def forward(self, scans: List[torch.Tensor], H: int, W: int) -> torch.Tensor:
        """scans: 4×[B, C, H*W] → [B, C, H, W]"""
        B, C, L = scans[0].shape
        
        # 反转各方向
        out = [
            scans[0].view(B, C, H, W),
            torch.flip(scans[1], dims=[-1]).view(B, C, H, W),
            scans[2].view(B, C, W, H).transpose(2, 3),
            torch.flip(scans[3], dims=[-1]).view(B, C, W, H).transpose(2, 3),
        ]
        
        return sum(out)


# =============================================================================
# SSM核心 - 选择性状态空间 (Mamba)
# =============================================================================

class MambaBlock2D(nn.Module):
    """
    2D Mamba Block - 用于四方向选择性扫描
    替代2D Self-Attention, O(n)复杂度
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank != "auto" else max(d_model // 16, 1)

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 深度卷积
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        
        # SSM参数
        self.dt_proj = nn.Linear(self.d_inner, self.dt_rank, bias=True)
        self.dt_proj_bias = nn.Parameter(torch.ones(self.dt_rank) * 0.1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # A矩阵
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        
        # D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_parameters()

    def _init_parameters(self):
        for m in [self.in_proj, self.out_proj, self.dt_proj, self.x_proj]:
            if m is not None:
                nn.init.xavier_uniform_(m.weight)
        
        dt = torch.exp(torch.rand(self.dt_rank) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        nn.init.uniform_(self.dt_proj_bias, -inv_dt.sum() / self.dt_rank, inv_dt.sum() / self.dt_rank)
        nn.init.uniform_(self.D, -0.5, 0.5)

    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """选择性扫描 - O(n)"""
        B, L, D = x.shape
        d_state = self.d_state

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # 卷积
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM参数
        x_dbl = self.x_proj(x_conv)
        dt, B_param, C = torch.split(x_dbl, [self.dt_rank, d_state, d_state], dim=-1)
        dt = self.dt_proj(dt)

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_proj_bias)

        # 展开扫描 (生产环境用cuDSSM)
        A_d = torch.exp(torch.einsum('bdt,dnd->btdn', dt, A))
        B_d = torch.einsum('bdt,bd->btd', dt, B_param)

        h = torch.zeros(B, D, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = torch.einsum('bnd,dn->bd', h, A_d[:, i]) + torch.einsum('bd,dn->bd', x_conv[:, i], B_d[:, i])
            y = torch.einsum('bd,dn->bn', h, C) + self.D.float() * x_conv[:, i]
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]  (已展平的序列)
        """
        y = self.selective_scan(x)
        y = self.norm(y)
        return x + self.dropout(y)


# =============================================================================
# VMamba风格的2D SSM层
# =============================================================================

class VSSBlock2D(nn.Module):
    """
    视觉状态空间块 - 四方向SSM处理2D图像
    完整流程: CrossScan → 4×SSM → CrossMerge
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_scan = CrossScan()
        self.cross_merge = CrossMerge()
        
        # 4个方向各一个Mamba Block (参数共享)
        self.ssm = MambaBlock2D(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        # 方向权重 (可学习)
        self.direction_weight = nn.Parameter(torch.ones(4) / 4)
        
        self.H = None
        self.W = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        self.H, self.W = x.shape[2], x.shape[3]
        
        # 四方向扫描
        scans = self.cross_scan(x)  # 4 × [B, C, L]
        
        # 并行SSM处理
        processed = [self.ssm(s) for s in scans]
        
        # 加权融合
        weights = torch.softmax(self.direction_weight, dim=0)
        fused = sum(w * p for w, p in zip(weights, processed))
        
        # 合并回2D
        out = self.cross_merge([fused], self.H, self.W)
        
        return out


# =============================================================================
# CNN分支 (U-Mamba启发) - 局部特征
# =============================================================================

class CNNBranch(nn.Module):
    """
    CNN分支 - 提取局部空间特征
    对应U-Mamba的CNN层: ResBlock + 逐点卷积
    """
    
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        
        # 残差块: 深度可分离卷积 + 归一化 + 激活
        self.depthwise = nn.Conv2d(
            d_model, d_model,
            kernel_size=3, padding=1,
            groups=d_model,  # 深度可分离
        )
        self.bn1 = nn.BatchNorm2d(d_model)
        self.act1 = nn.SiLU(inplace=True)
        
        # 逐点卷积
        self.pointwise = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(d_model)
        self.act2 = nn.SiLU(inplace=True)
        
        # 下采样 (可选)
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(d_model * 2),
            nn.SiLU(inplace=True),
        )
        
        self.upscale = nn.Conv2d(d_model * 2, d_model, kernel_size=1) if d_state > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, downsample: bool = False) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        residual = x
        
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        if downsample:
            out = self.downsample(out)
            residual = F.avg_pool2d(residual, kernel_size=2, stride=2)
            residual = self.upscale(residual) if isinstance(self.upscale, nn.Conv2d) else residual
        
        return out + residual


# =============================================================================
# CTM轨迹分析器 (从CTM-Guard移植)
# =============================================================================

class CTMTrajectoryAnalyzer(nn.Module):
    """
    CTM动力学轨迹分析 - 监控SSM隐藏态稳定性
    用稳定性/振荡/冲突作为幻觉风险信号
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
        trajectory: [B, T, D] - SSM状态序列 (T=时间步/层序号)
        """
        T = trajectory.shape[1]
        k = min(self.num_ticks, T)
        last_k = trajectory[:, -k:, :]  # [B, k, D]
        
        # 稳定性: 最后k步范数均值的倒数
        last_norms = torch.norm(last_k, dim=-1)
        stability = 1.0 / (1.0 + last_norms.mean(dim=-1))
        
        # 振荡: 相邻时间步差的方差
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
# 特征融合模块 (TransMamba启发)
# =============================================================================

class FeatureFusion(nn.Module):
    """
    CNN-SSM特征融合
    将CNN的局部特征与SSM的全局状态融合
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # CNN特征变换
        self.cnn_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        
        # SSM特征变换
        self.ssm_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1),
        )
        
        # 融合权重 (可学习)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, cnn_feat: torch.Tensor, ssm_feat: torch.Tensor) -> torch.Tensor:
        """
        cnn_feat: [B, C, H, W] - CNN分支输出
        ssm_feat: [B, C, H, W] - SSM分支输出
        """
        # 转换维度用于融合
        B, C, H, W = cnn_feat.shape
        
        # 各自投影到相同空间后加权融合
        c = self.cnn_proj(cnn_feat.flatten(2).transpose(1, 2))  # [B, H*W, C]
        s = self.ssm_proj(ssm_feat).flatten(2).transpose(1, 2)   # [B, H*W, C]
        
        w = torch.sigmoid(self.fusion_weight)
        fused = w * c + (1 - w) * s
        
        return fused.transpose(1, 2).view(B, C, H, W)


# =============================================================================
# 主编码器: CNN + SSM双分支
# =============================================================================

class DualBranchEncoder(nn.Module):
    """
    双分支编码器: CNN局部 + SSM全局
    融合U-Mamba(CNN+SSM混合) + VMamba(四方向扫描)
    """
    
    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_ctm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # 投影头: 将输入映射到d_model
        self.input_proj = nn.Conv2d(3, d_model, kernel_size=1)
        
        # SSM分支 (四方向扫描)
        self.ssm_branch = nn.ModuleList([
            VSSBlock2D(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
        # CNN分支 (局部特征)
        self.cnn_branch = nn.ModuleList([
            CNNBranch(d_model, d_state)
            for _ in range(n_layers)
        ])
        
        # 特征融合
        self.fusion = FeatureFusion(d_model)
        
        # CTM分析器 (每3层插一个)
        self.ctm_enabled = use_ctm
        if use_ctm:
            self.ctm_analyzers = nn.ModuleList([
                CTMTrajectoryAnalyzer(d_model, num_ticks=8)
                if i % 3 == 0 else None
                for i in range(n_layers)
            ])
        
        # 最终归一化
        self.final_norm = RMSNorm(d_model)
    
    def forward(self, x: torch.Tensor, return_ctm: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        x: [B, 3, H, W]
        return: (encoded, ctm_metrics)
        """
        B, C, H, W = x.shape
        
        # 投影到d_model
        x = self.input_proj(x)  # [B, d_model, H, W]
        
        # 收集SSM隐藏态用于CTM
        ssm_trajectories = [] if return_ctm and self.ctm_enabled else None
        
        for layer_idx in range(self.n_layers):
            # SSM分支
            ssm_out = self.ssm_branch[layer_idx](x)
            
            # CNN分支 (每2层执行一次以节省计算)
            if layer_idx % 2 == 0:
                cnn_out = self.cnn_branch[layer_idx](x)
                x = self.fusion(cnn_out, ssm_out)
            else:
                x = ssm_out
            
            # CTM分析
            if return_ctm and self.ctm_enabled and self.ctm_analyzers[layer_idx] is not None:
                # 用SSM输出作为轨迹点 (展平空间维)
                traj = ssm_out.flatten(2).transpose(1, 2).unsqueeze(1)  # [B, 1, H*W, C]
                ssm_trajectories.append(traj)
        
        x = self.final_norm(x)
        
        ctm_metrics = None
        if return_ctm and ssm_trajectories:
            all_trajs = torch.cat(ssm_trajectories, dim=1)  # [B, n_ctm_layers, H*W, C]
            all_trajs = all_trajs.transpose(2, 1)  # [B, H*W, n_ctm_layers, C]
            flat_trajs = all_trajs.reshape(B, -1, self.d_model)  # [B, n_patches*n_ctm, C]
            ctm_metrics = self.ctm_analyzers[0].hallucination_score(flat_trajs.unsqueeze(1))
        
        return x, ctm_metrics


# =============================================================================
# 解码头: 分类 + 分割
# =============================================================================

class MedMambaHead(nn.Module):
    """
    医学影像头: 分类 + 分割
    支持多任务学习
    """
    
    def __init__(self, d_model: int, num_classes: int = 2, num_diseases: int = 8):
        super().__init__()
        
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes),
        )
        
        # 分割头 (轻量U-Net风格)
        self.seg_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(d_model // 2, num_diseases, kernel_size=1),
        )
        
        # 上采样
        self.upscale = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    
    def forward(
        self, x: torch.Tensor, task: str = "classification"
    ) -> Dict[str, torch.Tensor]:
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 全局池化用于分类
        cls_feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        cls_logits = self.cls_head(cls_feat)
        
        if task == "classification":
            return {"logits": cls_logits}
        
        # 分割
        seg_logits = self.seg_head(x)
        seg_logits = self.upscale(seg_logits)
        
        if task == "segmentation":
            return {"seg_logits": seg_logits}
        
        # 多任务
        return {
            "cls_logits": cls_logits,
            "seg_logits": seg_logits,
        }


# =============================================================================
# 主模型
# =============================================================================

class MedMambaV2(nn.Module):
    """
    MedMamba V2 - 医学影像统一分类/分割模型
    
    融合架构:
    - CNN分支: 局部特征提取 (U-Mamba)
    - SSM分支: 四方向选择性扫描全局建模 (VMamba)
    - 特征融合: CNN-SSM动态加权融合 (TransMamba)
    - CTM动力学: 幻觉检测 + 轨迹稳定性监控
    
    优势:
    - O(n) vs ViT的O(n²)
    - 全局感受野 vs CNN的局部视野
    - 实时幻觉风险监控
    - 可切换分类/分割任务
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 2,
        num_diseases: int = 8,
        d_model: int = 384,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_ctm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = DualBranchEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_ctm=use_ctm,
        )
        
        self.head = MedMambaHead(d_model, num_classes, num_diseases)
        
        # Patch embedding用于降分辨率
        self.patch_embed = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
        )
        
        # 可学习cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        task: str = "classification",
        return_ctm: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        x: [B, 3, H, W]
        task: classification / segmentation / both
        return_ctm: 是否返回CTM幻觉风险分数
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, d_model, H/P, W/P]
        
        # 添加cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # 编码
        x, ctm_score = self.encoder(x, return_ctm=return_ctm)
        
        # 将x转为[B, C, H, W]格式给head
        # cls token信息注入到全局特征
        cls_out = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        x = x + cls_out  # 注入全局语义
        
        result = self.head(x, task=task if task != "both" else "classification")
        
        if return_ctm and ctm_score is not None:
            result["hallucination_risk"] = ctm_score
        
        return result


# 别名
MedMamba = MedMambaV2


# =============================================================================
# MedMamba V3 - 集成VMamba + HoME-MoE + CTM
# =============================================================================

class MedMambaV3(nn.Module):
    """
    MedMamba V3 - 完整融合架构
    
    融合组件:
    - VMamba: 四方向SSM扫描 (VMamba)
    - HoME-MoE: 分层软专家混合 (NeurIPS 2025)
    - Cross-Attention: CNN-SSM特征融合 (TransMamba)
    - CTM: 动力学幻觉检测
    
    优势:
    - 更全面的全局/局部特征建模
    - 专家多样性路由
    - 实时幻觉风险监控
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 2,
        num_diseases: int = 8,
        d_model: int = 384,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        use_ctm: bool = True,
        use_moe: bool = True,
        num_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_moe = use_moe
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
        )
        
        # VMamba编码器
        self.vmamba_encoder = VMamba2D(
            d_model=d_model,
            depth=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_cross_attn=True,  # 启用CNN-SSM融合
        )
        
        # CNN分支 (局部特征)
        self.cnn_branch = nn.ModuleList([
            CNNBranch(d_model, d_state)
            for _ in range(n_layers)
        ])
        
        # HoME-MoE
        if use_moe:
            self.moe = HoMEMoE2D(
                d_model=d_model,
                num_experts=num_experts,
                depth=2,
                routing_type="soft",
                dropout=dropout,
                use_ctm=use_ctm,
            )
        else:
            self.moe = None
        
        # CTM分析器
        if use_ctm:
            self.ctm = CTMTrajectoryAnalyzer(d_model)
        else:
            self.ctm = None
        
        # 分类/分割头
        self.head = MedMambaHead(d_model, num_classes, num_diseases)
        
        # 可学习cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        task: str = "classification",
        return_ctm: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        x: [B, 3, H, W]
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, d_model, H/P, W/P]
        
        # CNN分支 (每2层执行一次)
        cnn_feats = []
        for i, cnn in enumerate(self.cnn_branch):
            if i % 2 == 0:
                cnn_feats.append(cnn(x))
        
        # VMamba编码 (带CNN融合)
        ssm_feat = x
        for i, cnn_feat in enumerate(cnn_feats):
            ssm_feat = self.vmamba_encoder(ssm_feat, cnn_feat)
        
        # HoME-MoE处理
        moe_routing = None
        if self.moe is not None:
            moe_feat, moe_routing = self.moe(ssm_feat)
        else:
            moe_feat = ssm_feat
        
        # 注入cls token信息
        cls_out = moe_feat.mean(dim=(2, 3), keepdim=True)
        moe_feat = moe_feat + cls_out
        
        # CTM分析
        ctm_score = None
        if self.ctm is not None and return_ctm:
            x_flat = moe_feat.flatten(2).transpose(1, 2).unsqueeze(1)
            ctm_score = self.ctm.hallucination_score(x_flat)
        
        # 任务头
        result = self.head(moe_feat, task=task)
        
        if ctm_score is not None:
            result["hallucination_risk"] = ctm_score
        
        if moe_routing is not None:
            result["moe_entropy"] = torch.stack([r["entropy"] for r in moe_routing]).mean()
        
        return result


# =============================================================================
# 模型工厂
# =============================================================================

def create_medmamba(
    version: str = "v2",
    **kwargs,
) -> nn.Module:
    """
    创建MedMamba模型
    
    Args:
        version: v1/v2/v3
    """
    if version == "v1":
        from .selective_state_space import MedMambaBlock
        # 简单版本
        return MedMambaClassifier(**kwargs)
    elif version == "v2":
        return MedMambaV2(**kwargs)
    elif version == "v3":
        return MedMambaV3(**kwargs)
    else:
        raise ValueError(f"Unknown version: {version}")