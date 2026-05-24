"""
CrossScanRiskAnalyzer - Cross-Scan方向一致性风险分析器

核心公式 (来自MedMamba-Guard框架):
1. 空间特征散度: Risk_cross(i,j) = mean(||f_k - μ||²) 其中μ=(1/4)Σf_k
2. 余弦散度: Risk_cos = 1 - mean(cos(f_k, μ))
3. 融合: R_scan = λ₁*Risk_l2 + λ₂*Risk_cos

适用场景:
- VMamba的Cross-Scan四方向扫描一致性检测
- 检测各向异性特征中的方向不一致区域
- 识别可能的幻觉/错误特征

作者: MedMamba Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange


class CrossScanRiskAnalyzer(nn.Module):
    """
    Cross-Scan方向一致性风险分析器
    
    功能:
    1. 从VMamba的Cross-Scan机制提取四方向特征 F_dir1~F_dir4
    2. 计算空间特征散度 Risk_cross
    3. 计算余弦散度 Risk_cos
    4. 融合生成R_scan风险分数
    5. 生成2D一致性风险热力图
    """
    
    def __init__(
        self,
        feature_dim: int,
        lambda_l2: float = 0.5,
        lambda_cos: float = 0.5,
        eps: float = 1e-8,
    ):
        """
        Args:
            feature_dim: 特征维度
            lambda_l2: L2散度的融合权重
            lambda_cos: 余弦散度的融合权重
            eps: 数值稳定项
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.lambda_l2 = lambda_l2
        self.lambda_cos = lambda_cos
        self.eps = eps
        
        # 四方向的投影 (用于将特征映射到可比较的空间)
        self.dir_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim, bias=False)
            for _ in range(4)
        ])
        
        # 融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([lambda_l2, lambda_cos]))
        
        # 缓存
        self._scan_risk_score: Optional[float] = None
        self._scan_risk_map: Optional[torch.Tensor] = None
        self._directional_features: Optional[List[torch.Tensor]] = None
        self._current_risk_fused: Optional[torch.Tensor] = None
        
    def extract_directional_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        从输入提取四方向特征
        
        Args:
            x: [B, C, H, W] 输入特征图
            
        Returns:
            features: 四方向特征列表 [B, C, H*W] 各方向
        """
        B, C, H, W = x.shape
        
        # Cross-Scan四方向
        x_flat = x.view(B, C, H * W)
        
        # 方向0: 左上→右下 (raster scan)
        dir0 = x_flat
        
        # 方向1: 右下→左上 (reverse raster)
        dir1 = torch.flip(x_flat, dims=[-1])
        
        # 方向2: 右上→左下 (transpose raster)
        x_t = x.transpose(2, 3)  # [B, C, W, H]
        dir2 = x_t.reshape(B, C, H * W)
        
        # 方向3: 左下→右上 (reverse transpose)
        dir3 = torch.flip(dir2, dims=[-1])
        
        features = [dir0, dir1, dir2, dir3]
        
        # 应用投影使特征可比较
        projected = []
        for feat, proj in zip(features, self.dir_projections):
            # feat: [B, C, L] -> [B, L, C] -> proj -> [B, L, C] -> [B, C, L]
            feat_t = feat.transpose(1, 2)  # [B, L, C]
            feat_proj = proj(feat_t)        # [B, L, C]
            projected.append(feat_proj.transpose(1, 2))  # [B, C, L]
        
        self._directional_features = projected
        return projected
    
    def compute_divergence_maps(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算散度图
        
        公式:
        Risk_cross(i,j) = mean(||f_k - μ||²)
        Risk_cos = 1 - mean(cos(f_k, μ))
        
        Args:
            features: 四方向特征列表
            
        Returns:
            divergence_maps: 包含l2和cos散度图的字典
        """
        if len(features) != 4:
            raise ValueError(f"Expected 4 directional features, got {len(features)}")
            
        # Stack: [4, B, C, L]
        features_stack = torch.stack(features, dim=0)
        
        # 计算均值 μ = (1/4) Σ f_k
        mu = features_stack.mean(dim=0)  # [B, C, L]
        
        # ========== L2散度 ==========
        # 计算每个方向与均值的差的平方范数
        diffs = features_stack - mu.unsqueeze(0)  # [4, B, C, L]
        diff_norms_sq = torch.sum(diffs ** 2, dim=2)  # [4, B, L], 对C维度求和
        
        # Risk_cross = mean(||f_k - μ||²)
        risk_l2 = diff_norms_sq.mean(dim=0)  # [B, L]
        
        # ========== 余弦散度 ==========
        # 归一化
        features_norm = F.normalize(features_stack, p=2, dim=2)  # [4, B, C, L]
        mu_norm = F.normalize(mu, p=2, dim=1)                   # [B, C, L]
        
        # cos(f_k, μ) = f_k · μ / (||f_k|| * ||μ||)
        cos_sim = torch.sum(features_norm * mu_norm.unsqueeze(0), dim=2)  # [4, B, L]
        
        # Risk_cos = 1 - mean(cos)
        risk_cos = 1 - cos_sim.mean(dim=0)  # [B, L]
        
        # ========== 融合 ==========
        # 使用可学习权重
        weights = torch.softmax(self.fusion_weight, dim=0)
        risk_fused = weights[0] * risk_l2 + weights[1] * risk_cos  # [B, L]
        
        # 归一化到[0,1]
        risk_l2_norm = torch.sigmoid(risk_l2)
        risk_cos_norm = torch.sigmoid(risk_cos)
        risk_fused_norm = torch.sigmoid(risk_fused)
        
        # 保存用于后续获取
        self._current_risk_l2 = risk_l2_norm
        self._current_risk_cos = risk_cos_norm
        self._current_risk_fused = risk_fused_norm
        
        return {
            'risk_l2': risk_l2_norm,
            'risk_cos': risk_cos_norm,
            'risk_fused': risk_fused_norm,
        }
    
    def get_scan_risk_score(self, features: List[torch.Tensor] = None) -> float:
        if features is not None:
            divergence_maps = self.compute_divergence_maps(features)
            self._current_risk_fused = divergence_maps['risk_fused']
            self._scan_risk_score = divergence_maps['risk_fused'].mean().item()
            self._scan_risk_map = divergence_maps['risk_fused']
            return self._scan_risk_score
        if self._scan_risk_score is None:
            if self._current_risk_fused is not None:
                self._scan_risk_score = self._current_risk_fused.mean().item()
            else:
                return 0.0
                
        return self._scan_risk_score
    
    def get_scan_risk_map(self) -> torch.Tensor:
        """
        获取2D扫描风险热图
        
        Returns:
            risk_map: [B, H, W] 风险热图
        """
        if self._scan_risk_map is None:
            if self._current_risk_fused is not None:
                # [B, L] -> [B, H, W] (需要在调用前知道H, W)
                # 这里返回展平的风险，后续reshape
                self._scan_risk_map = self._current_risk_fused
            else:
                return torch.zeros(1, 1, 1)
                
        return self._scan_risk_map
    
    def reshape_risk_map(self, risk_map: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        将展平的风险图reshape为2D
        
        Args:
            risk_map: [B, L] 展平的风险
            H, W: 空间维度
            
        Returns:
            risk_map_2d: [B, H, W]
        """
        B = risk_map.shape[0]
        return risk_map.view(B, H, W)
    
    def get_inconsistent_regions(
        self,
        threshold: float = 0.3,
        min_area: int = 10
    ) -> List[Dict]:
        """
        获取高不一致区域
        
        Args:
            threshold: 风险阈值
            min_area: 最小区域面积
            
        Returns:
            regions: [{"bbox": [x1,y1,x2,y2], "risk_type": str, "risk_level": str}]
        """
        if self._current_risk_fused is None:
            return []
            
        risk_map = self._current_risk_fused  # [B, L]
        
        # 简单实现
        regions = []
        B = risk_map.shape[0]
        
        # 这里假设L已知为H*W，实际使用时需要传入H,W
        # 默认使用8x8网格
        grid_size = 8
        L = risk_map.shape[-1]
        side = int(L ** 0.5 + 0.5)
        
        for b in range(B):
            risk_b = risk_map[b]
            risk_2d = risk_b.view(side, side)
            
            for i in range(0, side, grid_size):
                for j in range(0, side, grid_size):
                    i_end = min(i + grid_size, side)
                    j_end = min(j + grid_size, side)
                    
                    region_risk = risk_2d[i:i_end, j:j_end].mean()
                    
                    if region_risk > threshold:
                        regions.append({
                            'bbox': [j, i, j_end, i_end],
                            'risk_type': 'cross_scan_inconsistency',
                            'risk_level': 'high' if region_risk > 0.6 else 'medium',
                        })
                        
        return regions
    
    def forward(
        self,
        x: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向接口
        
        Args:
            x: [B, C, H, W] 输入特征
            H, W: 空间维度 (可选)
            return_features: 是否返回中间特征
            
        Returns:
            results: 包含风险分数和热图的字典
        """
        # 提取四方向特征
        features = self.extract_directional_features(x)
        
        # 计算散度图
        divergence_maps = self.compute_divergence_maps(features)
        
        # 更新缓存
        self._scan_risk_score = divergence_maps['risk_fused'].mean().item()
        self._scan_risk_map = divergence_maps['risk_fused']
        
        # 构建结果
        results = {
            'risk_score': self._scan_risk_score,
            'risk_map': self._scan_risk_map,
            'risk_l2': divergence_maps['risk_l2'],
            'risk_cos': divergence_maps['risk_cos'],
        }
        
        if return_features:
            results['directional_features'] = features
            
        return results


# 简化的Cross-Scan风险分析器 (轻量版本)
class SimplifiedCrossScanRiskAnalyzer(nn.Module):
    """
    简化版本的Cross-Scan风险分析器
    
    不使用可学习参数，直接计算四方向的统计散度
    """
    
    def __init__(
        self,
        feature_dim: int = None,  # 为了兼容性保留
        lambda_l2: float = 0.5,
        lambda_cos: float = 0.5,
    ):
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_cos = lambda_cos
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W]
            
        Returns:
            risk_score: 标量风险分数
            risk_map: [B, H, W] 风险热图
        """
        B, C, H, W = x.shape
        
        # 提取四方向
        x_flat = x.view(B, C, H * W)
        
        dirs = [
            x_flat,
            torch.flip(x_flat, dims=[-1]),
            x.transpose(2, 3).reshape(B, C, H * W),
            torch.flip(x.transpose(2, 3).reshape(B, C, H * W), dims=[-1]),
        ]
        
        # 计算均值
        mu = torch.stack(dirs, dim=0).mean(dim=0)  # [B, C, L]
        
        # L2散度
        diffs = torch.stack(dirs, dim=0) - mu.unsqueeze(0)
        risk_l2 = torch.mean(torch.sum(diffs ** 2, dim=2), dim=0)  # [B, L]
        
        # 余弦散度
        dirs_norm = F.normalize(torch.stack(dirs, dim=0), p=2, dim=2)
        mu_norm = F.normalize(mu, p=2, dim=1)
        cos_sim = torch.mean(torch.sum(dirs_norm * mu_norm.unsqueeze(0), dim=2), dim=0)
        risk_cos = 1 - cos_sim
        
        # 融合
        risk = self.lambda_l2 * torch.sigmoid(risk_l2) + self.lambda_cos * torch.sigmoid(risk_cos)
        
        return {
            'risk_score': risk.mean().item(),
            'risk_map': risk.view(B, H, W),
        }