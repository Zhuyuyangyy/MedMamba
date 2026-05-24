"""
TaskConflictValidator - 分类-分割互证门控模块

核心公式 (来自MedMamba-Guard框架):
1. 分割空间证据:
   - Area_score: E_seg_area = clip(Area_lesion / (H*W*τ))
   - Compactness_score: = 4π*Area / Perimeter²
   - Boundary_stability: 边界状态稳定性
2. 综合E_seg = α*Area + β*Compactness + γ*BoundaryStability
3. 冲突分数: S_conflict = |P_cls - E_seg|
4. 门控判决: PASS / CONFLICT_LESION_MISSED / CONFLICT_OVER_CONFIDENT / REVIEW

适用场景:
- 分类任务(整体诊断)与分割任务(局部病灶)的互证
- 检测分类与分割结果不一致的情况
- 作为硬门控规则的输入

作者: MedMamba Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class TaskConflictValidator(nn.Module):
    """
    分类-分割互证门控模块
    
    功能:
    1. 从分类头获取P_cls (整体患病概率)
    2. 从分割头获取M_seg (二值掩码)
    3. 计算分割空间证据 (Area, Compactness, Boundary Stability)
    4. 计算冲突分数并做门控判决
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        alpha_area: float = 0.5,
        beta_compactness: float = 0.3,
        gamma_boundary: float = 0.2,
        conflict_threshold: float = 0.4,
        eps: float = 1e-8,
    ):
        """
        Args:
            hidden_dim: 特征维度
            alpha_area: Area_score权重
            beta_compactness: Compactness_score权重
            gamma_boundary: Boundary_stability权重
            conflict_threshold: 冲突阈值
            eps: 数值稳定项
        """
        super().__init__()
        self.alpha_area = alpha_area
        self.beta_compactness = beta_compactness
        self.gamma_boundary = gamma_boundary
        self.conflict_threshold = conflict_threshold
        self.eps = eps
        
        # 归一化因子 (可选)
        self.tau = 0.1  # 面积归一化系数
        
        # 缓存
        self._last_p_cls: Optional[torch.Tensor] = None
        self._last_m_seg: Optional[torch.Tensor] = None
        self._last_conflict_score: Optional[float] = None
        self._last_judgment: Optional[str] = None
        
    def _compute_area_score(self, mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        计算面积得分
        
        E_seg_area = clip(Area_lesion / (H*W*τ))
        
        Args:
            mask: [B, 1, H, W] 二值分割掩码
            H, W: 空间维度
            
        Returns:
            area_score: [B] 面积分数
        """
        # 病灶面积
        area = mask.sum(dim=(2, 3))  # [B, 1]
        
        # 归一化
        total_area = H * W * self.tau
        area_score = (area / total_area).clamp(0, 1)
        
        return area_score.squeeze(-1)  # [B]
    
    def _compute_compactness_score(self, mask: torch.Tensor) -> torch.Tensor:
        """
        计算紧密度得分
        
        Compactness = 4π*Area / Perimeter²
        
        Args:
            mask: [B, 1, H, W] 二值分割掩码
            
        Returns:
            compactness_score: [B] 紧密度分数
        """
        # 计算周长 (使用sobel梯度近似)
        # 方法1: 使用torch编写的简单周长计算
        def compute_perimeter(m):
            """计算二值掩码的周长"""
            # 膨胀然后减去原图得到边缘
            kernel = torch.ones(3, 3, device=m.device, dtype=m.dtype)
            m_pad = F.pad(m.float(), (1, 1, 1, 1), mode='constant', value=0)
            
            # 简单膨胀
            dilated = F.max_pool2d(m_pad, kernel_size=3, stride=1, padding=0)
            perimeter = (dilated - m.float()).sum(dim=(1, 2, 3))
            return perimeter
        
        area = mask.sum(dim=(2, 3)).float().squeeze(-1)  # [B]
        perimeter = compute_perimeter(mask)  # [B]
        
        # 紧密度公式: 4π*Area / Perimeter²
        # 圆形紧密度=1, 狭长形状紧密度<1
        compactness = (4 * np.pi * area) / (perimeter ** 2 + self.eps)
        compactness_score = compactness.clamp(0, 1)  # 归一化
        
        return compactness_score.squeeze(-1)  # [B]
    
    def _compute_boundary_stability(self, mask: torch.Tensor) -> torch.Tensor:
        """
        计算边界稳定性
        
        使用Canny边缘检测后的边缘质量作为边界稳定性指标
        简化版本: 使用梯度强度方差
        
        Args:
            mask: [B, 1, H, W] 二值分割掩码
            
        Returns:
            boundary_stability: [B] 边界稳定性分数
        """
        # 简化: 使用Sobel算子计算边缘强度
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=mask.dtype, device=mask.device).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=mask.dtype, device=mask.device).unsqueeze(0).unsqueeze(0)
        
        mask_float = mask.float()
        
        # 计算梯度
        grad_x = F.conv2d(mask_float, kernel_x, padding=1)
        grad_y = F.conv2d(mask_float, kernel_y, padding=1)
        
        # 边缘强度
        edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)
        
        # 边界稳定性: 边缘强度的方差 (方差小=稳定)
        # 注意: 这里是边缘区域, 所以只考虑边缘点
        edge_mask = edge_strength > 0
        if edge_mask.sum() > 0:
            edge_values = edge_strength[edge_mask]
            boundary_variance = torch.var(edge_values)
            # 方差小=稳定, 转为分数
            boundary_stability = 1 / (1 + boundary_variance)
        else:
            boundary_stability = torch.ones_like(mask[:, 0, 0, 0])
        
        return boundary_stability
    
    def _compute_segmentation_evidence(
        self,
        mask: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        计算综合分割证据
        
        E_seg = α*Area + β*Compactness + γ*BoundaryStability
        
        Args:
            mask: [B, 1, H, W] 二值分割掩码
            H, W: 空间维度
            
        Returns:
            e_seg: [B] 综合分割证据
        """
        # Area score
        e_area = self._compute_area_score(mask, H, W)
        
        # Compactness score
        e_compactness = self._compute_compactness_score(mask)
        
        # Boundary stability
        e_boundary = self._compute_boundary_stability(mask)
        
        # 融合
        e_seg = (
            self.alpha_area * e_area +
            self.beta_compactness * e_compactness +
            self.gamma_boundary * e_boundary
        )
        
        return e_seg
    
    def _compute_conflict_score(
        self,
        p_cls: torch.Tensor,
        e_seg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算冲突分数
        
        S_conflict = |P_cls - E_seg|
        
        Args:
            p_cls: [B] 分类概率
            e_seg: [B] 分割证据
            
        Returns:
            conflict: [B] 冲突分数
            direction: [B] 冲突方向 (+1=分类高于分割, -1=分割高于分类)
        """
        conflict = torch.abs(p_cls - e_seg)
        direction = torch.sign(p_cls - e_seg)  # +1: 可能漏诊, -1: 可能过度诊断
        
        return conflict, direction
    
    def _make_judgment(
        self,
        conflict: torch.Tensor,
        direction: torch.Tensor,
        p_cls: torch.Tensor,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        做出门控判决
        
        判决结果:
        - PASS: 分类和分割一致
        - CONFLICT_LESION_MISSED: 分类高但分割低 (可能漏诊)
        - CONFLICT_OVER_CONFIDENT: 分类低但分割高 (可能过度诊断)
        - REVIEW: 冲突严重，需要人工审核
        
        Args:
            conflict: [B] 冲突分数
            direction: [B] 冲突方向
            p_cls: [B] 分类概率
            threshold: 冲突阈值
            
        Returns:
            judgment: [B] 判决结果编码
            reason: [B] 判决原因
        """
        batch_size = conflict.shape[0]
        judgments = []
        reasons = []
        
        for i in range(batch_size):
            c = conflict[i].item()
            d = direction[i].item()
            p = p_cls[i].item()
            
            if c < threshold * 0.5:
                # 低冲突
                judgment = "PASS"
                reason = "low_conflict"
            elif c > threshold * 1.5:
                # 高冲突，需要审核
                judgment = "REVIEW"
                if d > 0:
                    reason = "high_conflict_classification_higher"
                else:
                    reason = "high_conflict_segmentation_higher"
            else:
                # 中等冲突，判定类型
                if d > 0 and p > 0.5:
                    # 分类预测有病(高置信)但分割没找到明显区域
                    judgment = "CONFLICT_LESION_MISSED"
                    reason = "classification_lesion_but_segmentation_empty"
                elif d < 0 and p < 0.5:
                    # 分类预测正常但分割发现了可疑区域
                    judgment = "CONFLICT_OVER_CONFIDENT"
                    reason = "classification_normal_but_segmentation_found"
                else:
                    judgment = "REVIEW"
                    reason = "moderate_conflict"
            
            judgments.append(judgment)
            reasons.append(reason)
            
        # 转为tensor
        judgment_map = {
            "PASS": 0,
            "CONFLICT_LESION_MISSED": 1,
            "CONFLICT_OVER_CONFIDENT": 2,
            "REVIEW": 3,
        }
        judgment_codes = torch.tensor(
            [judgment_map[j] for j in judgments],
            device=conflict.device,
            dtype=torch.long
        )
        
        return judgment_codes, judgments
    
    def validate(
        self,
        p_cls: torch.Tensor,
        m_seg: torch.Tensor,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        验证分类和分割的一致性
        
        Args:
            p_cls: [B, num_classes] 或 [B] 分类概率
            m_seg: [B, 1, H, W] 二值分割掩码
            return_details: 是否返回详细指标
            
        Returns:
            results: {
                'judgment': str,  # 最终判决
                'conflict_score': float,
                'p_cls': ...,
                'e_seg': ...,
                'e_area': ...,
                'e_compactness': ...,
                'e_boundary': ...,
            }
        """
        # 处理输入形状
        if p_cls.dim() > 1:
            p_cls = p_cls[:, 1] if p_cls.shape[1] > 1 else p_cls[:, 0]  # 取positive概率
            
        # 确保p_cls是[B]
        p_cls = p_cls.squeeze(-1) if p_cls.dim() > 1 else p_cls
        
        B, _, H, W = m_seg.shape
        
        # 计算分割证据
        e_seg = self._compute_segmentation_evidence(m_seg, H, W)
        
        # 计算冲突分数
        conflict, direction = self._compute_conflict_score(p_cls, e_seg)
        
        # 做出判决
        judgment_codes, judgments = self._make_judgment(
            conflict, direction, p_cls, self.conflict_threshold
        )
        
        # 缓存
        self._last_p_cls = p_cls
        self._last_m_seg = m_seg
        self._last_conflict_score = conflict.mean().item()
        self._last_judgment = judgments[0] if len(judgments) == 1 else judgments
        
        # 构建结果
        action_map = {
            'PASS': 'PASS',
            'CONFLICT_LESION_MISSED': 'REVIEW',
            'CONFLICT_OVER_CONFIDENT': 'REVIEW',
            'REVIEW': 'REVIEW',
        }
        e_area_val = self._compute_area_score(m_seg, H, W)
        e_compact_val = self._compute_compactness_score(m_seg)
        e_boundary_val = self._compute_boundary_stability(m_seg)
        results = {
            'judgment': judgments if B > 1 else judgments[0],
            'action': action_map.get(judgments[0], 'REVIEW') if B == 1 else [action_map.get(j, 'REVIEW') for j in judgments],
            'judgment_code': judgment_codes,
            'conflict_score': conflict,
            'conflict_direction': direction,
            'p_cls': p_cls,
            'e_seg': e_seg,
            'seg_evidence': {
                'area_score': e_area_val.mean().item() if isinstance(e_area_val, torch.Tensor) else float(e_area_val),
                'compactness_score': e_compact_val.mean().item() if isinstance(e_compact_val, torch.Tensor) else float(e_compact_val),
                'boundary_stability': e_boundary_val.mean().item() if isinstance(e_boundary_val, torch.Tensor) else float(e_boundary_val),
            },
        }
        
        if return_details:
            results.update({
                'e_area': self._compute_area_score(m_seg, H, W),
                'e_compactness': self._compute_compactness_score(m_seg),
                'e_boundary': self._compute_boundary_stability(m_seg),
            })
            
        return results
    
    def get_task_conflict_score(self) -> float:
        """获取上次调用的冲突分数"""
        if self._last_conflict_score is None:
            return 0.0
        return self._last_conflict_score
    
    def get_last_judgment(self) -> str:
        """获取上次调用的判决结果"""
        if self._last_judgment is None:
            return "PASS"
        return self._last_judgment
    
    def forward(
        self,
        p_cls: torch.Tensor,
        m_seg: torch.Tensor,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向接口
        """
        return self.validate(p_cls, m_seg, return_details)


# 辅助函数
def create_task_validator(
    hidden_dim: int = 768,
    alpha_area: float = 0.5,
    beta_compactness: float = 0.3,
    gamma_boundary: float = 0.2,
    conflict_threshold: float = 0.4,
) -> TaskConflictValidator:
    """创建TaskConflictValidator的工厂函数"""
    return TaskConflictValidator(
        hidden_dim=hidden_dim,
        alpha_area=alpha_area,
        beta_compactness=beta_compactness,
        gamma_boundary=gamma_boundary,
        conflict_threshold=conflict_threshold,
    )