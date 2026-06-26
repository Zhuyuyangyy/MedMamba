"""
MedMamba-Guard - 基于CTM状态轨迹的医学影像可信推理框架

核心组件:
1. CNN-SSM双分支编码器 (保留原MedMamba的SS-Conv-SSM)
2. CTMMonitor - SSM状态轨迹稳定性分析
3. CrossScanRiskAnalyzer - 四方向一致性风险检测
4. TaskConflictValidator - 分类-分割互证门控
5. HardGatingRules - 硬门控规则引擎
6. RiskHeatmapGenerator - 风险热力图生成
7. AuditLogger - 推理审计日志

核心公式:
- R_state = α*V_norm + β*D_Δ + γ*(1-C_layer) + δ*R_overconfident
- R_scan = λ₁*Risk_l2 + λ₂*Risk_cos
- S_conflict = |P_cls - E_seg|
- R_total = α*R_state + β*R_scan + γ*R_task + η*R_entropy

硬门控规则:
if R_total > θ_high(0.7): action = doctor_review
if R_task > θ_conflict(0.4): action = doctor_review  
if confidence > θ_conf(0.85) and R_state > θ_state(0.5): action = overconfidence_warning

作者: MedMamba Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Dict, List, Optional, Tuple, Any
import math
from datetime import datetime
import json

# 导入现有模块
from .vmamba_blocks import (
    CrossScan, CrossMerge, SS2D, VSSBlock2D,
    CrossAttentionFusion, channel_shuffle,
)
from .selective_state_space import (
    RMSNorm, MambaBlock, MedMambaBlock,
)

# 导入新增模块
from .ctm_monitor import CTMMonitor
from .cross_scan_risk import CrossScanRiskAnalyzer, SimplifiedCrossScanRiskAnalyzer
from .task_conflict_validator import TaskConflictValidator


# =============================================================================
# 硬门控规则引擎
# =============================================================================

class HardGatingRules:
    """
    硬门控规则引擎
    
    规则:
    1. R_total > θ_high(0.7) → doctor_review
    2. R_task > θ_conflict(0.4) → doctor_review
    3. confidence > θ_conf(0.85) and R_state > θ_state(0.5) → overconfidence_warning
    """
    
    def __init__(
        self,
        theta_high: float = 0.7,
        theta_conflict: float = 0.4,
        theta_conf: float = 0.85,
        theta_state: float = 0.5,
    ):
        self.theta_high = theta_high
        self.theta_conflict = theta_conflict
        self.theta_conf = theta_conf
        self.theta_state = theta_state
        
        # 权重 (用于计算R_total)
        self.w_state = 0.3
        self.w_scan = 0.2
        self.w_task = 0.3
        self.w_entropy = 0.2
        
    def compute_total_risk(
        self,
        r_state: float,
        r_scan: float,
        r_task: float,
        r_entropy: float = 0.0,
    ) -> float:
        """计算综合风险分数"""
        return (
            self.w_state * r_state +
            self.w_scan * r_scan +
            self.w_task * r_task +
            self.w_entropy * r_entropy
        )
    
    def evaluate(
        self,
        r_state: float,
        r_scan: float,
        r_task: float,
        r_entropy: float,
        confidence: float,
    ) -> Tuple[str, str, Dict]:
        """
        评估风险并返回动作和原因
        
        Returns:
            action: PASS / doctor_review / overconfidence_warning
            reason: 原因描述
            details: 详细规则命中情况
        """
        r_total = self.compute_total_risk(r_state, r_scan, r_task, r_entropy)
        
        details = {
            'r_total': r_total,
            'r_state': r_state,
            'r_scan': r_scan,
            'r_task': r_task,
            'r_entropy': r_entropy,
            'confidence': confidence,
            'rules_triggered': [],
        }
        
        # 规则1: R_total > θ_high
        if r_total > self.theta_high:
            details['rules_triggered'].append('R_total > theta_high')
            return 'doctor_review', f"high_total_risk({r_total:.3f})", details
        
        # 规则2: R_task > θ_conflict
        if r_task > self.theta_conflict:
            details['rules_triggered'].append('R_task > theta_conflict')
            return 'doctor_review', f"task_conflict_detected({r_task:.3f})", details
        
        # 规则3: confidence > θ_conf and R_state > θ_state
        if confidence > self.theta_conf and r_state > self.theta_state:
            details['rules_triggered'].append('overconfidence_detected')
            return 'overconfidence_warning', f"high_confidence({confidence:.3f})_but_unstable_state({r_state:.3f})", details
        
        # 默认
        return 'PASS', 'all_risk_within_threshold', details


# =============================================================================
# 风险热力图生成器
# =============================================================================

class RiskHeatmapGenerator:
    """
    风险热力图生成器
    
    融合CTM和Cross-Scan的风险图为统一输出
    """
    
    def __init__(
        self,
        w_ctm: float = 0.6,
        w_scan: float = 0.4,
    ):
        self.w_ctm = w_ctm
        self.w_scan = w_scan
        
    def generate(
        self,
        ctm_risk_map: torch.Tensor,
        scan_risk_map: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        B = ctm_risk_map.shape[0]

        if ctm_risk_map.numel() <= B:
            ctm_risk_map = ctm_risk_map.new_zeros(B, H, W)
        elif ctm_risk_map.dim() == 2:
            ctm_risk_map = ctm_risk_map.view(B, H, W)
        if scan_risk_map.numel() <= B:
            scan_risk_map = scan_risk_map.new_zeros(B, H, W)
        elif scan_risk_map.dim() == 2:
            scan_risk_map = scan_risk_map.view(B, H, W)

        fused = self.w_ctm * ctm_risk_map + self.w_scan * scan_risk_map

        return fused
    
    def generate_with_conflict(
        self,
        ctm_risk_map: torch.Tensor,
        scan_risk_map: torch.Tensor,
        conflict_map: torch.Tensor,
        H: int,
        W: int,
        w_conflict: float = 0.3,
    ) -> torch.Tensor:
        """
        生成包含冲突信息的融合风险热图
        
        Args:
            ctm_risk_map: [B, H*W] CTM风险图
            scan_risk_map: [B, H*W] Cross-Scan风险图
            conflict_map: [B, H*W] 冲突风险图
            H, W: 空间维度
            w_conflict: 冲突权重
        """
        base_fused = self.generate(ctm_risk_map, scan_risk_map, H, W)
        
        # 冲突区域加权
        conflict_map_2d = conflict_map.view(B, H, W) if conflict_map.dim() == 2 else conflict_map
        
        return base_fused + w_conflict * conflict_map_2d


# =============================================================================
# 审计日志
# =============================================================================

class AuditLogger:
    """
    推理审计日志
    
    记录每次推理的:
    - 模型版本
    - CTM指标
    - Cross-Scan指标
    - 门控判决
    - 风险分数
    """
    
    def __init__(self, model_version: str = "MedMamba-Guard-v1.0"):
        self.model_version = model_version
        self.enabled = True
        self._current_log: Optional[Dict] = None
        
    def start_inference(self, input_shape: Tuple) -> Dict:
        """开始一次推理记录"""
        self._current_log = {
            'model_version': self.model_version,
            'timestamp': datetime.now().isoformat(),
            'input_shape': list(input_shape),
            'ctm_metrics': {},
            'scan_metrics': {},
            'conflict_metrics': {},
            'final_decision': {},
        }
        return self._current_log
    
    def log_ctm_metrics(self, ctm_metrics: Dict):
        if self._current_log is not None:
            self._current_log['ctm_metrics'] = {
                k: v.mean().item() if isinstance(v, torch.Tensor) and v.numel() > 1 else (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in ctm_metrics.items()
            }
            
    def log_scan_metrics(self, scan_metrics: Dict):
        if self._current_log is not None:
            self._current_log['scan_metrics'] = {
                k: v.mean().item() if isinstance(v, torch.Tensor) and v.numel() > 1 else (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in scan_metrics.items()
            }
            
    def log_conflict_metrics(self, conflict_metrics: Dict):
        if self._current_log is not None:
            self._current_log['conflict_metrics'] = {
                k: v.float().mean().item() if isinstance(v, torch.Tensor) and v.numel() > 1 else (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in conflict_metrics.items()
            }
            
    def log_final_decision(
        self,
        prediction: str,
        confidence: float,
        risk_score: float,
        action: str,
        reason: str,
    ):
        """记录最终决策"""
        if self._current_log is not None:
            self._current_log['final_decision'] = {
                'prediction': prediction,
                'confidence': confidence,
                'risk_score': risk_score,
                'action': action,
                'reason': reason,
            }
            
    def get_current_log(self) -> Optional[Dict]:
        """获取当前日志"""
        return self._current_log
    
    def disable(self):
        """禁用日志"""
        self.enabled = False
        
    def enable(self):
        """启用日志"""
        self.enabled = True


# =============================================================================
# CNN分支
# =============================================================================

class CNNBranch(nn.Module):
    """CNN分支 - 提取局部空间特征"""
    
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            d_model, d_model,
            kernel_size=3, padding=1,
            groups=d_model,
        )
        self.bn1 = nn.BatchNorm2d(d_model)
        self.act1 = nn.SiLU(inplace=True)
        
        self.pointwise = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(d_model)
        self.act2 = nn.SiLU(inplace=True)
        
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(d_model * 2),
            nn.SiLU(inplace=True),
        )
        
        self.upscale = nn.Conv2d(d_model * 2, d_model, kernel_size=1) if d_state > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, downsample: bool = False) -> torch.Tensor:
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
# 特征融合模块
# =============================================================================

class FeatureFusion(nn.Module):
    """CNN-SSM特征融合"""
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.cnn_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        
        self.ssm_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=1),
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, cnn_feat: torch.Tensor, ssm_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = cnn_feat.shape
        
        c = self.cnn_proj(cnn_feat.flatten(2).transpose(1, 2))
        s = self.ssm_proj(ssm_feat).flatten(2).transpose(1, 2)
        
        w = torch.sigmoid(self.fusion_weight)
        fused = w * c + (1 - w) * s
        
        return fused.transpose(1, 2).view(B, C, H, W)


# =============================================================================
# MedMamba-Guard 主模型
# =============================================================================

class MedMambaGuard(nn.Module):
    """
    MedMamba-Guard 主模型
    
    整合所有CTM监控和风险分析组件:
    - CNN-SSM双分支编码器
    - CTMMonitor (状态轨迹分析)
    - CrossScanRiskAnalyzer (方向一致性分析)
    - TaskConflictValidator (分类-分割互证)
    - HardGatingRules (硬门控)
    - RiskHeatmapGenerator (风险热力图)
    - AuditLogger (审计日志)
    """
    
    def __init__(
        self,
        d_model=384,
        n_layers=12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        trajectory_window: int = 5,
        delta_window: int = 3,
        lambda_l2: float = 0.5,
        lambda_cos: float = 0.5,
        alpha_area: float = 0.5,
        beta_compactness: float = 0.3,
        gamma_boundary: float = 0.2,
        conflict_threshold: float = 0.4,
        theta_high: float = 0.7,
        theta_conflict: float = 0.4,
        theta_conf: float = 0.85,
        theta_state: float = 0.5,
        w_ctm: float = 0.6,
        w_scan: float = 0.4,
        use_ctm: bool = True,
        use_cross_scan: bool = True,
        use_task_validator: bool = True,
        model_version: str = "MedMamba-Guard-v1.0",
        # API-compat kwargs (MedMamba V2/V3-style signatures).
        # MedMambaGuard operates on raw spatial features, so img_size/patch_size
        # are accepted but unused.
        img_size: int = None,
        patch_size: int = None,
        **kwargs,
    ):
        from .ssm_config import MedMambaGuardConfig
        if isinstance(d_model, MedMambaGuardConfig):
            cfg = d_model
            d_model = cfg.d_model
            n_layers = getattr(cfg, 'n_layers', 12)
            d_state = cfg.d_state
            d_conv = cfg.d_conv
            expand = cfg.expand
            num_classes = cfg.num_classes
            dropout = cfg.dropout
            trajectory_window = cfg.trajectory_window
            delta_window = cfg.delta_window
            lambda_l2 = cfg.lambda_l2
            lambda_cos = cfg.lambda_cos
            alpha_area = cfg.alpha_area
            beta_compactness = cfg.beta_compactness
            gamma_boundary = cfg.gamma_boundary
            conflict_threshold = cfg.conflict_threshold
            theta_high = cfg.hard_gate_high
            theta_conflict = cfg.conflict_threshold
            theta_conf = getattr(cfg, 'theta_conf', 0.85)
            theta_state = cfg.state_risk_threshold
            w_ctm = cfg.w_ctm
            w_scan = cfg.w_scan
            use_ctm = cfg.use_ctm
            use_cross_scan = cfg.use_cross_scan
            use_task_validator = cfg.use_task_validator
            model_version = getattr(cfg, 'model_version', 'MedMamba-Guard-v1.0')
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.model_version = model_version
        
        # ========== 编码器 ==========
        self.input_proj = nn.Conv2d(3, d_model, kernel_size=1)
        
        # SSM分支
        self.ssm_branch = nn.ModuleList([
            VSSBlock2D(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
        # CNN分支
        self.cnn_branch = nn.ModuleList([
            CNNBranch(d_model, d_state)
            for _ in range(n_layers)
        ])
        
        # 特征融合
        self.fusion = FeatureFusion(d_model)
        
        # ========== CTM监控 ==========
        self.use_ctm = use_ctm
        if use_ctm:
            self.ctm_monitor = CTMMonitor(
                hidden_dim=d_model,
                trajectory_window=trajectory_window,
                delta_window=delta_window,
            )
        
        # ========== Cross-Scan风险分析 ==========
        self.use_cross_scan = use_cross_scan
        if use_cross_scan:
            self.scan_analyzer = CrossScanRiskAnalyzer(
                feature_dim=d_model,
                lambda_l2=lambda_l2,
                lambda_cos=lambda_cos,
            )
        
        # ========== 任务冲突验证 ==========
        self.use_task_validator = use_task_validator
        if use_task_validator:
            self.task_validator = TaskConflictValidator(
                hidden_dim=d_model,
                alpha_area=alpha_area,
                beta_compactness=beta_compactness,
                gamma_boundary=gamma_boundary,
                conflict_threshold=conflict_threshold,
            )
        
        # ========== 门控和热力图 ==========
        self.hard_gating = HardGatingRules(
            theta_high=theta_high,
            theta_conflict=theta_conflict,
            theta_conf=theta_conf,
            theta_state=theta_state,
        )
        
        self.risk_heatmap_gen = RiskHeatmapGenerator(
            w_ctm=w_ctm,
            w_scan=w_scan,
        )
        
        # ========== 审计日志 ==========
        self.audit_logger = AuditLogger(model_version=model_version)
        
        # ========== 分类头 ==========
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        
        # ========== 分割头 (轻量U-Net风格) ==========
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(d_model // 2, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(d_model // 4, 1, kernel_size=1),
        )
        
        # 最终归一化
        self.final_norm = RMSNorm(d_model)
        
    def _register_ssm_hooks(self):
        """注册SSM层的hook以提取隐藏态"""
        if self.use_ctm and hasattr(self, 'ctm_monitor'):
            self.ctm_monitor.clear_cache()
            self.ctm_monitor.register_hooks(self)
    
    def _unregister_hooks(self):
        """移除hook"""
        if self.use_ctm and hasattr(self, 'ctm_monitor'):
            self.ctm_monitor.remove_hooks()
    
    def forward(
        self,
        x: torch.Tensor,
        return_risk: bool = True,
        return_heatmaps: bool = True,
        return_audit: bool = False,
        mode: str = 'eval',  # eval / train / guard
    ) -> Dict[str, Any]:
        """
        前向推理
        
        Args:
            x: [B, 3, H, W] 输入图像
            return_risk: 是否返回风险分数
            return_heatmaps: 是否返回风险热力图
            return_audit: 是否返回审计日志
            mode: eval=只推理, train=训练模式, guard=完整风险评估
            
        Returns:
            output: {
                "prediction": str,
                "confidence": float,
                "risk_score": float,
                "risk_components": {...},
                "risk_heatmap": torch.Tensor,
                "review_regions": [...],
                "audit_log": {...}
            }
        """
        B, C, H, W = x.shape
        
        # 启动审计日志
        if return_audit:
            self.audit_logger.start_inference(x.shape)
        
        # ========== 编码器前向 ==========
        # 清理CTM缓存
        if self.use_ctm:
            self.ctm_monitor.clear_cache()
        
        # 注册hook提取隐藏态
        if return_risk and self.use_ctm and mode in ['eval', 'guard']:
            self._register_ssm_hooks()
        
        # 投影
        x = self.input_proj(x)  # [B, d_model, H, W]
        
        # 双分支处理
        for layer_idx in range(self.n_layers):
            # SSM分支
            ssm_out = self.ssm_branch[layer_idx](x)
            
            # CNN分支
            if layer_idx % 2 == 0:
                cnn_out = self.cnn_branch[layer_idx](x)
            else:
                cnn_out = self.cnn_branch[layer_idx](ssm_out)
            
            # 融合
            x = self.fusion(cnn_out, ssm_out)
        
        # 最终归一化
        encoded = self.final_norm(x)
        
        # ========== 分类预测 ==========
        cls_logits = self.classification_head(encoded)
        cls_probs = F.softmax(cls_logits, dim=-1)
        confidence, pred_idx = cls_probs.max(dim=-1)
        
        prediction = "lesion" if pred_idx[0].item() == 1 else "normal"
        confidence_val = confidence[0].item()
        
        # ========== 分割预测 ==========
        seg_mask_logits = self.segmentation_head(encoded)
        seg_mask = torch.sigmoid(seg_mask_logits)  # [B, 1, H, W]
        seg_binary = (seg_mask > 0.5).float()
        
        # ========== 风险评估 ==========
        r_state = 0.0
        r_scan = 0.0
        r_task = 0.0
        r_entropy = 0.0
        ctm_metrics = {}
        scan_metrics = {}
        
        if return_risk and mode in ['eval', 'guard']:
            # 1. CTM风险
            if self.use_ctm and self.ctm_monitor.hidden_states:
                ctm_metrics = self.ctm_monitor.compute_ctm_metrics(
                    hidden_states=None,
                    output_confidence=confidence_val,
                )
                r_state = ctm_metrics.get('r_state', torch.tensor(0.0))
                if isinstance(r_state, torch.Tensor):
                    r_state = r_state.item()
                r_state = min(max(r_state, 0.0), 1.0)
                    
            # 2. Cross-Scan风险
            if self.use_cross_scan:
                scan_results = self.scan_analyzer(encoded, H, W)
                r_scan = scan_results.get('risk_score', 0.0)
                r_scan = min(r_scan, 1.0)
                scan_metrics = {
                    'risk_l2': scan_results.get('risk_l2'),
                    'risk_cos': scan_results.get('risk_cos'),
                }
                
            # 3. 任务冲突风险
            if self.use_task_validator:
                conflict_results = self.task_validator(
                    p_cls=cls_probs,
                    m_seg=seg_binary,
                    return_details=True,
                )
                r_task = conflict_results.get('conflict_score', torch.tensor(0.0))
                if isinstance(r_task, torch.Tensor):
                    r_task = r_task.mean().item()
                r_task = min(max(r_task, 0.0), 1.0)
                    
            # 4. 熵风险 (预测不确定性)
            r_entropy = -(cls_probs * torch.log(cls_probs + 1e-8)).sum(dim=-1).mean().item()
            
            # 记录审计日志
            if return_audit:
                self.audit_logger.log_ctm_metrics(ctm_metrics)
                self.audit_logger.log_scan_metrics(scan_metrics)
                self.audit_logger.log_conflict_metrics(conflict_results if self.use_task_validator else {})
        
        # ========== 硬门控评估 ==========
        action = "PASS"
        reason = ""
        if mode in ['eval', 'guard']:
            action, reason, gating_details = self.hard_gating.evaluate(
                r_state=r_state,
                r_scan=r_scan,
                r_task=r_task,
                r_entropy=r_entropy,
                confidence=confidence_val,
            )
        else:
            gating_details = {}
        
        # ========== 风险热力图 ==========
        risk_heatmap = torch.zeros(B, H, W, device=x.device)
        if return_heatmaps and mode in ['eval', 'guard']:
            # CTM风险热图
            if self.use_ctm:
                ctm_risk_map = self.ctm_monitor.get_state_risk_map()
                if ctm_risk_map.shape[0] != B or ctm_risk_map.numel() <= B:
                    ctm_risk_map = torch.zeros(B, H * W, device=x.device)
                elif ctm_risk_map.dim() == 1:
                    ctm_risk_map = ctm_risk_map.unsqueeze(0).expand(B, -1)
            else:
                ctm_risk_map = torch.zeros(B, H * W, device=x.device)
                
            # Cross-Scan风险热图
            if self.use_cross_scan:
                scan_risk_map = self.scan_analyzer.get_scan_risk_map()
                if scan_risk_map.shape[0] != B or scan_risk_map.numel() <= B:
                    scan_risk_map = torch.zeros(B, H * W, device=x.device)
                elif scan_risk_map.dim() == 1:
                    scan_risk_map = scan_risk_map.unsqueeze(0).expand(B, -1)
            else:
                scan_risk_map = torch.zeros(B, H * W, device=x.device)
                
            # 融合
            risk_heatmap = self.risk_heatmap_gen.generate(
                ctm_risk_map=ctm_risk_map.view(B, -1),
                scan_risk_map=scan_risk_map.view(B, -1),
                H=H,
                W=W,
            )
        
        # ========== 高风险区域检测 ==========
        review_regions = []
        if mode in ['eval', 'guard']:
            # 从CTM获取过度自信区域
            if self.use_ctm:
                review_regions.extend(
                    self.ctm_monitor.get_overconfidence_regions(threshold=0.5)
                )
                
            # 从Cross-Scan获取不一致区域
            if self.use_cross_scan:
                review_regions.extend(
                    self.scan_analyzer.get_inconsistent_regions(threshold=0.3)
                )
        
        # ========== 移除hook ==========
        self._unregister_hooks()
        
        # ========== 构建输出 ==========
        output = {
            "prediction": prediction,
            "confidence": confidence_val,
            "risk_score": self.hard_gating.compute_total_risk(r_state, r_scan, r_task, r_entropy),
            "risk_components": {
                "R_state": r_state,
                "R_scan": r_scan,
                "R_task": r_task,
                "R_entropy": r_entropy,
            },
            "action": action,
            "reason": reason,
        }
        
        if return_heatmaps:
            output["risk_heatmap"] = risk_heatmap

        output["review_regions"] = review_regions

        if not return_audit:
            self.audit_logger.start_inference(x.shape)
        self.audit_logger.log_final_decision(
            prediction=prediction,
            confidence=confidence_val,
            risk_score=output["risk_score"],
            action=action,
            reason=reason,
        )
        if return_risk and mode in ['eval', 'guard']:
            if ctm_metrics:
                self.audit_logger.log_ctm_metrics(ctm_metrics)
            if scan_metrics:
                self.audit_logger.log_scan_metrics(scan_metrics)
            if self.use_task_validator and 'conflict_results' in dir():
                self.audit_logger.log_conflict_metrics(conflict_results)
        output["audit_log"] = self.audit_logger.get_current_log()
            
        # 额外输出 (用于训练)
        if mode == 'train':
            output["cls_logits"] = cls_logits
            output["seg_mask_logits"] = seg_mask_logits
            output["ctm_metrics"] = ctm_metrics
            output["scan_metrics"] = scan_metrics
            
        return output
    
    def predict(
        self,
        x: torch.Tensor,
        return_heatmaps: bool = True,
    ) -> Dict[str, Any]:
        return self.forward(
            x,
            return_risk=True,
            return_heatmaps=return_heatmaps,
            return_audit=True,
            mode='guard',
        )

    def hard_gating_rules(
        self,
        confidence: float,
        r_state: float,
        r_scan: float,
        r_task: float,
        r_entropy: float = 0.0,
        r_total: float = None,
    ) -> str:
        if r_total is not None:
            if r_total > self.hard_gating.theta_high:
                return 'doctor_review'
            if r_task > self.hard_gating.theta_conflict:
                return 'doctor_review'
            if confidence > self.hard_gating.theta_conf and r_state > self.hard_gating.theta_state:
                return 'overconfidence_warning'
            return 'PASS'
        action, _, _ = self.hard_gating.evaluate(
            r_state=r_state,
            r_scan=r_scan,
            r_task=r_task,
            r_entropy=r_entropy,
            confidence=confidence,
        )
        return action


# =============================================================================
# 工厂函数
# =============================================================================

def create_medmamba_guard(
    d_model: int = 384,
    n_layers: int = 12,
    num_classes: int = 2,
    **kwargs
) -> MedMambaGuard:
    """创建MedMamba-Guard模型"""
    return MedMambaGuard(
        d_model=d_model,
        n_layers=n_layers,
        num_classes=num_classes,
        **kwargs
    )


# =============================================================================
# 模型变体
# =============================================================================

class LightMedMambaGuard(MedMambaGuard):
    """轻量版MedMamba-Guard"""
    
    def __init__(self, **kwargs):
        # 使用更小的配置
        kwargs.setdefault('d_model', 192)
        kwargs.setdefault('n_layers', 6)
        kwargs.setdefault('use_ctm', True)
        kwargs.setdefault('use_cross_scan', True)
        kwargs.setdefault('use_task_validator', True)
        super().__init__(**kwargs)


class FullMedMambaGuard(MedMambaGuard):
    """完整版MedMamba-Guard (更大的模型)"""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('d_model', 512)
        kwargs.setdefault('n_layers', 16)
        kwargs.setdefault('trajectory_window', 7)
        kwargs.setdefault('delta_window', 5)
        super().__init__(**kwargs)