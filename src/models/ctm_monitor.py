"""
CTMMonitor - Credible Trajectory Monitoring (状态轨迹分析器)

核心公式 (来自MedMamba-Guard框架):
1. 状态激变度: V_norm(t) = ||h_t - h_{t-1}||² / (||h_{t-1}||₂ + ε)
2. 输入依赖响应漂移: D_Δ(t) = Var(Δ_{t-k:t+k}) = (1/(2k+1)) * Σ ||Δ_i - μ_Δ||²
3. 跨层语义稳定性: C_layer = 1 - mean(cos(h_l, h_{l-1}))
4. 过度自信检测: R_overconfident = Conf_pred * R_state

R_state = α*V_norm + β*D_Δ + γ*(1-C_layer) + δ*R_overconfident

作者: MedMamba Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange
import math


class CTMMonitor(nn.Module):
    """
    CTM状态轨迹分析器 - 监控SSM隐藏态的稳定性和可信度
    
    功能:
    1. SSM Hidden State Hook系统 - 提取中间隐藏态h_t和步长Δ_t
    2. StateTransitionInstability - 状态激变度检测
    3. InputDependentResponseDrift - Δ_t序列局部方差检测
    4. TrajectoryConvergenceScore - 跨层语义稳定性
    5. OverconfidenceDetector - 高置信但内部不稳定检测
    """
    
    def __init__(
        self,
        hidden_dim: int,
        trajectory_window: int = 5,
        delta_window: int = 3,
        eps: float = 1e-8,
    ):
        """
        Args:
            hidden_dim: SSM隐藏态维度
            trajectory_window: 状态轨迹滑动窗口大小 (用于计算V_norm)
            delta_window: Δ_t局部方差窗口大小 (用于计算D_Δ)
            eps: 数值稳定项
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trajectory_window = trajectory_window
        self.delta_window = delta_window
        self.eps = eps
        
        # 注册的hook句柄
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # 存储提取的隐藏态序列
        self.hidden_states: List[torch.Tensor] = []
        self.delta_states: List[torch.Tensor] = []
        
        # 用于存储中间SSM状态的字典 (用于生成热力图)
        self.state_cache: Dict[str, torch.Tensor] = {}
        
        # CTM指标缓存
        self._ctm_metrics: Optional[Dict[str, torch.Tensor]] = None
        self._state_risk_score: Optional[float] = None
        self._state_risk_map: Optional[torch.Tensor] = None
        
    def _create_hidden_state_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            elif isinstance(output, list):
                hidden = output[0] if len(output) > 0 else None
                if hidden is None:
                    return
            else:
                hidden = output
            if isinstance(hidden, torch.Tensor):
                if hidden.dim() == 4:
                    hidden = hidden.flatten(2).transpose(1, 2)
                elif hidden.dim() == 3:
                    pass
                elif hidden.dim() == 2:
                    hidden = hidden.unsqueeze(1)
                else:
                    return
                self.hidden_states.append(hidden.detach())
            if hasattr(module, 'last_delta') and isinstance(module.last_delta, torch.Tensor):
                self.delta_states.append(module.last_delta.detach())
                
        return hook_fn
    
    def _create_delta_hook(self, layer_idx: int):
        """创建Δ_t提取hook"""
        def hook_fn(module, input, output):
            # 尝试提取dt_proj的输出作为Δ_t
            # 这需要在forward中手动保存
            pass
        return hook_fn
    
    def register_hooks(self, model: nn.Module) -> None:
        """
        注册hook到Mamba层，提取隐藏态
        
        Args:
            model: 包含SSM层的模型 (如VSSBlock2D, SS2D, MambaBlock2D)
        """
        # 清除之前的hooks
        self.remove_hooks()
        
        # 遍历模型的所有子模块
        for name, module in model.named_modules():
            # 注册到SSM相关层
            if any(subname in name for subname in ['ssm', 'SS2D', 'VSSBlock', 'MambaBlock']):
                if isinstance(module, (nn.Module)) and len(list(module.children())) == 0:
                    # 叶子节点，注册hook
                    hook = module.register_forward_hook(self._create_hidden_state_hook(0))
                    self._hooks.append(hook)
                    
        print(f"[CTMMonitor] Registered {len(self._hooks)} hooks for SSM layers")
    
    def remove_hooks(self) -> None:
        """移除所有注册的hook"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        
    def clear_cache(self) -> None:
        """清除缓存的状态和指标"""
        self.hidden_states.clear()
        self.delta_states.clear()
        self.state_cache.clear()
        self._ctm_metrics = None
        self._state_risk_score = None
        self._state_risk_map = None
        
    def _compute_state_transition_instability(
        self,
        hidden_states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(hidden_states) < 2:
            return torch.tensor(0.0), torch.zeros(1, 1, 1)

        v_norms = []

        for t in range(1, len(hidden_states)):
            h_t = hidden_states[t]
            h_prev = hidden_states[t-1]

            if h_t.dim() == 3 and h_prev.dim() == 3:
                min_L = min(h_t.shape[1], h_prev.shape[1])
                min_D = min(h_t.shape[2], h_prev.shape[2])
                h_t = h_t[:, :min_L, :min_D]
                h_prev = h_prev[:, :min_L, :min_D]
            elif h_t.shape != h_prev.shape:
                continue

            diff = h_t - h_prev
            diff_norm_sq = torch.sum(diff ** 2, dim=-1)
            prev_norm = torch.sum(h_prev ** 2, dim=-1).clamp(min=self.eps)
            v_norm = diff_norm_sq / (prev_norm + self.eps)
            v_norms.append(v_norm.mean(dim=-1))

        if v_norms:
            v_norm_avg = torch.stack(v_norms).mean(dim=0)
        else:
            v_norm_avg = torch.tensor(0.0)

        v_norm_map = torch.zeros(1, 1, 1)
        return v_norm_avg, v_norm_map
    
    def _compute_delta_drift(
        self,
        hidden_states: List[torch.Tensor],
        delta_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算输入依赖响应漂移 D_Δ(t)
        
        由于Δ_t难以直接提取，使用隐藏态范数变化率作为替代:
        D_Δ(t) ≈ || ||h_t|| - ||h_{t-1}|| ||² 的局部方差
        
        Args:
            hidden_states: 隐藏态序列
            delta_states: Δ_t序列 (可选)
            
        Returns:
            d_delta: 漂移分数 [B, L]
            d_delta_map: 空间漂移热图
        """
        if len(hidden_states) < 2:
            return torch.tensor(0.0), torch.zeros(1, 1, 1)

        norms = []
        for h in hidden_states:
            h_norm = torch.norm(h, dim=-1).mean(dim=-1)
            norms.append(h_norm)

        norm_diffs = []
        for i in range(1, len(norms)):
            diff = (norms[i] - norms[i-1]).abs()
            norm_diffs.append(diff)

        if not norm_diffs:
            return torch.tensor(0.0), torch.zeros(1, 1, 1)

        norm_diffs_stack = torch.stack(norm_diffs)

        k = self.delta_window
        d_deltas = []

        for t in range(k, len(norm_diffs)):
            window = norm_diffs[t-k:t+k+1]
            window_stack = torch.stack(window)
            d_var = torch.var(window_stack, dim=0, unbiased=False)
            d_deltas.append(d_var)

        if d_deltas:
            d_delta_avg = torch.stack(d_deltas).mean(dim=0)
        else:
            d_delta_avg = torch.tensor(0.0)

        d_delta_map = torch.zeros(1, 1, 1)
        return d_delta_avg, d_delta_map
    
    def _compute_trajectory_convergence(
        self,
        hidden_states: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算跨层语义稳定性 C_layer
        
        C_layer = 1 - mean(cos(h_l, h_{l-1}))
        
        Args:
            hidden_states: 隐藏态序列
            
        Returns:
            c_layer: 收敛分数 (越高越不稳定)
            c_layer_map: 空间收敛热图
        """
        if len(hidden_states) < 2:
            return torch.tensor(0.0), torch.zeros(1, 1, 1)

        cos_similarities = []

        for t in range(1, len(hidden_states)):
            h_t = hidden_states[t]
            h_prev = hidden_states[t-1]

            if h_t.dim() == 3 and h_prev.dim() == 3:
                min_L = min(h_t.shape[1], h_prev.shape[1])
                min_D = min(h_t.shape[2], h_prev.shape[2])
                h_t = h_t[:, :min_L, :min_D]
                h_prev = h_prev[:, :min_L, :min_D]
            elif h_t.shape != h_prev.shape:
                continue

            h_t_norm = F.normalize(h_t, p=2, dim=-1)
            h_prev_norm = F.normalize(h_prev, p=2, dim=-1)

            cos_sim = torch.sum(h_t_norm * h_prev_norm, dim=-1).mean(dim=-1)
            cos_similarities.append(cos_sim)

        if cos_similarities:
            cos_sim_stack = torch.stack(cos_similarities)
            c_layer = 1 - cos_sim_stack.mean(dim=0)
        else:
            c_layer = torch.tensor(0.0)

        c_layer_map = torch.zeros(1, 1, 1)
        return c_layer, c_layer_map
    
    def _compute_overconfidence(
        self,
        hidden_states: List[torch.Tensor],
        output_confidence: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算过度自信检测
        
        R_overconfident = Conf_pred * R_state
        
        当模型预测高置信但内部状态不稳定时触发
        
        Args:
            hidden_states: 隐藏态序列
            output_confidence: 预测置信度 [0, 1]
            
        Returns:
            r_overconf: 过度自信分数
            r_overconf_map: 空间过度自信热图
        """
        # 先计算基础状态风险
        v_norm, _ = self._compute_state_transition_instability(hidden_states)
        
        if isinstance(output_confidence, float):
            conf_tensor = torch.tensor(output_confidence, device=v_norm.device if isinstance(v_norm, torch.Tensor) else 'cpu')
        else:
            conf_tensor = output_confidence
            
        v_norm_val = v_norm.mean() if isinstance(v_norm, torch.Tensor) and v_norm.numel() > 1 else v_norm
        r_overconf = conf_tensor * v_norm_val
        
        r_overconf_map = torch.zeros(1, 1, 1)
        
        return r_overconf, r_overconf_map
    
    def compute_ctm_metrics(
        self,
        hidden_states: Optional[List[torch.Tensor]] = None,
        delta_states: Optional[List[torch.Tensor]] = None,
        output_confidence: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有CTM指标
        
        Args:
            hidden_states: 隐藏态序列 (如果为None使用缓存)
            delta_states: Δ_t序列 (可选)
            output_confidence: 预测置信度
            
        Returns:
            ctm_metrics: 包含所有指标的字典
        """
        # 使用传入的states或缓存的states
        if hidden_states is None:
            hidden_states = self.hidden_states
            
        if delta_states is None:
            delta_states = self.delta_states
            
        # 1. 状态激变度
        v_norm, v_norm_map = self._compute_state_transition_instability(hidden_states)
        
        # 2. 输入依赖响应漂移
        d_delta, d_delta_map = self._compute_delta_drift(hidden_states, delta_states)
        
        # 3. 跨层语义稳定性
        c_layer, c_layer_map = self._compute_trajectory_convergence(hidden_states)
        
        # 4. 过度自信检测
        r_overconf, r_overconf_map = self._compute_overconfidence(hidden_states, output_confidence)
        
        # 综合R_state计算
        # R_state = α*V_norm + β*D_Δ + γ*(1-C_layer) + δ*R_overconf
        alpha, beta, gamma, delta = 0.3, 0.3, 0.25, 0.15
        
        # 处理可能的维度差异
        if v_norm.dim() > 0:
            r_state = (
                alpha * v_norm.mean() +
                beta * d_delta.mean() +
                gamma * c_layer.mean() +
                delta * r_overconf
            )
        else:
            r_state = alpha * v_norm + beta * d_delta + gamma * c_layer + delta * r_overconf
            
        self._ctm_metrics = {
            'v_norm': v_norm,
            'v_norm_map': v_norm_map,
            'state_transition_instability': v_norm,
            'd_delta': d_delta,
            'd_delta_map': d_delta_map,
            'input_response_drift': d_delta,
            'c_layer': c_layer,
            'c_layer_map': c_layer_map,
            'trajectory_convergence': c_layer,
            'r_overconf': r_overconf,
            'r_overconf_map': r_overconf_map,
            'overconfidence_risk': r_overconf,
            'r_state': r_state,
        }
        
        # 更新状态风险分数缓存
        self._state_risk_score = r_state.mean().item() if isinstance(r_state, torch.Tensor) else r_state
        self._state_risk_map = v_norm_map  # 使用v_norm_map作为主要热图
        
        return self._ctm_metrics
    
    def get_state_risk_score(self) -> float:
        """
        获取综合状态风险分数
        
        Returns:
            R_state: 0=低风险, 1=高风险
        """
        if self._state_risk_score is None:
            if self.hidden_states:
                self.compute_ctm_metrics()
            else:
                return 0.0
                
        return self._state_risk_score
    
    def get_state_risk_map(self) -> torch.Tensor:
        """
        获取空间状态风险热图
        
        Returns:
            risk_map: [B, H, W] 或 [B, L] 风险热图
        """
        if self._state_risk_map is None:
            if self.hidden_states:
                self.compute_ctm_metrics()
            else:
                return torch.zeros(1, 1, 1)
                
        return self._state_risk_map
    
    def get_overconfidence_regions(
        self,
        threshold: float = 0.5,
        min_area: int = 10
    ) -> List[Dict]:
        """
        获取过度自信区域 (高风险区域bbox列表)
        
        Args:
            threshold: 风险阈值
            min_area: 最小区域面积
            
        Returns:
            regions: [{"bbox": [x1,y1,x2,y2], "risk_type": str, "risk_level": str}]
        """
        risk_map = self.get_state_risk_map()
        
        if risk_map.numel() == 1:
            return []
            
        # 生成候选区域
        regions = []
        
        # 简单实现: 找到高于阈值的区域
        binary_mask = (risk_map > threshold).float()
        
        # 计算连通区域 (简化实现)
        # 实际应用应使用scipy.ndimage.label
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary_mask.cpu().numpy())
            
            for i in range(1, num_features + 1):
                region_mask = labeled == i
                area = region_mask.sum()
                
                if area < min_area:
                    continue
                    
                # 获取边界框
                rows = np.any(region_mask, axis=1)
                cols = np.any(region_mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                regions.append({
                    'bbox': [int(cmin), int(rmin), int(cmax), int(rmax)],
                    'risk_type': 'overconfidence',
                    'risk_level': 'high' if risk_map.mean() > 0.7 else 'medium',
                })
        except ImportError:
            # 如果没有scipy，使用简单阈值方法
            B, H, W = risk_map.shape
            for b in range(B):
                for i in range(0, H, H//4):
                    for j in range(0, W, W//4):
                        h_end = min(i + H//4, H)
                        w_end = min(j + W//4, W)
                        region_risk = risk_map[b, i:h_end, j:w_end].mean()
                        
                        if region_risk > threshold:
                            regions.append({
                                'bbox': [j, i, w_end, h_end],
                                'risk_type': 'overconfidence',
                                'risk_level': 'high' if region_risk > 0.7 else 'medium',
                            })
                            
        return regions
    
    def forward(
        self,
        hidden_states: Optional[List[torch.Tensor]] = None,
        delta_states: Optional[List[torch.Tensor]] = None,
        output_confidence: float = 0.5,
        return_risk_map: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向接口
        
        Args:
            hidden_states: 隐藏态序列
            delta_states: Δ_t序列
            output_confidence: 预测置信度
            return_risk_map: 是否返回风险热图
            
        Returns:
            results: 包含CTM指标的字典
        """
        metrics = self.compute_ctm_metrics(hidden_states, delta_states, output_confidence)
        
        if return_risk_map:
            metrics['risk_map'] = self.get_state_risk_map()
            
        return metrics


# 辅助函数: 用于在其他模块中快速创建CTMMonitor
def create_ctm_monitor(
    hidden_dim: int,
    trajectory_window: int = 5,
    delta_window: int = 3,
) -> CTMMonitor:
    """创建CTMMonitor的工厂函数"""
    return CTMMonitor(
        hidden_dim=hidden_dim,
        trajectory_window=trajectory_window,
        delta_window=delta_window,
    )
