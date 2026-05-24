"""
SSMConfig - SSM模型配置
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SSMConfig:
    """选择性状态空间模型配置"""

    # 模型维度
    d_model: int = 768          # 隐藏层维度
    d_state: int = 16            # SSM状态维度(类似RNN hidden)
    d_conv: int = 4              # 卷积核大小
    expand: int = 2              # 扩展因子

    # SSM参数
    dt_rank: str = "auto"        # 时间步排名(auto或int)
    dt_min: float = 0.001        # 最小时间步
    dt_max: float = 0.1          # 最大时间步
    dt_init: str = "random"      # 初始化: random/constant/zero

    # 正则化
    dropout: float = 0.1
    layer_norm: bool = True
    norm_eps: float = 1e-5

    # 医学影像专用
    img_size: int = 224           # 输入图像大小
    patch_size: int = 16         # Patch大小
    in_channels: int = 3         # 输入通道数

    # 分类头
    num_classes: int = 2         # 二分类(肿瘤/非肿瘤)
    vocab_size: int = 512        # 词表大小

    # 计算选项
    flip_skip_connection: bool = False  # 是否翻转残差
    init: str = "gelu"           # 初始化方式

    # 变体选项
    variant: str = "v0"          # v0=原始Mamba, v1=医学优化

    # 推理优化
    use_cache: bool = True       # KV缓存(用于生成)
    d_state_final: int = 0       # 最终状态输出维度(0=无变换)

    # 训练选项
    mixed_precision: bool = True # 混合精度训练
    compile: bool = False        # torch.compile加速

    # 位置编码
    use_pos_emb: bool = True     # 位置编码类型(none/sin/explicit)
    max_seq_len: int = 4096      # 最大序列长度

    def __post_init__(self):
        if self.dt_rank == "auto":
            self.dt_rank = max(self.d_model // 16, 1)
        if self.variant not in ("v0", "v1", "medical"):
            self.variant = "v0"


@dataclass
class MedMambaConfig(SSMConfig):
    """医学影像专用Mamba配置"""
    # 预训练模型名称(用于加载权重)
    pretrained_name: Optional[str] = None

    # 医学影像特殊配置
    organ_sites: int = 0          # 器官部位数(0=通用)
    modal: str = "CT"            # 成像模态: CT/MRI/X-ray/病理

    # 对比学习
    use_contrastive: bool = False
    projection_dim: int = 256

    # 多标签
    multi_label: bool = False

    # 8类常见病变分类
    num_diseases: int = 8

    # 4通道输入(CT/MRI多期)
    multi_phase: bool = False
    phase_channels: int = 4


@dataclass
class CTMConfig:
    """
    CTM (Credible Trajectory Monitoring) 状态轨迹监控配置
    
    核心参数:
    - trajectory_window: 状态轨迹滑动窗口大小 (计算V_norm)
    - delta_window: Δ_t局部方差窗口 (计算D_Δ)
    - state_risk_threshold: 状态风险阈值
    - scan_risk_threshold: Cross-Scan风险阈值
    - conflict_threshold: 互证冲突阈值
    - hard_gate_high: 总风险硬门控阈值 (>此值需医生审核)
    - overconf_threshold: 过度自信检测阈值
    """
    trajectory_window: int = 5      # 状态轨迹滑动窗口大小
    delta_window: int = 3          # Δ_t局部方差窗口
    state_risk_threshold: float = 0.5  # 状态风险阈值
    scan_risk_threshold: float = 0.3   # Cross-Scan风险阈值
    conflict_threshold: float = 0.4    # 互证冲突阈值
    hard_gate_high: float = 0.7        # 总风险硬门控阈值
    overconf_threshold: float = 0.5    # 过度自信检测阈值
    
    # Cross-Scan融合权重
    lambda_l2: float = 0.5
    lambda_cos: float = 0.5
    
    # 分割证据权重
    alpha_area: float = 0.5
    beta_compactness: float = 0.3
    gamma_boundary: float = 0.2
    
    # 风险热力图权重
    w_ctm: float = 0.6
    w_scan: float = 0.4


@dataclass
class MedMambaGuardConfig(MedMambaConfig, CTMConfig):
    model_version: str = "MedMamba-Guard-v1.0"
    n_layers: int = 12
    use_ctm: bool = True
    use_cross_scan: bool = True
    use_task_validator: bool = True
    theta_conf: float = 0.85
    theta_state: float = 0.5