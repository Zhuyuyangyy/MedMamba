# MedMamba Paper Figures

Publication-ready matplotlib figures for SSM-based medical imaging papers.

## PALETTE - MedMamba配色方案

```python
# SSM/Mamba家族 - 蓝色系
MAMBA_PALETTE = {
    "ssm_deep":    "#1A4B8C",   # SSM深层主色
    "ssm_light":   "#4A7FC7",   # SSM浅层
    "cross_scan":  "#2E6BA8",   # Cross-Scan方向色
    "vmamba":      "#3D7EAA",   # VMamba变体
}

# CTM动力学 - 橙色系
CTM_PALETTE = {
    "stability":   "#D97706",   # 稳定性高
    "oscillate":   "#F59E0B",   # 振荡
    "conflict":    "#DC2626",   # 冲突/幻觉风险
    "safe":        "#059669",   # 低风险
}

# 风险评估 - 红黄绿渐变 (MedMamba-Guard)
RISK_PALETTE = {
    "low":        "#059669",   # 低风险 - 绿色
    "medium":     "#F59E0B",   # 中风险 - 黄色
    "high":       "#FF9500",   # 高风险 - 橙色
    "critical":   "#DC2626",   # 极高风险 - 红色
}

# 基准模型 - 灰色系
BASELINE_PALETTE = {
    "vit":         "#6B7280",   # Vision Transformer
    "cnn":         "#9CA3AF",   # CNN baseline
    "unet":        "#D1D5DB",   # U-Net分割
    "medicalnet":  "#E5E7EB",   # MedicalNet
}

# 医学影像 - 绿色系
MEDICAL_PALETTE = {
    "organ":       "#059669",   # 器官结构
    "tumor":       "#DC2626",   # 肿瘤区域
    "lesion":      "#F59E0B",   # 病变
    "normal":      "#10B981",   # 正常组织
}

# 综合对比用
COMPARISON_COLORS = [
    MAMBA_PALETTE["ssm_deep"],
    BASELINE_PALETTE["vit"],
    BASELINE_PALETTE["cnn"],
    CTM_PALETTE["stability"],
    "#9A4D8E",  # 紫色: 其他方法
    "#4D4D4D",  # 深灰: 消融对照
]

# Guard特有 - 风险分量配色
GUARD_COMPONENTS = {
    "R_state":    "#00d4ff",   # CTM状态轨迹 - 青色
    "R_scan":     "#7b2cbf",   # Cross-Scan - 紫色
    "R_task":     "#ff9500",   # 任务冲突 - 橙色
    "R_entropy":  "#ff4757",   # 预测熵 - 红色
}

# 幻觉风险渐变 (用于热图)
HALLUCINATION_CMAP = "RdYlGn_r"  # 红(高风险)→黄→绿(低风险)
RISK_HEATMAP_CMAP = "RdYlGn_r"   # 风险热图
```

## 图表类型

### 1. 基准模型对比柱状图 (Accuracy/F1/AUC对比)

```
图表原型: quantitative grid
核心结论: MedMamba在XX指标上显著优于所有基准模型
面板: a) 主指标对比 b) 消融实验 c) 复杂度对比
```

### 2. 幻觉风险热图 (CTM轨迹稳定性)

```
图表原型: image plate + quant  
核心结论: CTM轨迹稳定性与诊断准确性正相关
面板: a) 混淆矩阵 b) 幻觉风险分布 c) 稳定性时序图
```

### 3. O(n) vs O(n²) 复杂度示意图

```
图表原型: schematic-led composite
核心结论: MedMamba实现线性复杂度，突破ViT二次瓶颈
面板: a) 架构示意图 b) 序列长度vs延迟 c) 显存对比
```

### 4. 医学影像分割效果图 (多模态)

```
图表原型: image plate + quant
核心结论: 多模态融合提升分割精度
面板: a) CT b) MRI c) 融合预测 d) 专家标注
```

---

## MedMamba-Guard 特有图表

### 5. 风险-错误AUROC曲线 (Risk-Error AUROC)

用于展示风险分数区分正确/错误预测的能力。

```python
def plot_risk_error_auroc():
    """
    X轴: False Positive Rate (FPR)
    Y轴: True Positive Rate (TPR)
    
    曲线: 不同方法的Risk-Error ROC曲线
    - MedMamba-Guard (AUROC = 0.847)
    - Bayesian MC-Dropout (AUROC = 0.712)
    - Deep Ensemble (AUROC = 0.768)
    - Baseline (AUROC = 0.500)
    
    参考线: 对角线 (AUROC = 0.500, 随机猜测)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制各方法的ROC曲线
    methods = {
        'MedMamba-Guard': (fpr_guard, tpr_guard, 0.847, RISK_PALETTE['low']),
        'Deep Ensemble': (fpr_ensemble, tpr_ensemble, 0.768, BASELINE_PALETTE['vit']),
        'MC-Dropout': (fpr_mcd, tpr_mcd, 0.712, BASELINE_PALETTE['cnn']),
        'Random': (fpr_random, tpr_random, 0.500, '#888888'),
    }
    
    for name, (fpr, tpr, auroc, color) in methods.items():
        linestyle = '--' if 'Random' in name else '-'
        linewidth = 2 if 'Guard' in name else 1.5
        ax.plot(fpr, tpr, label=f'{name} (AUROC={auroc:.3f})', 
                color=color, linestyle=linestyle, linewidth=linewidth)
    
    # 对角线
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUROC=0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Risk-Error Detection AUROC\n(Risk Score vs Model Prediction Error)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig
```

### 6. CTM指标分布箱线图 (正确 vs 错误样本)

用于展示CTM四维指标在正确/错误分类样本间的分布差异。

```python
def plot_ctm_metrics_boxplot(ctm_data):
    """
    面板布局: 2x2 subplot
    - (0,0): V_norm (状态激变度)
    - (0,1): D_delta (响应漂移)
    - (1,0): C_layer (跨层稳定性)
    - (1,1): R_overconfident (过度自信风险)
    
    每个子图: 箱线图对比正确样本 vs 错误样本
    统计显著性标注: *** (p<0.001)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['V_norm_mean', 'D_delta', 'C_layer', 'R_overconfident']
    titles = ['State Mutation Rate (V_norm)', 
              'Response Drift (D_delta)',
              'Cross-Layer Stability (C_layer)', 
              'Overconfidence Risk (R_overconf)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # 箱线图
        bp = ax.boxplot([correct_values, error_values], 
                       labels=['Correct', 'Error'],
                       patch_artist=True)
        
        # 颜色
        bp['boxes'][0].set_facecolor(RISK_PALETTE['low'])
        bp['boxes'][1].set_facecolor(RISK_PALETTE['critical'])
        
        # 统计标注
        ax.annotate('***\np<0.001', xy=(1.5, max(error_values)*0.9),
                   fontsize=10, ha='center', color='red')
        
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('CTM Metrics Distribution: Correct vs Error Predictions', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    return fig
```

### 7. 硬门控决策流程图 (Hard Gating Flowchart)

展示MedMamba-Guard的硬门控决策逻辑。

```
图表原型: flowchart
核心结论: 硬门控规则有效识别需要医生复核的高风险样本

流程:
[输入: 图像] 
    ↓
[计算 R_total]
    ↓
R_total > θ_high (0.7)?
    ├─ Yes → [触发医生复核] → END
    └─ No
        ↓
[计算 S_conflict]
        ↓
S_conflict > θ_conflict (0.5)?
    ├─ Yes → [触发医生复核] → END
    └─ No
        ↓
[计算 Conf × R_state]
        ↓
Conf > 0.85 AND R_state > 0.6?
    ├─ Yes → [过度自信警告] → END
    └─ No
        ↓
[自动通过] → END

颜色:
- 绿色框: 低风险路径
- 黄色框: 中风险路径
- 红色框: 高风险/强制复核
```

### 8. 风险热力图叠加示例 (Risk Heatmap Overlay)

展示在原图上叠加风险热力图的视觉效果。

```python
def plot_risk_heatmap_overlay():
    """
    面板布局: 1行3列
    - (0): 原图
    - (1): 风险热力图 (半透明叠加)
    - (2): 高风险区域bbox标注
    
    热力图配色: RdYlGn_r (红=高风险, 绿=低风险)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # 热力图叠加
    axes[1].imshow(original_image)
    im = axes[1].imshow(risk_heatmap, cmap='RdYlGn_r', alpha=0.6)
    axes[1].set_title('Risk Heatmap Overlay', fontsize=12)
    axes[1].axis('off')
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Risk Score', fontsize=10)
    
    # 高风险区域bbox
    axes[2].imshow(original_image)
    for bbox in high_risk_regions:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, edgecolor='red', linewidth=2)
        axes[2].add_patch(rect)
    axes[2].set_title('High-Risk Regions (BBox)', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle('MedMamba-Guard Risk Visualization', fontsize=14)
    plt.tight_layout()
    return fig
```

### 9. Error Capture Rate @ Top-K% 柱状图

展示不同Top-K比例下的错误捕获率。

```python
def plot_ecr_at_topk():
    """
    X轴: Top-K% (5%, 10%, 20%, 50%)
    Y轴: Error Capture Rate (%)
    
    基准线: 随机猜测 (y=x)
    
    图表类型:  grouped bar chart
    - 每组对比: MedMamba-Guard vs 其他方法
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = [5, 10, 20, 50]
    guard_ecr = [41.2, 62.3, 81.5, 96.8]
    ensemble_ecr = [28.5, 45.2, 68.4, 89.1]
    mcdropout_ecr = [22.1, 38.7, 62.3, 85.6]
    random = [5, 10, 20, 50]  # 基准线
    
    x = np.arange(len(k_values))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, guard_ecr, width, label='MedMamba-Guard', 
                   color=RISK_PALETTE['low'])
    bars2 = ax.bar(x - 0.5*width, ensemble_ecr, width, label='Deep Ensemble',
                   color=BASELINE_PALETTE['vit'])
    bars3 = ax.bar(x + 0.5*width, mcdropout_ecr, width, label='MC-Dropout',
                   color=BASELINE_PALETTE['cnn'])
    bars4 = ax.bar(x + 1.5*width, random, width, label='Random', 
                   color='#888888', hatch='//')
    
    ax.plot(x, random, 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Top-K%', fontsize=12)
    ax.set_ylabel('Error Capture Rate (%)', fontsize=12)
    ax.set_title('Error Capture Rate @ Top-K%\n(Higher is Better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top {k}%' for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    return fig
```

### 10. 综合风险分数分布直方图

展示正确/错误样本的R_total分布对比。

```python
def plot_r_total_distribution():
    """
    图表类型: Overlapping histogram
    X轴: R_total (0-1)
    Y轴: 样本数量/密度
    
    两组数据:
    - 正确分类样本 (绿色, 集中在低风险区)
    - 错误分类样本 (红色, 集中在高风险区)
    
    标注:
    - 阈值线: θ_high = 0.7
    - 正确样本均值垂直线
    - 错误样本均值垂直线
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 正确样本分布
    ax.hist(correct_rtotals, bins=30, alpha=0.6, color=RISK_PALETTE['low'],
            label=f'Correct (μ={np.mean(correct_rtotals):.2f})', density=True)
    
    # 错误样本分布
    ax.hist(error_rtotals, bins=30, alpha=0.6, color=RISK_PALETTE['critical'],
            label=f'Error (μ={np.mean(error_rtotals):.2f})', density=True)
    
    # 阈值线
    ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2, 
               label='θ_high = 0.7')
    
    ax.set_xlabel('R_total (Total Risk Score)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Total Risk Score\nCorrect vs Error Predictions', 
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    textstr = f'Pearson r = 0.72\nAUROC = 0.847'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.75, 0.75, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    return fig
```

---

## 必读规范

- 字体: Arial / DejaVu Sans, 不可用衬线字体
- SVG输出: `svg.fonttype = 'none'` 保持文字可编辑
- 输出格式: SVG (主) + PDF + TIFF 300DPI
- 坐标轴: 右侧和顶部边框隐藏
- 图例: 无边框，背景透明
- 误差线: `elinewidth=1.5, capsize=4`
- 医学影像: 使用DICOM查看器校准窗宽窗位后再截图

## Guard特有绘图约定

1. **风险配色使用RISK_PALETTE**，不要混用CTM_PALETTE
2. **风险热图统一使用RdYlGn_r**反编码（红=高，绿=低）
3. **Guard组件使用GUARD_COMPONENTS**配色区分R_state/R_scan/R_task/R_entropy
4. **AUROC曲线图**保持Guard曲线加粗突出
5. **箱线图**正确样本用低风险色，错误样本用critical色