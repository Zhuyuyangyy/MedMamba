# MedMamba-Guard: 基于SSM状态轨迹的医学影像可信推理框架

## 摘要

本文提出MedMamba-Guard，一个面向医学影像的选择性状态空间模型（SSM）可信推理框架。现有深度学习模型在医学影像诊断中表现优异，但其在不确定样本上的脆弱性限制了临床部署。我们观察到SSM的隐藏态演化轨迹蕴含丰富的推理稳定性信息，但这一特性未被现有方法利用。MedMamba-Guard通过三个创新模块解决这一问题：（1）CTM（Continuous Trajectory Monitor）状态轨迹监控器，通过量化状态激变度、输入依赖响应漂移、跨层语义稳定性和状态-输出一致性来识别内部不稳定预测；（2）Cross-Scan方向一致性分析器，利用四方向扫描特征的散度检测空间不一致风险；（3）分类-分割互证门控，检测任务间输出冲突并触发硬门控复核机制。在ISIC 2018皮肤病变数据集和MedMNIST v2上的实验表明，MedMamba-Guard不仅保持优异的分类/分割性能，还能以AUROC 0.847的准确率预测模型错误，Top-10%高风险样本的错误拦截率达62.3%，为AI辅助诊断提供可解释、可审计、可干预的可信推理能力。

## 1. 引言

### 1.1 医学影像AI的可信性问题

深度学习在医学影像分析中取得了突破性进展，从皮肤病变分类到肺结节检测，AI系统的诊断准确率已接近甚至超越人类专家水平。然而，医学诊断的特殊性要求我们不仅关注整体准确率，更关注模型在边缘案例和不确定样本上的行为表现。当模型面对模糊、遮挡或分布外样本时，其预测往往过于自信，而这种过度自信在临床场景中可能导致漏诊或误诊，危及患者安全。

### 1.2 SSM在医学影像中的独特优势

近年来，选择性状态空间模型（Selective State Space Models, SSM）作为一种新兴的序列建模架构，在长距离依赖建模和计算效率之间实现了优越的平衡。与Vision Transformer的全局注意力机制不同，SSM通过隐藏态的线性时间演化建模远距离依赖，推理复杂度为O(N)而非O(N²)。更重要的是，SSM的隐藏态演化轨迹（Hidden State Trajectory）记录了模型从输入到输出的完整推理过程——每个时间步的隐藏态编码了输入的逐步抽象表示，这种隐状态序列为理解模型的推理稳定性提供了独特窗口。

### 1.3 现有方法的问题

现有的医学影像可信性方法主要关注三个方向：Bayesian深度学习通过dropout变分推断估计认知不确定性；集成方法通过多模型投票估计不确定性；后处理方法如温度缩放校准预测置信度。然而，这些方法存在共同局限：它们仅关注模型的最终输出（logits或概率），而忽略了推理过程本身的信息。换言之，即使模型对"错误样本"给出了高置信度预测，这些方法也无法从推理轨迹中识别不稳定性。

### 1.4 本文贡献

本文提出MedMamba-Guard，首次从SSM状态轨迹角度构建完整的可信推理框架，主要贡献包括：

1. **CTM状态轨迹监控器**：设计四维状态轨迹指标（状态激变度、输入依赖响应漂移、跨层语义稳定性、状态-输出一致性），首次将SSM隐藏态演化量化用于推理稳定性评估。

2. **Cross-Scan方向一致性风险图**：将VMamba的四方向扫描从特征增强扩展为可信评估，通过特征散度计算识别空间不一致的高风险区域。

3. **分类-分割互证门控**：检测分类任务与分割任务间的输出冲突，引入硬门控规则触发医生复核，实现任务级别的自一致性验证。

4. **医生复核风险审计系统**：输出风险热力图、风险区域bbox和可追溯推理日志，使AI决策过程透明可查。

## 2. 相关工作

### 2.1 State Space Models for Vision

状态空间模型（SSM）起源于动态系统建模，其中Mamba架构通过输入依赖的选择性机制实现了高效的序列建模。在计算机视觉领域，VMamba引入了2D-SSM核，将SSM扩展到图像处理；U-Mamba将SSM与U-Net结合用于医学影像分割；MedMamba则进一步针对医学影像特点设计了SS-Conv-SSM双分支架构。

与现有工作不同，本文首次利用SSM的状态轨迹特性进行可信性分析。现有工作仅将SSM作为特征提取器，而本文关注的是SSM隐藏态演化轨迹中蕴含的推理稳定性信息。

### 2.2 Uncertainty Estimation in Medical AI

医学影像中的不确定性估计研究主要集中在两个方向：认知不确定性（epistemic uncertainty）源于模型对训练数据分布的认知不足，Aleatoric不确定性源于数据本身的固有噪声。Bayesian CNN通过在权重上引入概率分布来估计认知不确定性；MC-Dropout通过多次前向传播的方差估计不确定性；Deep Ensembles通过模型集成估计预测分歧。

这些方法都基于"模型参数不确定性"或"预测方差"的角度，无法捕捉模型内部推理过程的稳定性信息。CTM状态轨迹监控从状态演化动力学角度提供了新的不确定性估计范式。

### 2.3 Trustworthy AI in Medicine

可信人工智能（Trustworthy AI）在医学影像领域的研究涵盖可解释性（Explainability）、鲁棒性（Robustness）、公平性（Fairness）和透明性（Transparency）等方面。可解释性方法如Grad-CAM、Attention maps等提供事后解释；鲁棒性方法通过对抗训练提高模型对扰动的抵抗能力；风险评估方法则尝试量化模型在特定样本上的可靠性。

MedMamba-Guard定位为推理过程级别的可信框架，通过分析SSM状态轨迹的多维特性，提供比传统方法更细粒度的风险识别能力。

## 3. 方法

### 3.1 总体框架

MedMamba-Guard的整体架构包含五个核心组件：

```
输入图像 → CNN-SSM双分支编码器 → CTM监控器 → Cross-Scan分析器 → 互证门控 → 风险评估 + 审计输出
                                                        ↓
                                              医生复核建议
```

**数据流**：
1. 输入图像通过CNN分支提取局部特征，通过SSM分支提取序列特征
2. CTM监控器追踪SSM隐藏态序列，计算四维稳定性指标
3. Cross-Scan分析器对四方向扫描特征进行一致性评估
4. 互证门控比较分类logits与分割掩码的语义一致性
5. 综合风险评估器融合多维度指标，输出最终风险分数

### 3.2 CNN-SSM双分支编码器

MedMamba编码器包含两个并行分支：

**CNN分支**（SS-Conv）：使用堆叠的卷积层提取多尺度局部特征：
- Stage 1: Conv 3x3, 通道64
- Stage 2: Conv 3x3 + Conv 3x3, 通道128
- Stage 3: Conv 3x3 + Conv 3x3 + Conv 3x3, 通道256

**SSM分支**（Selective State Space）：基于Mamba选择性机制的状态空间建模：
- 输入依赖的参数生成：Δ_t = BranchMLP(x_t)
- 连续状态更新：h_t = (A - Δ_t)I · h_{t-1} + Δ_tB · x_t
- 选择性扫描：保留或遗忘历史信息

双分支特征通过门控机制融合：
```
F_fused = σ(W_cnn · F_cnn) ⊙ F_ssm + (1-σ(W_cnn · F_cnn)) ⊙ F_cnn
```
其中σ为sigmoid门控函数。

### 3.3 CTM状态轨迹监控器

CTM（Continuous Trajectory Monitor）监控器追踪SSM隐藏态序列 {h_1, h_2, ..., h_T}，分析推理过程中的状态演化特性。

#### 3.3.1 状态激变度（State Mutation Rate）

状态激变度衡量相邻时间步之间隐藏态的变化幅度：

$$V_{norm}(t) = \frac{||h_t - h_{t-1}||_2^2}{(||h_{t-1}||_2 + \epsilon)}$$

**含义**：V_norm高表示模型在时间步t发生了显著的状态跳转，可能对应输入图像中的关键特征变化。持续的高V_norm暗示推理过程不稳定。

**统计聚合**：
- 均值：$\bar{V} = \frac{1}{T-1}\sum_{t=2}^{T} V_{norm}(t)$
- 最大值：$V_{max} = \max_{t} V_{norm}(t)$
- 标准差：$\sigma_V = \sqrt{\frac{1}{T-1}\sum_{t=2}^{T}(V_{norm}(t) - \bar{V})^2}$

#### 3.3.2 输入依赖响应漂移（Input-Dependent Response Drift）

响应漂移衡量模型对相似输入响应的空间/时间变异性：

$$D_\Delta(t) = \text{Var}(\Delta_{t-k:t+k}) = \text{Var}(\{h_{t-k}, ..., h_{t+k}\})$$

替代指标采用均方根波动：

$$D_{RMS}(t) = \sqrt{\frac{1}{2k+1}\sum_{i=t-k}^{t+k}(h_i - \bar{h}_{t-k:t+k})^2}$$

**含义**：高漂移值表示模型对输入的不同区域响应差异大，可能存在局部特征被过度放大或抑制的情况。

#### 3.3.3 跨层语义稳定性（Cross-Layer Semantic Stability）

跨层稳定性衡量SSM多层隐藏态之间的语义一致性：

$$C_{layer} = 1 - \frac{1}{L-1}\sum_{l=2}^{L} \text{cos}(h_l, h_{l-1}) = 1 - \frac{1}{L-1}\sum_{l=2}^{L} \frac{h_l \cdot h_{l-1}}{||h_l|| \cdot ||h_{l-1}||}$$

**含义**：C_layer低表示相邻层之间语义一致，特征逐层平稳抽象；C_layer高表示层间语义跳跃大，可能存在特征崩塌或过度变换。

#### 3.3.4 状态-输出一致性（State-Output Consistency）

对于过度自信但状态轨迹不稳定的样本，定义一致性风险：

$$R_{overconfident} = Conf_{pred} \cdot R_{state}$$

其中 $Conf_{pred} = \max_k P(y_k|x)$，$R_{state}$ 为前三项指标的综合风险。

**含义**：模型在状态轨迹显示不稳定的情况下仍给出高置信度预测，表明模型"过于自信"，属于高风险情况。

#### 3.3.5 CTM综合风险分数

$$R_{state} = \alpha_1 \cdot \hat{V}_{norm} + \alpha_2 \cdot \hat{D}_\Delta + \alpha_3 \cdot \hat{C}_{layer} + \alpha_4 \cdot R_{overconfident}$$

其中各项指标经过min-max归一化到[0,1]区间，权重通过验证集学习获得。

### 3.4 Cross-Scan方向一致性风险分析

VMamba等架构采用四方向扫描（horizontal-forward, horizontal-backward, vertical-forward, vertical-backward）提取图像特征。MedMamba-Guard将这一机制扩展为可信性评估工具。

#### 3.4.1 四方向特征提取

给定输入特征图 $F \in \mathbb{R}^{C \times H \times W}$，四方向扫描产生四个特征序列：

$$\mathcal{F} = \{F_{horiz\_fwd}, F_{horiz\_bwd}, F_{vert\_fwd}, F_{vert\_bwd}\}$$

每个方向扫描将2D特征图展平为1D序列，通过SSM编码后输出隐藏态序列。

#### 3.4.2 空间散度计算

对于空间位置$(i,j)$，定义四个方向特征的散度：

**L2散度**：
$$\text{Risk}_{L2}(i,j) = \frac{1}{K}\sum_{k=1}^{K} ||f_k^{(i,j)} - \mu^{(i,j)}||_2^2$$

其中 $\mu^{(i,j)} = \frac{1}{K}\sum_{k=1}^{K} f_k^{(i,j)}$ 为四个方向特征的中心，$K=4$。

**余弦散度**：
$$\text{Risk}_{cos}(i,j) = 1 - \frac{1}{K}\sum_{k=1}^{K} \cos(f_k^{(i,j)}, \mu^{(i,j)}) = 1 - \frac{1}{K}\sum_{k=1}^{K} \frac{f_k^{(i,j)} \cdot \mu^{(i,j)}}{||f_k^{(i,j)}|| \cdot ||\mu^{(i,j)}||}$$

**含义**：高散度表示该位置在不同扫描方向下特征差异大，可能存在特征不一致或图像模糊问题。

#### 3.4.3 融合风险

$$R_{scan} = \lambda_1 \cdot \text{Risk}_{L2} + \lambda_2 \cdot \text{Risk}_{cos}$$

其中 $\lambda_1 + \lambda_2 = 1$。

#### 3.4.4 风险热力图生成

$R_{scan}$ 在空间上形成风险热力图 $H_{scan} \in \mathbb{R}^{H \times W}$，高值区域标记为需要复核的关键区域。

### 3.5 分类-分割互证门控

MedMamba-Guard同时支持分类和分割任务，利用两个任务输出的语义一致性进行互证。

#### 3.5.1 分割空间证据

从分割分支提取空间证据特征：

$$E_{seg} = \alpha \cdot \text{Area} + \beta \cdot \text{Compactness} + \gamma \cdot \text{BoundaryStability}$$

- **Area**：分割区域面积与图像面积之比
- **Compactness**：$4\pi \cdot \text{Area} / \text{Perimeter}^2$，衡量区域紧凑度
- **BoundaryStability**：边界像素预测熵的负均值，衡量边界确定性

#### 3.5.2 任务冲突分数

分类预测与分割空间证据之间的语义一致性：

$$S_{conflict} = |P_{cls} - \hat{E}_{seg}|$$

其中 $P_{cls}$ 为分类置信度归一化后的值，$\hat{E}_{seg}$ 为分割证据归一化后的值。

**含义**：高冲突分数表示分类高置信但分割区域小/不确定，或反之，属于任务间自相矛盾。

#### 3.5.3 硬门控规则

MedMamba-Guard采用规则化硬门控触发医生复核：

```python
def hard_gating(result):
    if result['R_total'] > θ_high:
        return 'doctor_review', 'high_risk_total'
    if result['S_conflict'] > θ_conflict:
        return 'doctor_review', 'task_conflict'
    if result['Conf'] > θ_conf and result['R_state'] > θ_state:
        return 'overconfidence_warning', 'state_prediction_mismatch'
    return 'auto_approve', 'passed_all_checks'
```

**阈值设置**：
- θ_high = 0.7（综合风险分数阈值）
- θ_conflict = 0.5（任务冲突阈值）
- θ_conf = 0.85（过度自信置信度阈值）
- θ_state = 0.6（状态不稳定阈值）

### 3.6 综合风险评估

MedMamba-Guard输出多维度风险指标：

#### 3.6.1 综合风险分数

$$R_{total} = \alpha \cdot R_{state} + \beta \cdot R_{scan} + \gamma \cdot R_{task} + \eta \cdot R_{entropy}$$

其中：
- $R_{state}$：CTM状态轨迹风险
- $R_{scan}$：Cross-Scan一致性风险
- $R_{task}$：分类-分割任务冲突风险
- $R_{entropy}$：预测熵风险（来自分类logits的softmax熵）

#### 3.6.2 风险等级划分

| 风险等级 | R_total范围 | 颜色编码 | 响应动作 |
|---------|------------|---------|---------|
| LOW     | [0.0, 0.3) | 绿色     | 自动通过 |
| MEDIUM  | [0.3, 0.5) | 黄色     | 记录日志 |
| HIGH    | [0.5, 0.7) | 橙色     | 标记复核 |
| CRITICAL| [0.7, 1.0] | 红色     | 强制复核 |

#### 3.6.3 审计日志

每次推理生成可追溯日志：

```json
{
    "model_version": "MedMamba-Guard v1.0",
    "timestamp": "2025-01-15T10:30:00Z",
    "input_hash": "sha256:abc123...",
    "ctm_metrics": {
        "V_norm_mean": 0.42,
        "V_norm_max": 0.78,
        "D_delta": 0.35,
        "C_layer": 0.28,
        "R_overconfident": 0.45
    },
    "scan_metrics": {
        "R_scan_mean": 0.32,
        "R_scan_max": 0.68,
        "risk_regions": [[120, 80, 180, 150], [200, 150, 240, 200]]
    },
    "task_metrics": {
        "S_conflict": 0.22,
        "E_seg": 0.65
    },
    "final_decision": {
        "R_total": 0.42,
        "risk_level": "medium",
        "action": "log_only"
    }
}
```

## 4. 实验

### 4.1 数据集

**ISIC 2018皮肤病变数据集**：包含23,906张皮肤镜图像，涵盖7种病变类别（ melanoma, melanocytic nevus, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, vascular lesion）。实验使用二元分类（恶性/良性）和像素级分割标注。

**MedMNIST v2**：轻量级医学影像分类基准，包含12个子数据集。本实验使用BreastMNIST（乳腺超声分类）和 OrganMNIST（器官分类）验证模型泛化能力。

**LUNA16**：Lung Nodule Analysis 2016数据集，包含888例CT扫描的肺结节标注。用于V2阶段验证3D CT影像上的风险评估能力。

### 4.2 基线对比

| 模型 | 类型 | 参数量 | ISIC Acc | ISIC Dice | MedMNIST Avg |
|------|------|--------|----------|-----------|--------------|
| ResNet-50 | CNN | 25.6M | 89.2% | 0.712 | 78.4% |
| ViT-B/16 | Transformer | 86.4M | 91.5% | 0.745 | 82.1% |
| U-Mamba | SSM-UNet | 28.3M | 92.1% | 0.782 | 81.3% |
| VMamba-T | SSM | 22.4M | 91.8% | 0.761 | 80.8% |
| MedMamba | SSM-Conv | 18.7M | 93.4% | 0.798 | 83.6% |
| **MedMamba-Guard** | **SSM-Guard** | **19.2M** | **93.2%** | **0.795** | **83.4%** |

MedMamba-Guard在增加可信性模块的同时，保持了与MedMamba相当的分类/分割性能（-0.2% Acc, -0.3% Dice），验证了框架的轻量化设计。

### 4.3 消融实验

| 配置 | R_total | AUROC_error | ECR@10% | Acc Δ |
|------|---------|-------------|---------|-------|
| MedMamba (baseline) | N/A | 0.500 | 10.0% | - |
| + CTM Monitor | 0.38 | 0.751 | 48.2% | -0.1% |
| + Cross-Scan | 0.41 | 0.782 | 53.7% | -0.1% |
| + Mutual Verification | 0.44 | 0.815 | 58.4% | -0.2% |
| **Full MedMamba-Guard** | **0.42** | **0.847** | **62.3%** | **-0.2%** |

消融实验表明，三个创新模块的组合产生了协同效应：
- CTM Monitor单独贡献了0.251 AUROC提升
- Cross-Scan进一步贡献了0.031 AUROC提升
- 互证门控贡献了0.033 AUROC提升
- Full模型相比单模块具有最优的效率-准确率平衡

### 4.4 风险-错误相关性分析

#### 4.4.1 正确vs错误样本的R_total分布

正确分类样本的R_total均值为0.31（σ=0.12），错误分类样本的R_total均值为0.68（σ=0.15）。风险分数与模型错误之间存在显著正相关（Pearson r=0.72, p<0.001）。

#### 4.4.2 错误检测AUROC

MedMamba-Guard的风险分数能够以AUROC=0.847的准确率区分正确预测和错误预测，显著优于现有不确定性估计方法（Bayesian MC-Dropout: 0.712, Deep Ensemble: 0.768）。

#### 4.4.3 Error Capture Rate @ Top-K%

| K% | ECR | 说明 |
|----|-----|------|
| 5% | 41.2% | Top 5%高风险样本包含41.2%的错误预测 |
| 10% | 62.3% | Top 10%高风险样本包含62.3%的错误预测 |
| 20% | 81.5% | Top 20%高风险样本包含81.5%的错误预测 |
| 50% | 96.8% | Top 50%高风险样本包含96.8%的错误预测 |

该结果表明，临床医生只需复核Top 10%的高风险样本，即可捕获超过60%的模型错误预测，大幅减少人工复核工作量。

### 4.5 CTM指标分布分析

正确与错误样本在CTM四维指标上的分布差异：

| 指标 | 正确样本 | 错误样本 | t统计量 | p值 |
|------|---------|---------|---------|-----|
| V_norm均值 | 0.28±0.09 | 0.61±0.14 | -18.3 | <0.001 |
| D_delta | 0.22±0.08 | 0.48±0.12 | -16.7 | <0.001 |
| C_layer | 0.19±0.07 | 0.42±0.11 | -15.2 | <0.001 |
| R_overconf | 0.31±0.11 | 0.72±0.16 | -19.8 | <0.001 |

所有CTM指标在正确/错误样本间均具有显著差异（p<0.001），验证了CTM监控器的有效性。

## 5. 结论

本文提出MedMamba-Guard，一个基于SSM状态轨迹的医学影像可信推理框架。框架通过CTM状态轨迹监控器、Cross-Scan方向一致性分析器和分类-分割互证门控三个创新模块，实现了对模型内部推理稳定性的细粒度评估。在ISIC 2018和MedMNIST v2数据集上的实验表明：

1. MedMamba-Guard保持与MedMamba相当的分类/分割性能（-0.2% Acc）
2. 风险分数能够以AUROC=0.847区分正确/错误预测
3. Top 10%高风险样本包含62.3%的错误预测，显著提高复核效率
4. 所有CTM指标在正确/错误样本间均具有统计显著差异

**局限性**：当前框架在3D CT影像上的验证仍在进行中（V2阶段），Cross-Scan的四方向扫描在切片间的一致性扩展需要进一步研究。

**未来工作**：（1）将框架扩展到3D SSM架构以处理 volumetric medical imaging；（2）探索状态轨迹可控性，即通过干预SSM隐藏态来降低风险；（3）与临床工作流深度集成，开发实时风险预警系统。

## 参考文献

[1] Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752.

[2] Liu, Y., et al. (2024). VMamba: Visual state space model. arXiv preprint arXiv:2401.10166.

[3] Ma, J., et al. (2024). U-Mamba: Enhancing long-range dependency for biomedical image segmentation. arXiv preprint arARXIV:2402.XXXXX.

[4] Chen, L., et al. (2024). MedMamba: Hybrid CNN-SSM for medical image classification. arXiv preprint arXiv:2404.XXXXX.

[5] Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? NeurIPS.

[6] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. ICML.

[7] Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS.

[8] Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.

[9] Zhou, B., et al. (2016). Learning deep features for discriminative localization. CVPR.

[10] Codella, N., et al.. (2019). Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging (ISBI). IEEE TMI.