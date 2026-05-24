# MedMamba-Guard

### 基于SSM状态轨迹的医学影像可信推理框架

[![][python-badge]](https://python.org)
[![][pytorch-badge]](https://pytorch.org)
[![][license-badge]](LICENSE)

## 一句话描述

通过SSM隐藏态演化轨迹分析，将医学影像AI从"黑盒分类器"升级为"可解释、可审计、可干预"的可信推理系统。

## 核心创新（4条）

1. **CTM状态轨迹监控** — 首次从SSM隐藏态演化角度量化推理稳定性，通过状态激变度、响应漂移、跨层一致性和状态-输出一致性四维指标识别不稳定预测。

2. **Cross-Scan一致性风险图** — 将多方向扫描从特征增强扩展为可信评估，通过四方向特征散度计算定位空间不一致的高风险区域。

3. **分类-分割互证门控** — 检测模型内部自相矛盾的输出，当分类置信度与分割空间证据冲突时触发硬门控规则。

4. **医生复核风险审计** — 输出风险热力图、复核建议和可追溯日志，支持AI决策全流程透明化。

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      MedMamba-Guard 架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ 输入图像  │───▶│ CNN-SSM 双分支 │───▶│ SSM隐藏态序列         │  │
│  └──────────┘    │   编码器      │    │ {h₁, h₂, ..., hₜ}    │  │
│                  └──────────────┘    └───────────┬───────────┘  │
│                                                   │              │
│         ┌──────────────────┬─────────────────────┴─────┐        │
│         ▼                  ▼                           ▼        │
│  ┌─────────────┐    ┌─────────────┐          ┌─────────────┐  │
│  │ CTM状态轨迹  │    │ Cross-Scan  │          │ 分类-分割    │  │
│  │  监控器     │    │ 方向一致性   │          │  互证门控    │  │
│  │            │    │  风险分析    │          │             │  │
│  └──────┬──────┘    └──────┬──────┘          └──────┬──────┘  │
│         │                  │                      │          │
│         └──────────────────┼──────────────────────┘          │
│                            ▼                                   │
│                  ┌───────────────────┐                        │
│                  │  综合风险评估器    │                        │
│                  │  R_total          │                        │
│                  └────────┬──────────┘                        │
│                           ▼                                     │
│         ┌──────────────────────────────┐                      │
│         │ 风险等级: LOW/MEDIUM/HIGH/   │                      │
│         │         CRITICAL            │                      │
│         │ 审计日志 + 风险热力图 +       │                      │
│         │ 医生复核建议                  │                      │
│         └──────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/your-repo/MedMamba.git
cd MedMamba

# 安装依赖
pip install -r requirements.txt

# 启动交互式菜单
bash start.sh

# 选项说明:
#   1 - Info       查看模型架构信息
#   2 - Benchmark  测试模型性能
#   3 - Train      训练模型
#   4 - Serve      启动API服务 (端口8866)
#   5 - Guard-Demo 风险评估演示（无需训练）
#   6 - Risk-Analysis 风险-错误相关性分析
#   7 - Ablation   消融实验
```

**Python API 调用示例**:

```python
import requests
import numpy as np
from PIL import Image

# 读取图像
img = Image.open("test_lesion.jpg")

# 调用预测+风险评估接口
response = requests.post(
    "http://localhost:8866/predict",
    files={"file": img_to_bytes(img)}
)

result = response.json()
print(f"预测: {result['data']['prediction']}")
print(f"置信度: {result['data']['confidence']:.2%}")
print(f"风险等级: {result['data']['risk_level']}")
print(f"综合风险分数: {result['data']['risk_score']:.2f}")
print(f"风险成分: {result['data']['risk_components']}")
```

## 评估指标

| 指标 | 描述 |
|------|------|
| Accuracy/Dice | 分类准确率和分割重叠率 |
| AUROC_error | 风险分数预测模型错误的AUC，越高表示风险区分度越好 |
| ECR@10% | Error Capture Rate@Top-10%，Top10%高风险样本中捕获的错误比例 |
| R_total | 综合风险分数，融合状态轨迹、扫描一致性、任务冲突和熵风险 |
| R_state | CTM状态轨迹风险分量 |
| R_scan | Cross-Scan一致性风险分量 |

**性能目标值**:
- AUROC_error ≥ 0.80（V1阶段）
- ECR@10% ≥ 55%（V1阶段）
- Accuracy 损失 < 1%（相比无Guard版本）

## 主要接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/predict` | POST | 预测+风险评估，返回完整结果 |
| `/predict-risk-only` | POST | 仅风险评估（无预测结果） |
| `/ctm-metrics` | GET | CTM详细四维指标 |
| `/health` | GET | 服务健康检查 |

## 项目结构

```
MedMamba/
├── src/
│   ├── api/server.py          # FastAPI 服务
│   ├── models/                 # 模型定义
│   │   ├── medmamba.py        # MedMamba核心架构
│   │   └── medmamba_guard.py  # Guard框架
│   └── evaluator.py           # 评估器
├── frontend/
│   └── index.html             # Web演示界面
├── docs/
│   ├── MEDMAMBA_GUARD_TECHNICAL_REPORT.md  # 完整技术报告
│   └── figure_guide.md        # 可视化指南
├── start.sh                   # 启动脚本
├── train_medmamba.py          # 标准训练
└── train_medmamba_guard.py   # Guard训练
```

## 引用

如果你使用了MedMamba-Guard，请引用：

```bibtex
@article{medmamba2025guard,
  title={MedMamba-Guard: A Trustworthy Inference Framework for Medical Imaging Based on SSM State Trajectory Analysis},
  author={Chen, L. and Wang, X. and Liu, Y. and Zhang, J.},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

## 联系方式

- 邮箱: medmamba@example.com
- 项目主页: https://github.com/your-repo/MedMamba
- 文档: https://medmamba.readthedocs.io

[python-badge]: https://img.shields.io/badge/Python-3.8+-blue.svg
[pytorch-badge]: https://img.shields.io/badge/PyTorch-2.0+-red.svg
[license-badge]: https://img.shields.io/badge/License-MIT-green.svg