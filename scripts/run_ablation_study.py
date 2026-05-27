"""
MedMamba消融实验统一脚本
========================

支持四创新点独立验证:
1. CTM状态轨迹监控 (--component=ctm)
2. Cross-Scan一致性风险图 (--component=cross-scan)
3. 分类-分割互证门控 clDice (--component=clDice)
4. 医生复核风险审计 Safe-Mamba (--component=safe-mamba)
5. 全部实验 (--component=all)

使用方法:
    python scripts/run_ablation_study.py --component=all --dataset=synthetic
    python scripts/run_ablation_study.py --component=ctm --dataset=real
    python scripts/run_ablation_study.py --component=all --dataset=synthetic --output_dir=./ablation

输出:
    - ablation_results.csv: 所有实验结果
    - ablation_summary.md: 汇总报告

四创新点对应关系:
    Innovation 1 (CTM):         MedMambaGuard.ctm_monitor - SSM状态轨迹稳定性分析
    Innovation 2 (Cross-Scan): MedMambaGuard.scan_analyzer - 四方向特征散度
    Innovation 3 (clDice):     MedMambaGuard.task_validator - 分类-分割互证
    Innovation 4 (Safe-Mamba): MedMambaGuard.hard_gating - 硬门控规则引擎

作者: MedMamba Team
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# 导入项目模块
from src.models.medmamba_guard import MedMambaGuard, create_medmamba_guard, LightMedMambaGuard
from src.models.medmamba import create_medmamba
from src.data.dataset_guard import create_medical_dataloader, create_dataset, SyntheticMedicalDataset
from src.evaluator import compute_error_detection_auroc, compute_error_capture_rate


# =============================================================================
# 实验配置
# =============================================================================

class Component(Enum):
    """可独立验证的创新组件"""
    CTM = "ctm"           # Innovation 1: CTM状态轨迹监控
    CROSS_SCAN = "cross-scan"  # Innovation 2: Cross-Scan一致性风险图
    CL_DICE = "clDice"    # Innovation 3: 分类-分割互证门控
    SAFE_MAMBA = "safe-mamba"  # Innovation 4: 医生复核风险审计
    ALL = "all"           # 全部实验


@dataclass
class AblationConfig:
    """消融实验配置"""
    # 数据配置
    data_root: str = "./data"
    dataset_name: str = "synthetic"  # synthetic / isic2018 / medmnist
    output_dir: str = "./ablation_results"
    
    # 模型配置
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    d_model: int = 192  # 使用较小模型加速实验
    n_layers: int = 6
    num_classes: int = 2
    
    # 训练配置
    epochs: int = 20
    learning_rate: float = 1e-4
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 要验证的组件
    component: str = "all"  # ctm / cross-scan / clDice / safe-mamba / all


# 四创新点消融实验配置
# 每个元组: (实验名, 禁用哪些组件, 描述)
ABLATION_EXPERIMENTS = {
    Component.ALL: [
        ('Ours_full', {}, '完整MedMamba-Guard (四创新点全开)'),
        ('w/o_CTM', {'disable_ctm': True}, '禁用CTM状态轨迹监控 (创新点1)'),
        ('w/o_CrossScan', {'disable_cross_scan': True}, '禁用Cross-Scan一致性风险图 (创新点2)'),
        ('w/o_clDice', {'disable_task_validator': True}, '禁用分类-分割互证门控 (创新点3)'),
        ('w/o_SafeMamba', {'disable_hard_gating': True}, '禁用医生复核风险审计 (创新点4)'),
        ('Baseline', {'model_type': 'medmamba'}, '标准MedMamba (无Guard机制)'),
    ],
    Component.CTM: [
        ('Ours_full', {}, '完整MedMamba-Guard'),
        ('w/o_CTM', {'disable_ctm': True}, '禁用CTM状态轨迹监控'),
        ('Baseline', {'model_type': 'medmamba'}, '标准MedMamba baseline'),
    ],
    Component.CROSS_SCAN: [
        ('Ours_full', {}, '完整MedMamba-Guard'),
        ('w/o_CrossScan', {'disable_cross_scan': True}, '禁用Cross-Scan一致性风险图'),
        ('Baseline', {'model_type': 'medmamba'}, '标准MedMamba baseline'),
    ],
    Component.CL_DICE: [
        ('Ours_full', {}, '完整MedMamba-Guard'),
        ('w/o_clDice', {'disable_task_validator': True}, '禁用分类-分割互证门控'),
        ('Baseline', {'model_type': 'medmamba'}, '标准MedMamba baseline'),
    ],
    Component.SAFE_MAMBA: [
        ('Ours_full', {}, '完整MedMamba-Guard'),
        ('w/o_SafeMamba', {'disable_hard_gating': True}, '禁用医生复核风险审计'),
        ('Baseline', {'model_type': 'medmamba'}, '标准MedMamba baseline'),
    ],
}

# 对比指标
METRICS_TO_TRACK = [
    'accuracy',
    'dice', 
    'f1',
    'auroc',
    'risk_detection_auroc',
    'error_capture_rate@10%',
    'error_capture_rate@20%',
    'r_total_mean',
    'r_state_mean',
    'r_scan_mean',
    'r_task_mean',
]


# =============================================================================
# 模型创建函数
# =============================================================================

def create_model_for_ablation(
    experiment_name: str,
    config: AblationConfig,
    experiment_kwargs: Dict[str, Any],
) -> Tuple[nn.Module, bool]:
    """
    根据消融实验配置创建模型
    
    Args:
        experiment_name: 实验名称
        config: 实验配置
        experiment_kwargs: 实验特定参数
    
    Returns:
        model: 模型实例
        is_guard_model: 是否为Guard模型
    """
    model_type = experiment_kwargs.get('model_type', 'medmamba_guard')
    disable_ctm = experiment_kwargs.get('disable_ctm', False)
    disable_cross_scan = experiment_kwargs.get('disable_cross_scan', False)
    disable_task_validator = experiment_kwargs.get('disable_task_validator', False)
    disable_hard_gating = experiment_kwargs.get('disable_hard_gating', False)
    
    is_guard_model = model_type != 'medmamba'
    
    if model_type == 'medmamba':
        # 标准MedMamba (无Guard)
        model = create_medmamba(
            d_model=config.d_model,
            n_layers=config.n_layers,
            num_classes=config.num_classes,
        )
    else:
        # MedMamba-Guard变体
        model = create_medmamba_guard(
            d_model=config.d_model,
            n_layers=config.n_layers,
            num_classes=config.num_classes,
        )
        
        # 根据实验配置选择性禁用模块
        if disable_ctm:
            model.use_ctm = False
            if hasattr(model, 'ctm_monitor'):
                for param in model.ctm_monitor.parameters():
                    param.requires_grad = False
        
        if disable_cross_scan:
            model.use_cross_scan = False
            if hasattr(model, 'scan_analyzer'):
                for param in model.scan_analyzer.parameters():
                    param.requires_grad = False
        
        if disable_task_validator:
            model.use_task_validator = False
            if hasattr(model, 'task_validator'):
                for param in model.task_validator.parameters():
                    param.requires_grad = False
        
        if disable_hard_gating:
            # 将所有硬门控阈值设为最高，使其永远不触发
            if hasattr(model, 'hard_gating'):
                model.hard_gating.theta_high = 1.0
                model.hard_gating.theta_conflict = 1.0
                model.hard_gating.theta_conf = 1.0
                model.hard_gating.theta_state = 1.0
    
    return model, is_guard_model


# =============================================================================
# 训练函数
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    config: AblationConfig,
    epoch: int,
) -> Dict[str, float]:
    """训练模型一个epoch"""
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(config.device)
        labels = batch['label'].to(config.device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if isinstance(outputs, dict):
            logits = outputs.get('logits') or outputs.get('cls_logits', outputs.get('output'))
        else:
            logits = outputs
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total if total > 0 else 0.0,
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    config: AblationConfig,
    is_guard_model: bool = True,
) -> Dict[str, Any]:
    """评估模型"""
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_risks = []
    
    # 风险组件
    r_state_list = []
    r_scan_list = []
    r_task_list = []
    r_entropy_list = []
    r_total_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(config.device)
            labels = batch['label'].to(config.device)
            
            outputs = model(images)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits') or outputs.get('cls_logits', outputs.get('output'))
                
                # 提取风险组件 (如果是Guard模型)
                if is_guard_model and 'risk_components' in outputs:
                    risk_comp = outputs['risk_components']
                    for k, v in risk_comp.items():
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        if k == 'R_state':
                            r_state_list.append(v)
                        elif k == 'R_scan':
                            r_scan_list.append(v)
                        elif k == 'R_task':
                            r_task_list.append(v)
                        elif k == 'R_entropy':
                            r_entropy_list.append(v)
                    
                    r_total = outputs.get('risk_score', 0)
                    if isinstance(r_total, torch.Tensor):
                        r_total = r_total.item()
                    r_total_list.append(r_total)
                    all_risks.append(r_total)
            else:
                logits = outputs
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # 计算基础指标
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    results = {
        'loss': total_loss / len(data_loader),
        'accuracy': correct / total if total > 0 else 0.0,
    }
    
    # 计算AUC
    if config.num_classes == 2:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve, auc
            probs_pos = all_probs[:, 1]
            results['auroc'] = auc(*roc_curve(all_labels, probs_pos, pos_label=1)[:2])
        except:
            results['auroc'] = 0.0
        
        try:
            from sklearn.metrics import f1_score
            results['f1'] = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        except:
            results['f1'] = 0.0
    else:
        try:
            from sklearn.metrics import roc_auc_score
            results['auroc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except:
            results['auroc'] = 0.0
        
        try:
            from sklearn.metrics import f1_score
            results['f1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        except:
            results['f1'] = 0.0
    
    # Dice系数 (简化计算，假设二分类)
    if config.num_classes == 2:
        intersection = np.sum((all_preds == 1) & (all_labels == 1))
        results['dice'] = 2 * intersection / (np.sum(all_preds == 1) + np.sum(all_labels == 1) + 1e-8)
    else:
        results['dice'] = 0.0
    
    # 风险指标 (如果是Guard模型)
    if is_guard_model and len(r_total_list) > 0:
        r_total_np = np.array(r_total_list)
        
        results['r_state_mean'] = np.mean(r_state_list) if r_state_list else 0.0
        results['r_scan_mean'] = np.mean(r_scan_list) if r_scan_list else 0.0
        results['r_task_mean'] = np.mean(r_task_list) if r_task_list else 0.0
        results['r_entropy_mean'] = np.mean(r_entropy_list) if r_entropy_list else 0.0
        results['r_total_mean'] = np.mean(r_total_np)
        
        # 错误检测AUROC
        is_error = (all_preds != all_labels).astype(int)
        if is_error.sum() > 0 and is_error.sum() < len(is_error):
            results['risk_detection_auroc'] = compute_error_detection_auroc(is_error, r_total_np)
            results['error_capture_rate@10%'] = compute_error_capture_rate(is_error, r_total_np, top_k=0.1)
            results['error_capture_rate@20%'] = compute_error_capture_rate(is_error, r_total_np, top_k=0.2)
        else:
            results['risk_detection_auroc'] = 0.5
            results['error_capture_rate@10%'] = 0.0
            results['error_capture_rate@20%'] = 0.0
    else:
        results['r_state_mean'] = 0.0
        results['r_scan_mean'] = 0.0
        results['r_task_mean'] = 0.0
        results['r_entropy_mean'] = 0.0
        results['r_total_mean'] = 0.0
        results['risk_detection_auroc'] = 0.5
        results['error_capture_rate@10%'] = 0.0
        results['error_capture_rate@20%'] = 0.0
    
    return results


# =============================================================================
# 消融实验运行器
# =============================================================================

class AblationRunner:
    """消融实验统一运行器"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.device = config.device
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果存储
        self.results = []
        
        # 获取实验列表
        self.component_enum = Component(config.component)
        self.experiments = ABLATION_EXPERIMENTS.get(self.component_enum, ABLATION_EXPERIMENTS[Component.ALL])
    
    def run_single_experiment(
        self,
        experiment_name: str,
        experiment_config: Dict[str, Any],
        experiment_description: str,
    ) -> Dict[str, Any]:
        """运行单个消融实验"""
        print(f"\n{'='*60}")
        print(f"[{experiment_name}] {experiment_description}")
        print(f"Config: {experiment_config}")
        print('='*60)
        
        start_time = time.time()
        
        # 创建数据加载器
        train_loader = create_medical_dataloader(
            dataset_name=self.config.dataset_name,
            data_root=self.config.data_root,
            split='train',
            batch_size=self.config.batch_size,
            img_size=self.config.img_size,
            num_workers=self.config.num_workers,
        )
        
        val_loader = create_medical_dataloader(
            dataset_name=self.config.dataset_name,
            data_root=self.config.data_root,
            split='val',
            batch_size=self.config.batch_size,
            img_size=self.config.img_size,
            num_workers=self.config.num_workers,
        )
        
        # 创建模型
        is_guard_model = experiment_config.get('model_type', 'medmamba_guard') != 'medmamba'
        model, _ = create_model_for_ablation(experiment_name, self.config, experiment_config)
        model = model.to(self.device)
        
        # 训练
        print(f"\n[{experiment_name}] Training for {self.config.epochs} epochs...")
        for epoch in range(self.config.epochs):
            train_metrics = train_model(model, train_loader, self.config, epoch)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"[{experiment_name}] Epoch {epoch+1}/{self.config.epochs} - "
                      f"Loss: {train_metrics['loss']:.4f} - Acc: {train_metrics['accuracy']:.4f}")
        
        # 评估
        print(f"\n[{experiment_name}] Evaluating...")
        eval_results = evaluate_model(model, val_loader, self.config, is_guard_model)
        
        elapsed_time = time.time() - start_time
        
        # 合并结果
        result = {
            'experiment': experiment_name,
            'description': experiment_description,
            'time_seconds': elapsed_time,
            **eval_results,
        }
        
        print(f"\n[{experiment_name}] Results:")
        for k, v in eval_results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        return result
    
    def run_all_experiments(self) -> pd.DataFrame:
        """运行所有消融实验"""
        print("\n" + "="*80)
        print("MedMamba Ablation Study - Four Innovations Verification")
        print("="*80)
        print(f"Component: {self.component_enum.value}")
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Output: {self.output_dir}")
        print("="*80)
        
        for experiment_name, experiment_config, description in self.experiments:
            result = self.run_single_experiment(experiment_name, experiment_config, description)
            self.results.append(result)
            
            # 保存中间结果
            df_partial = pd.DataFrame(self.results)
            df_partial.to_csv(self.output_dir / "ablation_results_partial.csv", index=False)
        
        # 创建结果DataFrame
        df = pd.DataFrame(self.results)
        
        # 保存完整结果
        df.to_csv(self.output_dir / "ablation_results.csv", index=False)
        print(f"\n[Ablation] Results saved to {self.output_dir / 'ablation_results.csv'}")
        
        # 生成汇总报告
        self._generate_summary_report(df)
        
        return df
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """生成消融实验汇总报告"""
        report_path = self.output_dir / "ablation_summary.md"
        
        lines = []
        lines.append("# MedMamba-Guard Ablation Study Summary\n")
        lines.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Component Tested**: {self.component_enum.value}\n")
        lines.append(f"**Dataset**: {self.config.dataset_name}\n")
        lines.append("\n---\n")
        
        # 实验配置
        lines.append("## Experiment Configuration\n")
        lines.append(f"- Image Size: {self.config.img_size}")
        lines.append(f"- Batch Size: {self.config.batch_size}")
        lines.append(f"- Epochs: {self.config.epochs}")
        lines.append(f"- Model: d_model={self.config.d_model}, n_layers={self.config.n_layers}\n")
        
        # 四创新点说明
        lines.append("## Four Innovations (四创新点)\n")
        lines.append("| ID | Innovation | Component | Description |\n")
        lines.append("|---|-----------|-----------|-------------|\n")
        lines.append("| 1 | CTM状态轨迹监控 | `ctm_monitor` | SSM隐藏态演化轨迹稳定性分析 |")
        lines.append("| 2 | Cross-Scan一致性风险图 | `scan_analyzer` | 四方向特征散度计算 |")
        lines.append("| 3 | 分类-分割互证门控 | `task_validator` | 分类置信度与分割空间证据冲突检测 |")
        lines.append("| 4 | 医生复核风险审计 | `hard_gating` | 硬门控规则引擎，决定是否需要医生复核 |\n")
        
        # 结果表格
        lines.append("## Results\n")
        
        display_cols = ['experiment', 'description', 'accuracy', 'auroc', 'f1', 'dice', 
                        'risk_detection_auroc', 'error_capture_rate@10%', 'error_capture_rate@20%',
                        'r_total_mean']
        
        # 过滤可用列
        display_cols = [c for c in display_cols if c in df.columns]
        
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(display_cols)) + "|")
        
        for _, row in df.iterrows():
            row_data = []
            for col in display_cols:
                val = row.get(col, 0)
                if isinstance(val, float):
                    row_data.append(f"{val:.4f}")
                else:
                    row_data.append(str(val))
            lines.append("| " + " | ".join(row_data) + " |")
        
        lines.append("\n")
        
        # 关键发现
        lines.append("## Key Findings\n")
        
        if 'risk_detection_auroc' in df.columns:
            best_risk_idx = df['risk_detection_auroc'].idxmax()
            best_risk_exp = df.loc[best_risk_idx, 'experiment']
            best_risk_val = df.loc[best_risk_idx, 'risk_detection_auroc']
            lines.append(f"- **Best Error Detection AUROC**: **{best_risk_exp}** ({best_risk_val:.4f})\n")
        
        if 'error_capture_rate@10%' in df.columns:
            best_ecr_idx = df['error_capture_rate@10%'].idxmax()
            best_ecr_exp = df.loc[best_ecr_idx, 'experiment']
            best_ecr_val = df.loc[best_ecr_idx, 'error_capture_rate@10%']
            lines.append(f"- **Best Error Capture Rate @10%**: **{best_ecr_exp}** ({best_ecr_val:.2%})\n")
        
        # Innovation贡献分析
        lines.append("## Innovation Contribution Analysis\n")
        
        # 找到full model的结果
        if 'Ours_full' in df['experiment'].values:
            full_row = df[df['experiment'] == 'Ours_full'].iloc[0]
            baseline_row = df[df['experiment'] == 'Baseline'].iloc[0] if 'Baseline' in df['experiment'].values else None
            
            lines.append("### vs Baseline\n")
            for col in ['accuracy', 'risk_detection_auroc', 'error_capture_rate@10%']:
                if col in df.columns and baseline_row is not None:
                    delta = full_row[col] - baseline_row[col]
                    sign = "+" if delta > 0 else ""
                    lines.append(f"- {col}: {full_row[col]:.4f} ({sign}{delta:.4f} vs baseline)\n")
        
        # 写入文件
        report_content = "\n".join(lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"[Ablation] Summary report saved to {report_path}")
        print("\n" + report_content)


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MedMamba消融实验统一脚本 - 四创新点独立验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_ablation_study.py --component=all --dataset=synthetic
  python scripts/run_ablation_study.py --component=ctm --dataset=real
  python scripts/run_ablation_study.py --component=cross-scan --dataset=synthetic --epochs=10

组件选项:
  all         - 运行所有消融实验 (默认)
  ctm         - 仅验证CTM状态轨迹监控 (创新点1)
  cross-scan  - 仅验证Cross-Scan一致性风险图 (创新点2)
  clDice      - 仅验证分类-分割互证门控 (创新点3)
  safe-mamba  - 仅验证医生复核风险审计 (创新点4)

数据集选项:
  synthetic   - 合成数据集 (默认，快速)
  isic2018    - ISIC 2018皮肤病变数据集
  medmnist    - MedMNIST标准化数据集
        """
    )
    
    parser.add_argument("--component", type=str, default="all",
                        choices=["all", "ctm", "cross-scan", "clDice", "safe-mamba"],
                        help="要验证的组件 (默认: all)")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "isic2018", "medmnist"],
                        help="数据集名称 (默认: synthetic)")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="数据根目录 (默认: ./data)")
    parser.add_argument("--output_dir", type=str, default="./ablation_results",
                        help="输出目录 (默认: ./ablation_results)")
    
    parser.add_argument("--img_size", type=int, default=224,
                        help="图像尺寸 (默认: 224)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批量大小 (默认: 16)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数 (默认: 4)")
    
    parser.add_argument("--d_model", type=int, default=192,
                        help="模型维度 (默认: 192)")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="模型层数 (默认: 6)")
    
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮数 (默认: 20)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率 (默认: 1e-4)")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (默认: cuda)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建配置
    config = AblationConfig(
        data_root=args.data_root,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        d_model=args.d_model,
        n_layers=args.n_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        component=args.component,
    )
    
    print(f"MedMamba Ablation Study Configuration:")
    print(f"  Component: {config.component}")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.epochs}")
    
    # 创建运行器
    runner = AblationRunner(config)
    
    # 运行所有实验
    df = runner.run_all_experiments()
    
    print("\n" + "="*60)
    print("Ablation Study Complete!")
    print("="*60)
    print(f"\nResults saved to: {config.output_dir}/ablation_results.csv")
    print(f"Summary saved to: {config.output_dir}/ablation_summary.md")
    print("\nResults DataFrame:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()