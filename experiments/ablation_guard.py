"""
MedMamba-Guard CTM机制消融实验

实验配置:
- Ours_full: 完整MedMamba-Guard (CTM + Cross-Scan + Task Conflict)
- w/o_R_state: 禁用CTM状态轨迹分析
- w/o_R_scan: 禁用Cross-Scan一致性风险分析
- w/o_R_task: 禁用分类-分割互证门控
- only_entropy: 仅使用熵风险
- baseline: 标准MedMamba (无Guard机制)

对比指标:
- accuracy, dice, f1, auroc
- risk_detection_auroc, error_capture_rate@10%, error_capture_rate@20%

使用方法:
    python experiments/ablation_guard.py --data_root ./data --output_dir ./ablation_results

作者: MedMamba Team
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.medmamba_guard import MedMambaGuard, create_medmamba_guard, LightMedMambaGuard
from src.models.medmamba import create_medmamba
from src.data.dataset_guard import create_medical_dataloader, create_dataset
from src.evaluator import Evaluator, EvaluatorConfig, compute_error_detection_auroc, compute_error_capture_rate

# 尝试导入scipy
try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Statistical tests will be skipped.")


# =============================================================================
# 实验配置
# =============================================================================

@dataclass
class AblationConfig:
    """消融实验配置"""
    data_root: str = "./data"
    dataset_name: str = "synthetic"  # synthetic / isic2018 / medmnist
    output_dir: str = "./ablation_results"
    
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    
    # 模型配置
    d_model: int = 192  # 使用较小模型加速实验
    n_layers: int = 6
    num_classes: int = 2
    
    # 训练配置
    epochs: int = 20
    learning_rate: float = 1e-4
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# 消融实验配置列表
ABLATION_EXPERIMENTS = [
    ('Ours_full', {}),
    ('w/o_R_state', {'disable_ctm': True}),
    ('w/o_R_scan', {'disable_cross_scan': True}),
    ('w/o_R_task', {'disable_task_validator': True}),
    ('only_entropy', {'risk_only_entropy': True}),
    ('baseline', {'model_type': 'medmamba'}),
]

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
) -> nn.Module:
    """
    根据消融实验配置创建模型
    
    Args:
        experiment_name: 实验名称
        config: 实验配置
    
    Returns:
        model: 模型实例
    """
    if experiment_name == 'baseline':
        # 标准MedMamba (无Guard)
        return create_medmamba(
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
        
        # 根据实验配置禁用某些模块
        disable_ctm = 'disable_ctm' in ABLATION_EXPERIMENTS[[n for n, c in ABLATION_EXPERIMENTS].index(experiment_name)][1]
        disable_cross_scan = 'disable_cross_scan' in ABLATION_EXPERIMENTS[[n for n, c in ABLATION_EXPERIMENTS].index(experiment_name)][1]
        disable_task_validator = 'disable_task_validator' in ABLATION_EXPERIMENTS[[n for n, c in ABLATION_EXPERIMENTS].index(experiment_name)][1]
        risk_only_entropy = 'risk_only_entropy' in ABLATION_EXPERIMENTS[[n for n, c in ABLATION_EXPERIMENTS].index(experiment_name)][1]
        
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
        
        if risk_only_entropy:
            # 只使用熵作为风险指标 (其他权重设为0)
            model.hard_gating.w_state = 0.0
            model.hard_gating.w_scan = 0.0
            model.hard_gating.w_task = 0.0
            model.hard_gating.w_entropy = 1.0
        
        return model


# =============================================================================
# 训练函数
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    config: AblationConfig,
    epoch: int = 20,
) -> Dict[str, float]:
    """
    训练模型一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        config: 实验配置
        epoch: 训练轮数
    
    Returns:
        metrics: 训练指标
    """
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
    """
    评估模型
    
    Args:
        model: 模型
        data_loader: 数据加载器
        config: 实验配置
        is_guard_model: 是否为Guard模型 (决定是否计算风险指标)
    
    Returns:
        results: 评估结果
    """
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
        # Dice = 2*|X∩Y| / |X|+|Y|
        intersection = np.sum((all_preds == 1) & (all_labels == 1))
        results['dice'] = 2 * intersection / (np.sum(all_preds == 1) + np.sum(all_labels == 1) + 1e-8)
    else:
        results['dice'] = 0.0
    
    # 风险指标 (如果是Guard模型)
    if is_guard_model and len(r_total_list) > 0:
        r_total_np = np.array(r_total_list)
        
        # 风险组件统计
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
        # 非Guard模型
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
    """消融实验运行器"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.device = config.device
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果存储
        self.results = []
    
    def run_single_experiment(
        self,
        experiment_name: str,
        experiment_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        运行单个消融实验
        
        Args:
            experiment_name: 实验名称
            experiment_config: 实验配置
        
        Returns:
            results: 实验结果
        """
        print("\n" + "=" * 60)
        print(f"Running experiment: {experiment_name}")
        print(f"Config: {experiment_config}")
        print("=" * 60)
        
        # 记录开始时间
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
        is_guard_model = experiment_name != 'baseline'
        model = create_model_for_ablation(experiment_name, self.config)
        model = model.to(self.device)
        
        # 训练
        print(f"\n[{experiment_name}] Training for {self.config.epochs} epochs...")
        for epoch in range(self.config.epochs):
            train_metrics = train_model(model, train_loader, self.config)
            print(f"[{experiment_name}] Epoch {epoch+1}/{self.config.epochs} - Loss: {train_metrics['loss']:.4f} - Acc: {train_metrics['accuracy']:.4f}")
        
        # 评估
        print(f"\n[{experiment_name}] Evaluating...")
        eval_results = evaluate_model(model, val_loader, self.config, is_guard_model)
        
        # 记录时间
        elapsed_time = time.time() - start_time
        
        # 合并结果
        result = {
            'experiment': experiment_name,
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
        """
        运行所有消融实验
        
        Returns:
            df: 结果DataFrame
        """
        print("\n" + "=" * 80)
        print("Starting Ablation Study")
        print("=" * 80)
        
        for experiment_name, experiment_config in ABLATION_EXPERIMENTS:
            result = self.run_single_experiment(experiment_name, experiment_config)
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
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 实验配置
        lines.append("## Experiment Configuration\n")
        lines.append(f"- Dataset: {self.config.dataset_name}")
        lines.append(f"- Image Size: {self.config.img_size}")
        lines.append(f"- Batch Size: {self.config.batch_size}")
        lines.append(f"- Epochs: {self.config.epochs}")
        lines.append(f"- Model: d_model={self.config.d_model}, n_layers={self.config.n_layers}\n")
        
        # 结果表格
        lines.append("## Results\n")
        
        # 选择要显示的列
        display_cols = ['experiment', 'accuracy', 'auroc', 'f1', 'dice', 
                        'risk_detection_auroc', 'error_capture_rate@10%', 'error_capture_rate@20%',
                        'r_total_mean']
        
        # 添加可用列
        for col in display_cols:
            if col not in df.columns:
                display_cols.remove(col)
        
        # 生成表格
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
        
        # 找到最佳模型
        if 'risk_detection_auroc' in df.columns:
            best_risk_idx = df['risk_detection_auroc'].idxmax()
            best_risk_exp = df.loc[best_risk_idx, 'experiment']
            best_risk_val = df.loc[best_risk_idx, 'risk_detection_auroc']
            lines.append(f"- Best Error Detection AUROC: **{best_risk_exp}** ({best_risk_val:.4f})\n")
        
        if 'error_capture_rate@10%' in df.columns:
            best_ecr_idx = df['error_capture_rate@10%'].idxmax()
            best_ecr_exp = df.loc[best_ecr_idx, 'experiment']
            best_ecr_val = df.loc[best_ecr_idx, 'error_capture_rate@10%']
            lines.append(f"- Best Error Capture Rate @10%: **{best_ecr_exp}** ({best_ecr_val:.2%})\n")
        
        # 写入文件
        report_content = "\n".join(lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"[Ablation] Summary report saved to {report_path}")
        
        # 也打印到控制台
        print("\n" + report_content)


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MedMamba-Guard Ablation Study")
    
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Data root directory")
    parser.add_argument("--dataset_name", type=str, default="synthetic",
                        choices=["synthetic", "isic2018", "medmnist"],
                        help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./ablation_results",
                        help="Output directory")
    
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers")
    
    parser.add_argument("--d_model", type=int, default=192,
                        help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Number of layers")
    
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs per experiment")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建配置
    config = AblationConfig(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        d_model=args.d_model,
        n_layers=args.n_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    
    # 创建运行器
    runner = AblationRunner(config)
    
    # 运行所有实验
    df = runner.run_all_experiments()
    
    print("\n" + "=" * 60)
    print("Ablation Study Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {config.output_dir}/ablation_results.csv")
    print(f"Summary saved to: {config.output_dir}/ablation_summary.md")
    print("\nResults DataFrame:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()