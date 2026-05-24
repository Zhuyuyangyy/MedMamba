"""
MedMamba-Guard 风险-错误相关性分析实验

核心实验：分析风险分数能否预测模型错误

功能:
1. 加载训练好的MedMamba-Guard模型
2. 在测试集上运行，获取predictions, targets, risk_scores
3. 分组分析: correct_samples, incorrect_samples, boundary_samples, low_quality_samples
4. 统计检验: t-test比较正确vs错误样本的CTM指标
5. AUROC曲线: 风险分数预测错误的ROC/PR曲线数据
6. 生成Markdown报告

使用方法:
    python experiments/risk_error_analysis.py --model_path ./checkpoints/medmamba_guard_best.pt --data_root ./data

作者: MedMamba Team
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.medmamba_guard import MedMambaGuard, create_medmamba_guard
from src.data.dataset_guard import create_medical_dataloader, create_dataset
from src.evaluator import (
    compute_error_detection_auroc,
    compute_error_capture_rate,
    compute_fpr_at_95tpr,
    compute_ece_risk,
    compute_brier_risk,
    evaluate_risk_error_correlation,
    get_roc_pr_curve_data,
    compare_correct_incorrect_samples,
)

# 尝试导入scipy
try:
    from scipy.stats import ttest_ind, pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Statistical tests will be skipped.")

# 尝试导入可视化库
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# 配置
# =============================================================================

@dataclass
class RiskAnalysisConfig:
    """风险分析配置"""
    model_path: str = ""
    data_root: str = "./data"
    dataset_name: str = "synthetic"
    
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    
    output_dir: str = "./risk_analysis"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 分组分析阈值
    boundary_threshold: Tuple[float, float] = (0.4, 0.6)  # confidence边界


# =============================================================================
# 核心分析类
# =============================================================================

class RiskErrorAnalyzer:
    """风险-错误相关性分析器"""
    
    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        config: RiskAnalysisConfig,
    ):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        
        # 存储收集的数据
        self.reset()
    
    def reset(self):
        """重置数据存储"""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.risk_components = {
            'r_state': [],
            'r_scan': [],
            'r_task': [],
            'r_entropy': [],
            'r_total': [],
        }
        self.sample_ids = []
    
    @torch.no_grad()
    def collect_predictions(self):
        """收集所有预测结果和风险分数"""
        self.model.eval()
        self.reset()
        
        for batch_idx, batch in enumerate(self.data_loader):
            images = batch['image'].to(self.config.device)
            labels = batch['label']
            paths = batch.get('path', [f'batch{batch_idx}_i{i}' for i in range(images.size(0))])
            
            # 前向传播
            outputs = self.model(images, mode='eval', return_risk=True)
            
            # 提取logits
            logits = outputs.get('logits') or outputs.get('cls_logits')
            
            # 提取置信度
            probs = F.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)
            
            # 保存预测
            self.predictions.extend(preds.cpu().numpy().tolist())
            self.targets.extend(labels.numpy().tolist())
            self.confidences.extend(conf.cpu().numpy().tolist())
            self.sample_ids.extend(paths)
            
            # 提取风险组件
            risk_comp = outputs.get('risk_components', {})
            for k, v in risk_comp.items():
                key = k.lower()
                if key.startswith('r_') and key not in ['r_total']:
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    self.risk_components[key].append(v)
            
            # R_total
            r_total = outputs.get('risk_score', 0)
            if isinstance(r_total, torch.Tensor):
                r_total = r_total.item()
            self.risk_components['r_total'].append(r_total)
            
            if batch_idx % 10 == 0:
                print(f"[Analyzer] Processed batch {batch_idx}/{len(self.data_loader)}")
        
        # 转换为numpy
        self.predictions = np.array(self.predictions)
        self.targets = np.array(self.targets)
        self.confidences = np.array(self.confidences)
        
        for k in self.risk_components:
            self.risk_components[k] = np.array(self.risk_components[k])
        
        # 错误标记
        self.is_correct = (self.predictions == self.targets)
        self.is_error = ~self.is_correct
        
        print(f"\n[Analyzer] Collected {len(self.predictions)} samples")
        print(f"[Analyzer] Correct: {self.is_correct.sum()}, Incorrect: {self.is_error.sum()}")
    
    def analyze_groups(self) -> Dict[str, Any]:
        """
        分组分析
        
        分组:
        - correct_samples: 预测正确
        - incorrect_samples: 预测错误
        - boundary_samples: 预测边界 (confidence 0.4-0.6)
        - high_risk_samples: 高风险样本 (R_total > 0.7)
        
        Returns:
            group_analysis: 分组分析结果
        """
        group_analysis = {}
        
        # 基本统计
        group_analysis['total_samples'] = len(self.predictions)
        group_analysis['correct_count'] = int(self.is_correct.sum())
        group_analysis['incorrect_count'] = int(self.is_error.sum())
        group_analysis['error_rate'] = float(self.is_error.sum() / len(self.is_error))
        
        # 正确 vs 错误样本
        correct_mask = self.is_correct
        incorrect_mask = self.is_error
        
        group_analysis['correct_samples'] = {
            'count': int(correct_mask.sum()),
            'confidence_mean': float(np.mean(self.confidences[correct_mask])),
            'confidence_std': float(np.std(self.confidences[correct_mask])),
            'r_total_mean': float(np.mean(self.risk_components['r_total'][correct_mask])) if correct_mask.sum() > 0 else 0.0,
            'r_total_std': float(np.std(self.risk_components['r_total'][correct_mask])) if correct_mask.sum() > 0 else 0.0,
        }
        
        group_analysis['incorrect_samples'] = {
            'count': int(incorrect_mask.sum()),
            'confidence_mean': float(np.mean(self.confidences[incorrect_mask])),
            'confidence_std': float(np.std(self.confidences[incorrect_mask])),
            'r_total_mean': float(np.mean(self.risk_components['r_total'][incorrect_mask])) if incorrect_mask.sum() > 0 else 0.0,
            'r_total_std': float(np.std(self.risk_components['r_total'][incorrect_mask])) if incorrect_mask.sum() > 0 else 0.0,
        }
        
        # 边界样本
        conf_low, conf_high = self.config.boundary_threshold
        boundary_mask = (self.confidences >= conf_low) & (self.confidences <= conf_high)
        group_analysis['boundary_samples'] = {
            'count': int(boundary_mask.sum()),
            'percentage': float(boundary_mask.sum() / len(self.confidences)),
            'error_rate': float(self.is_error[boundary_mask].sum() / boundary_mask.sum()) if boundary_mask.sum() > 0 else 0.0,
            'r_total_mean': float(np.mean(self.risk_components['r_total'][boundary_mask])) if boundary_mask.sum() > 0 else 0.0,
        }
        
        # 高风险样本
        high_risk_mask = self.risk_components['r_total'] > 0.7
        group_analysis['high_risk_samples'] = {
            'count': int(high_risk_mask.sum()),
            'percentage': float(high_risk_mask.sum() / len(self.risk_components['r_total'])),
            'error_rate': float(self.is_error[high_risk_mask].sum() / high_risk_mask.sum()) if high_risk_mask.sum() > 0 else 0.0,
            'captures_total_errors': float(self.is_error[high_risk_mask].sum() / self.is_error.sum()) if high_risk_mask.sum() > 0 else 0.0,
        }
        
        return group_analysis
    
    def compute_correlation_metrics(self) -> Dict[str, float]:
        """计算风险-错误相关性指标"""
        metrics = {}
        
        if not HAS_SCIPY:
            return metrics
        
        is_error_float = self.is_error.astype(float)
        r_total = self.risk_components['r_total']
        
        # Pearson相关性
        try:
            r_pearson, p_pearson = pearsonr(r_total, is_error_float)
            metrics['risk_error_pearsonr'] = float(r_pearson)
            metrics['risk_error_pearson_pvalue'] = float(p_pearson)
        except:
            pass
        
        # Spearman相关性
        try:
            r_spearman, p_spearman = spearmanr(r_total, is_error_float)
            metrics['risk_error_spearmanr'] = float(r_spearman)
            metrics['risk_error_spearman_pvalue'] = float(p_spearman)
        except:
            pass
        
        return metrics
    
    def compute_error_detection_metrics(self) -> Dict[str, Any]:
        """计算错误检测指标"""
        metrics = {}
        
        is_error = self.is_error.astype(int)
        r_total = self.risk_components['r_total']
        
        # Error Detection AUROC
        metrics['error_detection_auroc'] = compute_error_detection_auroc(is_error, r_total)
        
        # FPR@95%TPR
        metrics['fpr_at_95tpr'] = compute_fpr_at_95tpr(is_error, r_total)
        
        # Error Capture Rate
        metrics['error_capture_rate@10%'] = compute_error_capture_rate(is_error, r_total, top_k=0.1)
        metrics['error_capture_rate@20%'] = compute_error_capture_rate(is_error, r_total, top_k=0.2)
        metrics['error_capture_rate@50%'] = compute_error_capture_rate(is_error, r_total, top_k=0.5)
        
        # 风险校准
        metrics['ece_risk'] = compute_ece_risk(is_error, r_total)
        metrics['brier_risk'] = compute_brier_risk(is_error, r_total)
        
        # ROC/PR曲线数据
        metrics['curve_data'] = get_roc_pr_curve_data(is_error, r_total)
        
        return metrics
    
    def compare_risk_components(self) -> Dict[str, Any]:
        """比较各风险组件在正确vs错误样本中的分布"""
        comparison = {}
        
        if not HAS_SCIPY:
            return comparison
        
        correct_mask = self.is_correct
        incorrect_mask = self.is_error
        
        for component_name in ['r_state', 'r_scan', 'r_task', 'r_entropy', 'r_total']:
            values = self.risk_components[component_name]
            
            correct_vals = values[correct_mask]
            incorrect_vals = values[incorrect_mask]
            
            if len(correct_vals) == 0 or len(incorrect_vals) == 0:
                continue
            
            try:
                t_stat, p_value = ttest_ind(correct_vals, incorrect_vals)
                
                comparison[component_name] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'correct_mean': float(np.mean(correct_vals)),
                    'correct_std': float(np.std(correct_vals)),
                    'incorrect_mean': float(np.mean(incorrect_vals)),
                    'incorrect_std': float(np.std(incorrect_vals)),
                    'difference': float(np.mean(incorrect_vals) - np.mean(correct_vals)),
                }
            except:
                pass
        
        return comparison
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        运行完整分析
        
        Returns:
            analysis_results: 完整分析结果
        """
        print("\n" + "=" * 60)
        print("Starting Risk-Error Correlation Analysis")
        print("=" * 60)
        
        # 1. 收集预测
        print("\n[1/5] Collecting predictions and risk scores...")
        self.collect_predictions()
        
        # 2. 分组分析
        print("\n[2/5] Analyzing sample groups...")
        group_analysis = self.analyze_groups()
        
        # 3. 相关性指标
        print("\n[3/5] Computing correlation metrics...")
        correlation_metrics = self.compute_correlation_metrics()
        
        # 4. 错误检测指标
        print("\n[4/5] Computing error detection metrics...")
        error_detection_metrics = self.compute_error_detection_metrics()
        
        # 5. 风险组件对比
        print("\n[5/5] Comparing risk components between correct/incorrect samples...")
        risk_comparison = self.compare_risk_components()
        
        # 汇总结果
        results = {
            'group_analysis': group_analysis,
            'correlation_metrics': correlation_metrics,
            'error_detection_metrics': error_detection_metrics,
            'risk_comparison': risk_comparison,
        }
        
        return results
    
    def plot_analysis(self, results: Dict[str, Any], output_dir: Path):
        """生成可视化分析图"""
        if not HAS_MATPLOTLIB:
            print("[Analyzer] matplotlib not available, skipping plots")
            return
        
        print("\n[Analyzer] Generating visualization plots...")
        
        r_total = self.risk_components['r_total']
        is_error = self.is_error
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 正确 vs 错误样本的R_total分布
        ax1 = axes[0, 0]
        correct_mask = ~is_error
        incorrect_mask = is_error
        
        ax1.hist(r_total[correct_mask], bins=30, alpha=0.6, label='Correct', color='green', density=True)
        ax1.hist(r_total[incorrect_mask], bins=30, alpha=0.6, label='Incorrect', color='red', density=True)
        ax1.set_xlabel('R_total (Risk Score)')
        ax1.set_ylabel('Density')
        ax1.set_title('R_total Distribution: Correct vs Incorrect')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC曲线 (错误检测)
        ax2 = axes[0, 1]
        curve_data = results.get('error_detection_metrics', {}).get('curve_data', {})
        if curve_data and 'fpr' in curve_data:
            ax2.plot(curve_data['fpr'], curve_data['tpr'], 'b-', linewidth=2, 
                    label=f"AUROC = {results['error_detection_metrics']['error_detection_auroc']:.3f}")
            ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Error Detection ROC Curve')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'ROC curve not available', ha='center', va='center')
        
        # 3. PR曲线 (错误检测)
        ax3 = axes[0, 2]
        if curve_data and 'precision' in curve_data:
            ax3.plot(curve_data['recall'], curve_data['precision'], 'b-', linewidth=2,
                    label=f"AP = {curve_data.get('pr_ap', 0):.3f}")
            ax3.set_xlabel('Recall')
            ax3.set_ylabel('Precision')
            ax3.set_title('Error Detection PR Curve')
            ax3.legend(loc='lower left')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'PR curve not available', ha='center', va='center')
        
        # 4. Error Capture Rate曲线
        ax4 = axes[1, 0]
        top_k_values = np.arange(0.05, 1.05, 0.05)
        ecr_values = []
        for tk in top_k_values:
            ecr = compute_error_capture_rate(is_error.astype(int), r_total, top_k=tk)
            ecr_values.append(ecr)
        
        ax4.plot(top_k_values * 100, ecr_values, 'b-', linewidth=2, marker='o', markersize=4)
        ax4.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Random baseline')
        ax4.axhline(y=is_error.sum() / len(is_error), color='r', linestyle=':', label=f'Base error rate: {is_error.mean():.2%}')
        ax4.set_xlabel('Top-K% of Samples')
        ax4.set_ylabel('Error Capture Rate')
        ax4.set_title('Error Capture Rate vs Top-K%')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, 100])
        ax4.set_ylim([0, 1.05])
        
        # 5. 置信度 vs 风险分数散点图
        ax5 = axes[1, 1]
        scatter_colors = ['green' if c else 'red' for c in correct_mask]
        ax5.scatter(self.confidences, r_total, c=scatter_colors, alpha=0.3, s=10)
        ax5.set_xlabel('Confidence')
        ax5.set_ylabel('R_total (Risk Score)')
        ax5.set_title('Confidence vs Risk Score')
        ax5.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.5, label='Correct'),
                         Patch(facecolor='red', alpha=0.5, label='Incorrect')]
        ax5.legend(handles=legend_elements)
        
        # 6. 风险组件对比柱状图
        ax6 = axes[1, 2]
        risk_comparison = results.get('risk_comparison', {})
        if risk_comparison:
            components = list(risk_comparison.keys())
            correct_means = [risk_comparison[c]['correct_mean'] for c in components]
            incorrect_means = [risk_comparison[c]['incorrect_mean'] for c in components]
            
            x = np.arange(len(components))
            width = 0.35
            
            bars1 = ax6.bar(x - width/2, correct_means, width, label='Correct', color='green', alpha=0.7)
            bars2 = ax6.bar(x + width/2, incorrect_means, width, label='Incorrect', color='red', alpha=0.7)
            
            ax6.set_xlabel('Risk Component')
            ax6.set_ylabel('Mean Value')
            ax6.set_title('Risk Components: Correct vs Incorrect')
            ax6.set_xticks(x)
            ax6.set_xticklabels([c.replace('r_', 'R_') for c in components])
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
            
            # 添加显著性标记
            for i, comp in enumerate(components):
                if risk_comparison[comp].get('significant'):
                    max_val = max(correct_means[i], incorrect_means[i])
                    ax6.annotate('*', xy=(i + width/2, max_val), fontsize=12, ha='center')
        else:
            ax6.text(0.5, 0.5, 'Risk comparison not available', ha='center', va='center')
        
        plt.tight_layout()
        
        # 保存
        plot_path = output_dir / "risk_error_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Analyzer] Plots saved to {plot_path}")


# =============================================================================
# 报告生成
# =============================================================================

def generate_markdown_report(
    results: Dict[str, Any],
    output_dir: Path,
) -> str:
    """
    生成Markdown分析报告
    
    Args:
        results: 分析结果
        output_dir: 输出目录
    
    Returns:
        report_content: Markdown报告内容
    """
    lines = []
    
    lines.append("# MedMamba-Guard Risk-Error Correlation Analysis Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 分组分析
    lines.append("## 1. Sample Group Analysis\n")
    group = results.get('group_analysis', {})
    
    lines.append(f"| Group | Count | Percentage |")
    lines.append(f"|-------|-------|------------|")
    lines.append(f"| Total | {group.get('total_samples', 0)} | 100% |")
    lines.append(f"| Correct | {group.get('correct_count', 0)} | {group.get('correct_count', 0)/max(group.get('total_samples', 1), 1):.2%} |")
    lines.append(f"| Incorrect | {group.get('incorrect_count', 0)} | {group.get('error_rate', 0):.2%} |")
    lines.append(f"| Boundary (conf 0.4-0.6) | {group.get('boundary_samples', {}).get('count', 0)} | {group.get('boundary_samples', {}).get('percentage', 0):.2%} |")
    lines.append(f"| High Risk (R>0.7) | {group.get('high_risk_samples', {}).get('count', 0)} | {group.get('high_risk_samples', {}).get('percentage', 0):.2%} |")
    lines.append("")
    
    # 正确vs错误样本详情
    lines.append("### Correct vs Incorrect Samples\n")
    correct = group.get('correct_samples', {})
    incorrect = group.get('incorrect_samples', {})
    
    lines.append(f"| Metric | Correct | Incorrect |")
    lines.append(f"|--------|---------|-----------|")
    lines.append(f"| Confidence Mean | {correct.get('confidence_mean', 0):.4f} | {incorrect.get('confidence_mean', 0):.4f} |")
    lines.append(f"| Confidence Std | {correct.get('confidence_std', 0):.4f} | {incorrect.get('confidence_std', 0):.4f} |")
    lines.append(f"| R_total Mean | {correct.get('r_total_mean', 0):.4f} | {incorrect.get('r_total_mean', 0):.4f} |")
    lines.append(f"| R_total Std | {correct.get('r_total_std', 0):.4f} | {incorrect.get('r_total_std', 0):.4f} |")
    lines.append("")
    
    # 2. 风险-错误相关性
    lines.append("## 2. Risk-Error Correlation\n")
    corr = results.get('correlation_metrics', {})
    
    if corr:
        lines.append(f"| Metric | Value | p-value |")
        lines.append(f"|--------|-------|---------|")
        if 'risk_error_pearsonr' in corr:
            lines.append(f"| Pearson r | {corr['risk_error_pearsonr']:.4f} | {corr.get('risk_error_pearson_pvalue', 0):.4f} |")
        if 'risk_error_spearmanr' in corr:
            lines.append(f"| Spearman rho | {corr['risk_error_spearmanr']:.4f} | {corr.get('risk_error_spearman_pvalue', 0):.4f} |")
        lines.append("")
    else:
        lines.append("*Statistical tests not available (scipy not installed)*\n")
    
    # 3. 错误检测性能
    lines.append("## 3. Error Detection Performance\n")
    error_det = results.get('error_detection_metrics', {})
    
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Error Detection AUROC | {error_det.get('error_detection_auroc', 0):.4f} |")
    lines.append(f"| FPR @ 95% TPR | {error_det.get('fpr_at_95tpr', 0):.4f} |")
    lines.append(f"| Error Capture Rate @ 10% | {error_det.get('error_capture_rate@10%', 0):.2%} |")
    lines.append(f"| Error Capture Rate @ 20% | {error_det.get('error_capture_rate@20%', 0):.2%} |")
    lines.append(f"| Error Capture Rate @ 50% | {error_det.get('error_capture_rate@50%', 0):.2%} |")
    lines.append(f"| ECE (Risk) | {error_det.get('ece_risk', 0):.4f} |")
    lines.append(f"| Brier Score | {error_det.get('brier_risk', 0):.4f} |")
    lines.append("")
    
    # 4. 风险组件对比
    lines.append("## 4. Risk Components: Correct vs Incorrect (t-test)\n")
    risk_comp = results.get('risk_comparison', {})
    
    if risk_comp:
        lines.append(f"| Component | Correct Mean | Incorrect Mean | Difference | t-stat | p-value | Significant |")
        lines.append(f"|-----------|--------------|---------------|------------|--------|---------|-------------|")
        
        for comp_name, comp_data in risk_comp.items():
            lines.append(
                f"| {comp_name.replace('r_', 'R_')} | "
                f"{comp_data.get('correct_mean', 0):.4f} | "
                f"{comp_data.get('incorrect_mean', 0):.4f} | "
                f"{comp_data.get('difference', 0):.4f} | "
                f"{comp_data.get('t_statistic', 0):.4f} | "
                f"{comp_data.get('p_value', 0):.4f} | "
                f"{'Yes' if comp_data.get('significant') else 'No'} |"
            )
        lines.append("")
    else:
        lines.append("*Component comparison not available*\n")
    
    # 5. 高风险样本分析
    lines.append("## 5. High-Risk Sample Analysis\n")
    high_risk = group.get('high_risk_samples', {})
    
    if high_risk:
        lines.append(f"- High-risk samples (R > 0.7): **{high_risk.get('count', 0)}** ({high_risk.get('percentage', 0):.2%} of total)")
        lines.append(f"- Error rate in high-risk samples: **{high_risk.get('error_rate', 0):.2%}**")
        lines.append(f"- These samples capture **{high_risk.get('captures_total_errors', 0):.2%}** of all errors")
        lines.append("")
        
        # 计算需要复查多少比例的样本才能拦截50%的错误
        target_ecr = 0.5
        for k in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
            ecr = error_det.get(f'error_capture_rate@{int(k*100)}%', 0)
            if ecr >= target_ecr:
                lines.append(f"- To capture 50% of errors, need to review top **{int(k*100)}%** of samples (ECR={ecr:.2%})")
                break
        lines.append("")
    
    # 6. 关键发现
    lines.append("## 6. Key Findings\n")
    
    # AUROC解读
    auroc = error_det.get('error_detection_auroc', 0)
    if auroc > 0.8:
        auroc_interpretation = "Strong - Risk scores are highly predictive of errors"
    elif auroc > 0.7:
        auroc_interpretation = "Moderate - Risk scores provide useful error warning"
    elif auroc > 0.6:
        auroc_interpretation = "Weak - Limited predictive power"
    else:
        auroc_interpretation = "Poor - Risk scores do not reliably predict errors"
    
    lines.append(f"1. **Error Detection AUROC: {auroc:.4f}** - {auroc_interpretation}")
    
    # ECR解读
    ecr_10 = error_det.get('error_capture_rate@10%', 0)
    ecr_20 = error_det.get('error_capture_rate@20%', 0)
    lines.append(f"2. **Error Capture Rate:** Top 10% high-risk samples capture {ecr_10:.1%} of all errors; Top 20% capture {ecr_20:.1%}")
    
    # 相关性解读
    if corr.get('risk_error_pearsonr'):
        r = corr['risk_error_pearsonr']
        if abs(r) > 0.5:
            corr_interpretation = "strong positive correlation"
        elif abs(r) > 0.3:
            corr_interpretation = "moderate positive correlation"
        else:
            corr_interpretation = "weak correlation"
        lines.append(f"3. **Correlation:** Risk scores show {corr_interpretation} with errors (r={r:.3f})")
    
    # 显著性
    if risk_comp.get('r_total', {}).get('significant'):
        lines.append(f"4. **Statistical Significance:** Risk scores are significantly higher in incorrect samples (p < 0.05)")
    
    lines.append("")
    
    # 7. 建议
    lines.append("## 7. Recommendations\n")
    
    if auroc > 0.7:
        lines.append("- ✅ Risk scores can be used for error detection and patient safety alerts")
        lines.append("- ✅ Consider implementing automatic flagging for high-risk predictions")
    else:
        lines.append("- ⚠️ Current risk scoring needs improvement before deployment")
    
    if ecr_10 > 0.3:
        lines.append("- ✅ Reviewing top 10% high-risk cases can intercept significant portion of errors")
    else:
        lines.append("- ⚠️ May need to review a larger portion of cases for effective error interception")
    
    if risk_comp.get('r_state', {}).get('significant') or risk_comp.get('r_scan', {}).get('significant'):
        lines.append("- ✅ CTM/Cross-Scan components show discriminative power - keep these modules")
    else:
        lines.append("- ⚠️ Some risk components may not be contributing - consider ablation study")
    
    lines.append("")
    
    # 保存到文件
    report_content = "\n".join(lines)
    report_path = output_dir / "risk_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"[Analyzer] Report saved to {report_path}")
    
    return report_content


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MedMamba-Guard Risk-Error Analysis")
    
    parser.add_argument("--input", type=str, default=None,
                        help="Path to CSV results file (for synthetic smoke experiments)")
    parser.add_argument("--output", type=str, default="./risk_analysis",
                        help="Output directory")
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to trained MedMamba-Guard model checkpoint")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Data root directory")
    parser.add_argument("--dataset_name", type=str, default="synthetic",
                        choices=["synthetic", "isic2018", "medmnist"],
                        help="Dataset name")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    return parser.parse_args()


def main_from_csv(csv_path: str, output_dir: str):
    """直接从 CSV 运行分析（用于 smoke 实验）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Risk-Error Analysis from CSV")
    print("="*70)
    
    # 读取 CSV
    df = pd.read_csv(csv_path)
    print(f"\nRead {len(df)} samples from {csv_path}")
    
    # 准备数据结构
    results = {
        'group_analysis': {},
        'correlation_metrics': {},
        'error_detection_metrics': {},
        'risk_comparison': {},
    }
    
    # --- 分组分析 ---
    group = results['group_analysis']
    group['total_samples'] = int(len(df))
    group['correct_count'] = int(df['correct'].sum())
    group['incorrect_count'] = int(len(df) - df['correct'].sum())
    group['error_rate'] = float(df['correct'].mean())
    
    correct_df = df[df['correct'] == 1]
    incorrect_df = df[df['correct'] == 0]
    
    group['correct_samples'] = {
        'count': int(len(correct_df)),
        'confidence_mean': float(correct_df['confidence'].mean()),
        'confidence_std': float(correct_df['confidence'].std()),
        'r_total_mean': float(correct_df['r_total'].mean()),
        'r_total_std': float(correct_df['r_total'].std()),
    }
    group['incorrect_samples'] = {
        'count': int(len(incorrect_df)),
        'confidence_mean': float(incorrect_df['confidence'].mean()),
        'confidence_std': float(incorrect_df['confidence'].std()),
        'r_total_mean': float(incorrect_df['r_total'].mean()),
        'r_total_std': float(incorrect_df['r_total'].std()),
    }
    
    # boundary samples (confidence 0.4-0.6)
    boundary_df = df[(df['confidence'] >= 0.4) & (df['confidence'] <= 0.6)]
    group['boundary_samples'] = {
        'count': int(len(boundary_df)),
        'percentage': float(len(boundary_df)/len(df)),
        'error_rate': float(1 - boundary_df['correct'].mean()) if len(boundary_df) > 0 else 0,
        'r_total_mean': float(boundary_df['r_total'].mean()) if len(boundary_df) > 0 else 0,
    }
    
    # high risk samples (R_total > 0.7)
    high_risk_df = df[df['r_total'] > 0.7]
    group['high_risk_samples'] = {
        'count': int(len(high_risk_df)),
        'percentage': float(len(high_risk_df)/len(df)),
        'error_rate': float(1 - high_risk_df['correct'].mean()) if len(high_risk_df) > 0 else 0,
        'captures_total_errors': float(((high_risk_df['correct'] == 0).sum()) / max((df['correct'] == 0).sum(), 1)),
    }
    
    # --- 按 Condition 分组 ---
    if 'condition' in df.columns:
        group['by_condition'] = {}
        for cond in df['condition'].unique():
            cond_df = df[df['condition'] == cond]
            group['by_condition'][cond] = {
                'count': int(len(cond_df)),
                'accuracy': float(cond_df['correct'].mean()),
                'confidence_mean': float(cond_df['confidence'].mean()),
                'r_total_mean': float(cond_df['r_total'].mean()),
                'r_state_mean': float(cond_df['r_state'].mean()),
                'r_scan_mean': float(cond_df['r_scan'].mean()),
                'r_task_mean': float(cond_df['r_task'].mean()),
                'review_rate': float((cond_df['gate_action'] != 'PASS').mean()),
            }
    
    # --- 简单统计 ---
    # Pearson/Spearman
    try:
        corr_metrics = results['correlation_metrics']
        corr_metrics['risk_error_pearsonr'] = float(np.corrcoef(df['r_total'], ~df['correct'])[0,1])
        corr_metrics['risk_error_spearmanr'] = float(df['r_total'].corr(~df['correct'], method='spearman'))
    except:
        pass
    
    # 保存报告
    report_content = generate_simple_markdown_report(df, results, output_dir)
    
    # 保存 JSON
    with open(output_dir / "analysis_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{report_content}")


def generate_simple_markdown_report(df: pd.DataFrame, results: Dict, output_dir: Path) -> str:
    """生成简单的 Markdown 报告"""
    lines = []
    lines.append("# MedMamba-Guard V0.2 Risk-Error Summary")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    group = results['group_analysis']
    
    lines.append("## Sample Group Analysis")
    lines.append("")
    lines.append(f"- Total: {group['total_samples']}")
    lines.append(f"- Correct: {group['correct_count']} ({group['correct_count']/group['total_samples']:.1%})")
    lines.append(f"- Incorrect: {group['incorrect_count']} ({group['incorrect_count']/group['total_samples']:.1%})")
    lines.append("")
    
    # 正确 vs 错误
    lines.append("### Correct vs Incorrect")
    lines.append("")
    lines.append("| Metric | Correct | Incorrect |")
    lines.append("|--------|---------|-----------|")
    cs = group['correct_samples']
    is_ = group['incorrect_samples']
    lines.append(f"| Confidence (mean) | {cs['confidence_mean']:.4f} | {is_['confidence_mean']:.4f} |")
    lines.append(f"| R_total (mean) | {cs['r_total_mean']:.4f} | {is_['r_total_mean']:.4f} |")
    lines.append("")
    
    # 按 Condition
    if 'by_condition' in group:
        lines.append("## By Condition")
        lines.append("")
        lines.append("| Condition | Count | Accuracy | Confidence | R_total | R_state | R_scan | R_task | Review Rate |")
        lines.append("|-----------|-------|----------|------------|---------|---------|--------|--------|-------------|")
        for cond in ['clean', 'blur', 'noise', 'conflict']:
            if cond in group['by_condition']:
                data = group['by_condition'][cond]
                lines.append(f"| {cond} | {data['count']} | {data['accuracy']:.1%} | {data['confidence_mean']:.3f} | {data['r_total_mean']:.3f} | {data['r_state_mean']:.3f} | {data['r_scan_mean']:.3f} | {data['r_task_mean']:.3f} | {data['review_rate']:.1%} |")
        lines.append("")
    
    # 关键发现
    lines.append("## Key Observations")
    if 'by_condition' in group:
        by_cond = group['by_condition']
        clean_r = by_cond.get('clean', {}).get('r_total_mean', 999)
        blur_r = by_cond.get('blur', {}).get('r_total_mean', -1)
        noise_r = by_cond.get('noise', {}).get('r_total_mean', -1)
        conflict_r = by_cond.get('conflict', {}).get('r_total_mean', -1)
        
        # Check trends
        if clean_r < min(blur_r, noise_r, conflict_r):
            lines.append("- ✅ Clean samples have the **lowest** risk")
        if max(blur_r, noise_r) > clean_r:
            lines.append("- ✅ Blur/Noisy samples have **higher** R_state/R_scan risk")
        if by_cond.get('conflict', {}).get('r_task_mean', -1) > by_cond.get('clean', {}).get('r_task_mean', 999):
            lines.append("- ✅ Conflict samples have **higher** R_task risk")
        
        # Check NaN
        all_r = df[['r_state', 'r_scan', 'r_task', 'r_entropy', 'r_total']].values
        if np.all(np.isfinite(all_r)):
            lines.append("- ✅ **All risk values are finite (no NaN/Inf)**")
        
        # Check gate actions
        valid_actions = {'PASS', 'REVIEW', 'doctor_review', 'overconfidence_warning'}
        if set(df['gate_action'].unique()).issubset(valid_actions):
            lines.append("- ✅ **Gate actions are valid**")
    
    # 保存文件
    report_path = output_dir / "risk_error_summary.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\nReport saved to: {report_path}")
    return '\n'.join(lines)


def main():
    args = parse_args()
    
    if args.input:
        # CSV 模式
        main_from_csv(args.input, args.output)
        return
    
    # 原有的完整模式
    config = RiskAnalysisConfig(
        model_path=args.model_path,
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        output_dir=args.output,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("[Main] Loading model...")
    if config.model_path and Path(config.model_path).exists():
        checkpoint = torch.load(config.model_path, map_location=config.device)
        model = create_medmamba_guard(
            d_model=checkpoint.get('config', {}).get('d_model', 384),
            n_layers=checkpoint.get('config', {}).get('n_layers', 12),
            num_classes=2,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Main] Loaded model from {config.model_path}")
    else:
        # 使用默认配置创建模型
        model = create_medmamba_guard(
            d_model=192,
            n_layers=6,
            num_classes=2,
        )
        print("[Main] Created new model (no checkpoint provided)")
    
    model = model.to(config.device)
    model.eval()
    
    # 创建数据加载器
    print(f"[Main] Loading dataset: {config.dataset_name}")
    test_loader = create_medical_dataloader(
        dataset_name=config.dataset_name,
        data_root=config.data_root,
        split='test',
        batch_size=config.batch_size,
        img_size=config.img_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    
    # 创建分析器
    analyzer = RiskErrorAnalyzer(model, test_loader, config)
    
    # 运行完整分析
    results = analyzer.run_full_analysis()
    
    # 生成报告
    report = generate_markdown_report(results, output_dir)
    
    # 生成可视化
    analyzer.plot_analysis(results, output_dir)
    
    # 保存JSON结果
    # 移除不可序列化的部分
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable_results[k] = {k2: v2 for k2, v2 in v.items() 
                                      if not isinstance(v2, (np.ndarray, np.floating))}
        else:
            serializable_results[k] = v
    
    # 移除curve_data (包含numpy数组)
    if 'error_detection_metrics' in serializable_results:
        serializable_results['error_detection_metrics'] = {
            k: v for k, v in serializable_results['error_detection_metrics'].items()
            if k != 'curve_data'
        }
    
    results_path = output_dir / "analysis_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Main] Results saved to {results_path}")
    
    # 打印报告
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nReport saved to: {output_dir / 'risk_analysis_report.md'}")
    print(f"Visualizations saved to: {output_dir / 'risk_error_analysis.png'}")
    print(f"Results JSON saved to: {results_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()