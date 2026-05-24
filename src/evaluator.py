"""
MedMamba评估器模块

功能:
- 模型评估: 分类报告, 混淆矩阵
- 可视化: 混淆矩阵热图, ROC曲线, PR曲线
- 幻觉风险分布统计 (CTM)
- 风险-错误相关性分析 (MedMamba-Guard)

支持:
- 分类评估: accuracy, precision, recall, F1, AUC, ROC, PR
- 分割评估: Dice, IoU, HD95
- 风险评估: risk_error_correlation, error_detection_auroc, error_capture_rate

作者: MedMamba Team
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

# 尝试导入scipy for statistical tests
try:
    from scipy.stats import ttest_ind, pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Statistical tests will be skipped.")

# 尝试导入可视化库
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# =============================================================================
# 配置
# =============================================================================

@dataclass
class EvaluatorConfig:
    """评估器配置"""
    num_classes: int = 2
    multi_label: bool = False
    task: str = "classification"  # classification / segmentation
    save_dir: str = "./results"
    save_predictions: bool = True
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_pr_curve: bool = True
    # 风险评估配置
    compute_risk_metrics: bool = False  # 是否计算风险相关指标
    risk_threshold: float = 0.7  # 硬门控阈值


# =============================================================================
# 辅助函数
# =============================================================================

def compute_error_capture_rate(
    is_error: np.ndarray,
    risk_scores: np.ndarray,
    top_k: float = 0.1
) -> float:
    """
    计算Error Capture Rate @ Top-K%
    
    在高风险样本Top-K%中能够拦截多少错误样本
    
    Args:
        is_error: 错误标记数组 (bool)
        risk_scores: 风险分数数组
        top_k: Top-K比例 (默认0.1表示10%)
    
    Returns:
        ecr: Error Capture Rate
    """
    n = int(len(risk_scores) * top_k)
    if n <= 0:
        return 0.0
    
    # 找到风险最高的top_k样本索引
    top_k_indices = np.argsort(risk_scores)[-n:]
    
    # 计算拦截的错误数
    captured_errors = is_error[top_k_indices].sum()
    total_errors = is_error.sum()
    
    if total_errors == 0:
        return 0.0
    
    return captured_errors / total_errors


def compute_fpr_at_95tpr(
    is_error: np.ndarray,
    risk_scores: np.ndarray
) -> float:
    """
    计算FPR@95%TPR (错误检测False Positive Rate at 95% True Positive Rate)
    
    Args:
        is_error: 错误标记数组
        risk_scores: 风险分数数组
    
    Returns:
        fpr: FPR at 95% TPR
    """
    if len(np.unique(is_error)) < 2:
        return 0.0
    
    fpr, tpr, thresholds = roc_curve(is_error, risk_scores)
    
    # 找到TPR >= 95% 的最低阈值
    idx_95 = np.searchsorted(tpr, 0.95, side='right') - 1
    idx_95 = max(0, min(idx_95, len(fpr) - 1))
    
    return fpr[idx_95]


def compute_ece_risk(
    is_error: np.ndarray,
    risk_scores: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    计算风险校准误差 (Expected Calibration Error)
    
    ECE = sum(|B_m| / n * |acc(B_m) - conf(B_m)|)
    
    Args:
        is_error: 错误标记数组
        risk_scores: 风险分数数组 (作为confidence)
        n_bins: 分箱数量
    
    Returns:
        ece: 校准误差
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(risk_scores)
    
    for i in range(n_bins):
        bin_mask = (risk_scores >= bins[i]) & (risk_scores < bins[i + 1])
        if i == n_bins - 1:  # 最后一个bin包含右边界
            bin_mask = (risk_scores >= bins[i]) & (risk_scores <= bins[i + 1])
        
        bin_size = bin_mask.sum()
        if bin_size == 0:
            continue
        
        # bin中的平均风险分数
        bin_confidence = risk_scores[bin_mask].mean()
        
        # bin中的错误率
        bin_error_rate = is_error[bin_mask].mean()
        
        # 加权绝对误差
        ece += (bin_size / total) * abs(bin_error_rate - bin_confidence)
    
    return ece


def compute_brier_risk(
    is_error: np.ndarray,
    risk_scores: np.ndarray
) -> float:
    """
    计算风险Brier评分
    
    Brier Score = mean((risk_score - is_error)^2)
    
    Args:
        is_error: 错误标记数组
        risk_scores: 风险分数数组
    
    Returns:
        brier: Brier评分
    """
    return np.mean((risk_scores - is_error.astype(float)) ** 2)


# =============================================================================
# 风险-错误相关性分析
# =============================================================================

def evaluate_risk_error_correlation(
    is_error: np.ndarray,
    risk_scores: np.ndarray
) -> Dict[str, float]:
    """
    计算风险分数与预测错误的相关性
    
    Args:
        is_error: 错误标记数组 (bool)
        risk_scores: 风险分数数组
    
    Returns:
        correlation_metrics: 相关性指标字典
    """
    metrics = {}
    
    if not HAS_SCIPY:
        return metrics
    
    # 转换为float
    is_error_float = is_error.astype(float)
    
    # Pearson相关系数
    try:
        r_pearson, p_pearson = pearsonr(risk_scores, is_error_float)
        metrics['risk_error_pearsonr'] = float(r_pearson)
        metrics['risk_error_pearson_pvalue'] = float(p_pearson)
    except:
        pass
    
    # Spearman相关系数
    try:
        r_spearman, p_spearman = spearmanr(risk_scores, is_error_float)
        metrics['risk_error_spearmanr'] = float(r_spearman)
        metrics['risk_error_spearman_pvalue'] = float(p_spearman)
    except:
        pass
    
    return metrics


def compute_error_detection_auroc(
    is_error: np.ndarray,
    risk_scores: np.ndarray
) -> float:
    """
    用风险分数预测模型错误的AUROC
    
    Args:
        is_error: 错误标记数组
        risk_scores: 风险分数数组
    
    Returns:
        auroc: AUROC分数
    """
    if len(np.unique(is_error)) < 2:
        return 0.5  # 无法计算时返回0.5 (随机)
    
    try:
        auroc = roc_auc_score(is_error, risk_scores)
    except:
        auroc = 0.5
    
    return auroc


def get_roc_pr_curve_data(
    is_error: np.ndarray,
    risk_scores: np.ndarray
) -> Dict[str, Any]:
    """
    获取ROC和PR曲线的原始数据
    
    Args:
        is_error: 错误标记数组
        risk_scores: 风险分数数组
    
    Returns:
        curve_data: 包含fpr, tpr, precision, recall等的字典
    """
    curve_data = {}
    
    if len(np.unique(is_error)) < 2:
        return curve_data
    
    # ROC曲线
    try:
        fpr, tpr, roc_thresholds = roc_curve(is_error, risk_scores)
        roc_auc = auc(fpr, tpr)
        curve_data['fpr'] = fpr.tolist()
        curve_data['tpr'] = tpr.tolist()
        curve_data['roc_thresholds'] = roc_thresholds.tolist()
        curve_data['roc_auc'] = float(roc_auc)
    except:
        pass
    
    # PR曲线
    try:
        precision, recall, pr_thresholds = precision_recall_curve(is_error, risk_scores)
        pr_ap = average_precision_score(is_error, risk_scores)
        curve_data['precision'] = precision.tolist()
        curve_data['recall'] = recall.tolist()
        curve_data['pr_thresholds'] = pr_thresholds.tolist()
        curve_data['pr_ap'] = float(pr_ap)
    except:
        pass
    
    return curve_data


def compare_correct_incorrect_samples(
    correct_metrics: Dict[str, np.ndarray],
    incorrect_metrics: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    比较正确样本 vs 错误样本的CTM指标分布
    
    Args:
        correct_metrics: 正确样本的指标字典
        incorrect_metrics: 错误样本的指标字典
    
    Returns:
        comparison: t-test结果字典
    """
    comparison = {}
    
    if not HAS_SCIPY:
        return comparison
    
    for key in correct_metrics.keys():
        correct_vals = correct_metrics[key]
        incorrect_vals = incorrect_metrics[key]
        
        if len(correct_vals) == 0 or len(incorrect_vals) == 0:
            continue
        
        try:
            t_stat, p_value = ttest_ind(correct_vals, incorrect_vals)
            
            comparison[key] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'correct_mean': float(np.mean(correct_vals)),
                'correct_std': float(np.std(correct_vals)),
                'incorrect_mean': float(np.mean(incorrect_vals)),
                'incorrect_std': float(np.std(incorrect_vals)),
                'significant': p_value < 0.05,
            }
        except:
            pass
    
    return comparison


# =============================================================================
# 评估器
# =============================================================================

class Evaluator:
    """
    MedMamba评估器
    
    支持:
    - 分类评估: accuracy, precision, recall, F1, AUC, ROC, PR
    - 混淆矩阵可视化
    - 幻觉风险分布统计 (CTM)
    - 分割评估: Dice, IoU
    - 风险-错误相关性分析 (MedMamba-Guard)
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        config: EvaluatorConfig,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
    ):
        """
        初始化评估器
        
        Args:
            model: 模型
            data_loader: 数据加载器
            config: 配置
            criterion: 损失函数
            device: 设备
            class_names: 类别名称列表
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        self.class_names = class_names or [f"Class_{i}" for i in range(config.num_classes)]
        
        # 保存路径
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储预测结果
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.all_paths = []
        self.all_hallucination_risks = []
        
        # 风险相关存储
        self.all_risk_components = {
            'r_state': [],
            'r_scan': [],
            'r_task': [],
            'r_entropy': [],
            'r_total': [],
        }
        self.all_confidences = []
    
    def reset(self):
        """重置所有预测结果"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.all_paths = []
        self.all_hallucination_risks = []
        self.all_risk_components = {
            'r_state': [],
            'r_scan': [],
            'r_task': [],
            'r_entropy': [],
            'r_total': [],
        }
        self.all_confidences = []
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        执行评估
        
        Returns:
            results: 评估结果字典
        """
        self.model.eval()
        self.reset()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.data_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            paths = batch.get('path', [''] * images.size(0))
            
            # 前向传播
            outputs = self.model(images)
            
            # 提取logits
            if isinstance(outputs, dict):
                logits = outputs.get('logits') or outputs.get('cls_logits')
                
                # 提取CTM风险
                hallucination_risk = outputs.get('hallucination_risk')
                if hallucination_risk is not None:
                    self.all_hallucination_risks.append(
                        hallucination_risk.cpu().numpy()
                    )
                
                # 提取MedMamba-Guard风险组件
                risk_components = outputs.get('risk_components', {})
                if risk_components:
                    for k in ['r_state', 'r_scan', 'r_task', 'r_entropy']:
                        if k in risk_components:
                            v = risk_components[k]
                            if isinstance(v, torch.Tensor):
                                v = v.item()
                            self.all_risk_components[k].append(v)
                    
                    # R_total
                    r_total = outputs.get('risk_score', 0)
                    if isinstance(r_total, torch.Tensor):
                        r_total = r_total.item()
                    self.all_risk_components['r_total'].append(r_total)
                
                # 置信度
                confidence = outputs.get('confidence', 0)
                if isinstance(confidence, torch.Tensor):
                    confidence = confidence.item()
                self.all_confidences.append(confidence)
                
            else:
                logits = outputs
            
            # 损失
            if self.criterion is not None:
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
            
            # 预测
            if self.config.multi_label:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
            else:
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
            
            # 准确率
            if self.config.multi_label:
                correct += (preds == labels).all(dim=1).sum().item()
            else:
                correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 保存预测
            self.all_preds.append(preds.cpu().numpy())
            self.all_labels.append(labels.cpu().numpy())
            self.all_probs.append(probs.cpu().numpy())
            self.all_paths.extend(paths)
        
        # 合并所有批次
        all_preds = np.concatenate(self.all_preds)
        all_labels = np.concatenate(self.all_labels)
        all_probs = np.concatenate(self.all_probs)
        
        # 计算指标
        results = self._compute_metrics(all_preds, all_labels, all_probs)
        results['loss'] = total_loss / len(self.data_loader)
        results['accuracy'] = correct / total if total > 0 else 0
        
        # 风险指标 (如果启用且有风险数据)
        if self.config.compute_risk_metrics and len(self.all_risk_components['r_total']) > 0:
            risk_results = self._compute_risk_metrics(all_labels, all_preds)
            results.update(risk_results)
        
        return results
    
    def _compute_risk_metrics(
        self,
        all_labels: np.ndarray,
        all_preds: np.ndarray
    ) -> Dict[str, Any]:
        """
        计算风险相关指标
        
        Args:
            all_labels: 真实标签
            all_preds: 预测标签
        
        Returns:
            risk_metrics: 风险指标字典
        """
        risk_metrics = {}
        
        # 收集风险数据
        r_state = np.array(self.all_risk_components['r_state']) if self.all_risk_components['r_state'] else np.array([0])
        r_scan = np.array(self.all_risk_components['r_scan']) if self.all_risk_components['r_scan'] else np.array([0])
        r_task = np.array(self.all_risk_components['r_task']) if self.all_risk_components['r_task'] else np.array([0])
        r_entropy = np.array(self.all_risk_components['r_entropy']) if self.all_risk_components['r_entropy'] else np.array([0])
        r_total = np.array(self.all_risk_components['r_total']) if self.all_risk_components['r_total'] else np.array([0])
        
        # 错误标记
        is_error = (all_preds != all_labels).astype(int)
        
        # Risk-Error相关性
        correlation = evaluate_risk_error_correlation(is_error, r_total)
        risk_metrics.update(correlation)
        
        # Error Detection AUROC
        risk_metrics['risk_detection_auroc'] = compute_error_detection_auroc(is_error, r_total)
        
        # FPR@95%TPR
        risk_metrics['fpr_at_95tpr'] = compute_fpr_at_95tpr(is_error, r_total)
        
        # Error Capture Rate @ 10% and @ 20%
        risk_metrics['error_capture_rate@10%'] = compute_error_capture_rate(is_error, r_total, top_k=0.1)
        risk_metrics['error_capture_rate@20%'] = compute_error_capture_rate(is_error, r_total, top_k=0.2)
        
        # 风险校准误差
        risk_metrics['ece_risk'] = compute_ece_risk(is_error, r_total)
        
        # 风险Brier评分
        risk_metrics['brier_risk'] = compute_brier_risk(is_error, r_total)
        
        # 各风险组件的统计
        risk_metrics['r_state_mean'] = float(np.mean(r_state))
        risk_metrics['r_state_std'] = float(np.std(r_state))
        risk_metrics['r_scan_mean'] = float(np.mean(r_scan))
        risk_metrics['r_scan_std'] = float(np.std(r_scan))
        risk_metrics['r_task_mean'] = float(np.mean(r_task))
        risk_metrics['r_task_std'] = float(np.std(r_task))
        risk_metrics['r_entropy_mean'] = float(np.mean(r_entropy))
        risk_metrics['r_entropy_std'] = float(np.std(r_entropy))
        risk_metrics['r_total_mean'] = float(np.mean(r_total))
        risk_metrics['r_total_std'] = float(np.std(r_total))
        
        # 分组分析：正确样本 vs 错误样本
        correct_mask = is_error == 0
        incorrect_mask = is_error == 1
        
        if correct_mask.sum() > 0 and incorrect_mask.sum() > 0:
            correct_r = r_total[correct_mask]
            incorrect_r = r_total[incorrect_mask]
            
            if HAS_SCIPY:
                try:
                    t_stat, p_value = ttest_ind(correct_r, incorrect_r)
                    risk_metrics['ttest_r_total_tstat'] = float(t_stat)
                    risk_metrics['ttest_r_total_pvalue'] = float(p_value)
                    risk_metrics['ttest_r_total_significant'] = p_value < 0.05
                except:
                    pass
            
            risk_metrics['correct_r_total_mean'] = float(np.mean(correct_r))
            risk_metrics['incorrect_r_total_mean'] = float(np.mean(incorrect_r))
            risk_metrics['correct_r_total_std'] = float(np.std(correct_r))
            risk_metrics['incorrect_r_total_std'] = float(np.std(incorrect_r))
        
        return risk_metrics
    
    def _compute_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
    ) -> Dict[str, Any]:
        """
        计算评估指标
        
        Args:
            preds: 预测标签
            labels: 真实标签
            probs: 预测概率
        
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        # 多标签处理
        if self.config.multi_label:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            
            metrics['accuracy'] = accuracy_score(labels, preds > 0.5)
            metrics['f1'] = f1_score(labels, preds > 0.5, average='macro', zero_division=0)
            
            try:
                metrics['auc'] = roc_auc_score(labels, probs, average='macro')
            except ValueError:
                metrics['auc'] = 0.0
        else:
            # 二分类/多分类指标
            if self.config.num_classes == 2:
                # 二分类
                probs_pos = probs[:, 1]
                
                # AUC
                try:
                    metrics['auc'] = auc(*roc_curve(labels, probs_pos, pos_label=1)[:2])
                except ValueError:
                    metrics['auc'] = 0.0
                
                # F1
                try:
                    metrics['f1'] = f1_score(labels, preds, average='binary', zero_division=0)
                except ValueError:
                    metrics['f1'] = 0.0
                
                # Precision/Recall
                try:
                    metrics['precision'] = classification_report(
                        labels, preds, output_dict=True, zero_division=0
                    )['1']['precision']
                    metrics['recall'] = classification_report(
                        labels, preds, output_dict=True, zero_division=0
                    )['1']['recall']
                except:
                    metrics['precision'] = 0.0
                    metrics['recall'] = 0.0
            else:
                # 多分类
                try:
                    metrics['auc'] = roc_auc_score(
                        labels, probs, multi_class='ovr', average='macro'
                    )
                except ValueError:
                    metrics['auc'] = 0.0
                
                try:
                    metrics['f1'] = f1_score(labels, preds, average='macro', zero_division=0)
                except ValueError:
                    metrics['f1'] = 0.0
        
        # 分类报告
        try:
            if self.config.multi_label:
                metrics['classification_report'] = classification_report(
                    labels, preds > 0.5, target_names=self.class_names, output_dict=True, zero_division=0
                )
            else:
                metrics['classification_report'] = classification_report(
                    labels, preds, target_names=self.class_names, output_dict=True, zero_division=0
                )
        except:
            metrics['classification_report'] = {}
        
        # 混淆矩阵
        if self.config.multi_label:
            metrics['confusion_matrix'] = confusion_matrix(
                labels.flatten(), preds.flatten()
            )
        else:
            metrics['confusion_matrix'] = confusion_matrix(labels, preds)
        
        # CTM幻觉风险统计
        if len(self.all_hallucination_risks) > 0:
            hallucination_risks = np.concatenate(self.all_hallucination_risks)
            metrics['hallucination_risk_mean'] = float(np.mean(hallucination_risks))
            metrics['hallucination_risk_std'] = float(np.std(hallucination_risks))
            metrics['hallucination_risk_max'] = float(np.max(hallucination_risks))
            metrics['hallucination_risk_min'] = float(np.min(hallucination_risks))
            
            # 幻觉风险分布
            metrics['hallucination_risk_distribution'] = {
                'low (0-0.3)': float(np.sum(hallucination_risks < 0.3) / len(hallucination_risks)),
                'medium (0.3-0.7)': float(np.sum((hallucination_risks >= 0.3) & (hallucination_risks < 0.7)) / len(hallucination_risks)),
                'high (0.7-1.0)': float(np.sum(hallucination_risks >= 0.7) / len(hallucination_risks)),
            }
        
        return metrics
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成评估报告
        
        Args:
            results: 评估结果
        
        Returns:
            report: 文本报告
        """
        lines = []
        lines.append("=" * 60)
        lines.append("MedMamba Evaluation Report")
        lines.append("=" * 60)
        
        # 基本指标
        lines.append(f"\nOverall Metrics:")
        lines.append(f"  Loss:      {results.get('loss', 0):.4f}")
        lines.append(f"  Accuracy:  {results.get('accuracy', 0):.4f}")
        lines.append(f"  F1 Score:  {results.get('f1', 0):.4f}")
        lines.append(f"  AUC:       {results.get('auc', 0):.4f}")
        
        if 'precision' in results:
            lines.append(f"  Precision: {results['precision']:.4f}")
        if 'recall' in results:
            lines.append(f"  Recall:    {results['recall']:.4f}")
        
        # CTM幻觉风险
        if 'hallucination_risk_mean' in results:
            lines.append(f"\nCTM Hallucination Risk:")
            lines.append(f"  Mean: {results['hallucination_risk_mean']:.4f}")
            lines.append(f"  Std:  {results['hallucination_risk_std']:.4f}")
            lines.append(f"  Max:  {results['hallucination_risk_max']:.4f}")
            lines.append(f"  Min:  {results['hallucination_risk_min']:.4f}")
            lines.append(f"\n  Risk Distribution:")
            risk_dist = results['hallucination_risk_distribution']
            lines.append(f"    Low (0-0.3):    {risk_dist.get('low (0-0.3)', 0):.2%}")
            lines.append(f"    Medium (0.3-0.7): {risk_dist.get('medium (0.3-0.7)', 0):.2%}")
            lines.append(f"    High (0.7-1.0):   {risk_dist.get('high (0.7-1.0)', 0):.2%}")
        
        # 风险-错误相关性指标
        if 'risk_detection_auroc' in results:
            lines.append(f"\nRisk-Error Analysis:")
            lines.append(f"  Risk Detection AUROC: {results['risk_detection_auroc']:.4f}")
            lines.append(f"  Error Capture Rate @ 10%: {results.get('error_capture_rate@10%', 0):.2%}")
            lines.append(f"  Error Capture Rate @ 20%: {results.get('error_capture_rate@20%', 0):.2%}")
            lines.append(f"  FPR @ 95% TPR: {results.get('fpr_at_95tpr', 0):.4f}")
            
            if 'risk_error_pearsonr' in results:
                lines.append(f"  Pearson Correlation: {results['risk_error_pearsonr']:.4f} (p={results.get('risk_error_pearson_pvalue', 0):.4f})")
            if 'risk_error_spearmanr' in results:
                lines.append(f"  Spearman Correlation: {results['risk_error_spearmanr']:.4f} (p={results.get('risk_error_spearman_pvalue', 0):.4f})")
            
            lines.append(f"\n  Risk Components:")
            lines.append(f"    R_state:  {results.get('r_state_mean', 0):.4f} +/- {results.get('r_state_std', 0):.4f}")
            lines.append(f"    R_scan:   {results.get('r_scan_mean', 0):.4f} +/- {results.get('r_scan_std', 0):.4f}")
            lines.append(f"    R_task:   {results.get('r_task_mean', 0):.4f} +/- {results.get('r_task_std', 0):.4f}")
            lines.append(f"    R_entropy:{results.get('r_entropy_mean', 0):.4f} +/- {results.get('r_entropy_std', 0):.4f}")
            lines.append(f"    R_total:  {results.get('r_total_mean', 0):.4f} +/- {results.get('r_total_std', 0):.4f}")
            
            if 'correct_r_total_mean' in results:
                lines.append(f"\n  Correct vs Incorrect Samples:")
                lines.append(f"    Correct R_total:   {results['correct_r_total_mean']:.4f} +/- {results.get('correct_r_total_std', 0):.4f}")
                lines.append(f"    Incorrect R_total:{results['incorrect_r_total_mean']:.4f} +/- {results.get('incorrect_r_total_std', 0):.4f}")
                if 'ttest_r_total_pvalue' in results:
                    lines.append(f"    t-test p-value: {results['ttest_r_total_pvalue']:.4f} {'(significant)' if results.get('ttest_r_total_significant') else ''}")
        
        # 分类报告
        if 'classification_report' in results:
            lines.append(f"\nClassification Report:")
            report = results['classification_report']
            
            if self.config.num_classes <= 10:
                lines.append(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
                lines.append("  " + "-" * 55)
                
                for class_name in self.class_names:
                    if class_name in report:
                        metrics = report[class_name]
                        lines.append(
                            f"  {class_name:<15} "
                            f"{metrics['precision']:>10.4f} "
                            f"{metrics['recall']:>10.4f} "
                            f"{metrics['f1-score']:>10.4f} "
                            f"{metrics['support']:>10.0f}"
                        )
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def generate_risk_analysis_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        生成风险分析Markdown报告
        
        Args:
            results: 评估结果
            output_path: 输出路径
        
        Returns:
            report: Markdown报告内容
        """
        lines = []
        lines.append("# MedMamba-Guard Risk Analysis Report\n")
        
        # 1. 风险-错误相关性表格
        lines.append("## 1. Risk-Error Correlation Analysis\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        
        if 'risk_detection_auroc' in results:
            lines.append(f"| Error Detection AUROC | {results['risk_detection_auroc']:.4f} |")
        if 'risk_error_pearsonr' in results:
            lines.append(f"| Pearson Correlation | {results['risk_error_pearsonr']:.4f} |")
            lines.append(f"| Pearson p-value | {results.get('risk_error_pearson_pvalue', 0):.4f} |")
        if 'risk_error_spearmanr' in results:
            lines.append(f"| Spearman Correlation | {results['risk_error_spearmanr']:.4f} |")
            lines.append(f"| Spearman p-value | {results.get('risk_error_spearman_pvalue', 0):.4f} |")
        
        lines.append("")
        
        # 2. 错误拦截效率
        lines.append("## 2. Error Capture Efficiency\n")
        lines.append("| Top-K% | Capture Rate |")
        lines.append("|--------|--------------|")
        lines.append(f"| 10% | {results.get('error_capture_rate@10%', 0):.2%} |")
        lines.append(f"| 20% | {results.get('error_capture_rate@20%', 0):.2%} |")
        lines.append("")
        
        # 3. FPR@95%TPR
        lines.append("## 3. FPR at 95% TPR\n")
        lines.append(f"{results.get('fpr_at_95tpr', 0):.4f}\n")
        
        # 4. 风险校准
        lines.append("## 4. Risk Calibration\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| ECE (Risk) | {results.get('ece_risk', 0):.4f} |")
        lines.append(f"| Brier Score | {results.get('brier_risk', 0):.4f} |")
        lines.append("")
        
        # 5. 风险组件统计
        lines.append("## 5. Risk Components Statistics\n")
        lines.append("| Component | Mean | Std |")
        lines.append("|-----------|------|-----|")
        lines.append(f"| R_state | {results.get('r_state_mean', 0):.4f} | {results.get('r_state_std', 0):.4f} |")
        lines.append(f"| R_scan | {results.get('r_scan_mean', 0):.4f} | {results.get('r_scan_std', 0):.4f} |")
        lines.append(f"| R_task | {results.get('r_task_mean', 0):.4f} | {results.get('r_task_std', 0):.4f} |")
        lines.append(f"| R_entropy | {results.get('r_entropy_mean', 0):.4f} | {results.get('r_entropy_std', 0):.4f} |")
        lines.append(f"| R_total | {results.get('r_total_mean', 0):.4f} | {results.get('r_total_std', 0):.4f} |")
        lines.append("")
        
        # 6. 正确 vs 错误样本对比
        if 'correct_r_total_mean' in results:
            lines.append("## 6. Correct vs Incorrect Samples (R_total)\n")
            lines.append("| Group | Mean | Std |")
            lines.append("|-------|------|-----|")
            lines.append(f"| Correct | {results['correct_r_total_mean']:.4f} | {results.get('correct_r_total_std', 0):.4f} |")
            lines.append(f"| Incorrect | {results['incorrect_r_total_mean']:.4f} | {results.get('incorrect_r_total_std', 0):.4f} |")
            
            if 'ttest_r_total_pvalue' in results:
                lines.append("")
                lines.append(f"**t-test p-value**: {results['ttest_r_total_pvalue']:.4f}")
                if results.get('ttest_r_total_significant'):
                    lines.append("**Significant difference (p < 0.05)**")
            lines.append("")
        
        report_content = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"[Evaluator] Risk analysis report saved to {output_path}")
        
        return report_content
    
    def plot_confusion_matrix(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ):
        """绘制混淆矩阵"""
        if not HAS_MATPLOTLIB:
            print("[Evaluator] matplotlib not available, skipping confusion matrix plot")
            return
        
        cm = results.get('confusion_matrix')
        if cm is None:
            return
        
        fig, ax = plt.subplots(figsize=(max(8, len(self.class_names)), max(6, len(self.class_names))))
        
        # 归一化
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制热图
        sns_plot = sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={'label': 'Proportion'},
        )
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "confusion_matrix.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Evaluator] Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """绘制ROC曲线"""
        if not HAS_MATPLOTLIB:
            print("[Evaluator] matplotlib not available, skipping ROC curve plot")
            return
        
        if self.config.num_classes == 2:
            fpr, tpr, _ = roc_curve(labels, probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.save_dir / "roc_curve.png"
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[Evaluator] ROC curve saved to {save_path}")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for i, class_name in enumerate(self.class_names):
                y_true_bin = (labels == i).astype(int)
                y_score = probs[:, i]
                
                if len(np.unique(y_true_bin)) > 1:
                    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves (One-vs-Rest)')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.save_dir / "roc_curve.png"
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[Evaluator] ROC curves saved to {save_path}")
    
    def plot_pr_curve(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """绘制Precision-Recall曲线"""
        if not HAS_MATPLOTLIB:
            print("[Evaluator] matplotlib not available, skipping PR curve plot")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if self.config.num_classes == 2:
            precision, recall, _ = precision_recall_curve(labels, probs[:, 1], pos_label=1)
            ap = average_precision_score(labels, probs[:, 1])
            
            ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {ap:.3f})')
        else:
            for i, class_name in enumerate(self.class_names):
                y_true_bin = (labels == i).astype(int)
                y_score = probs[:, i]
                
                if len(np.unique(y_true_bin)) > 1:
                    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
                    ap = average_precision_score(y_true_bin, y_score)
                    ax.plot(recall, precision, label=f'{class_name} (AP = {ap:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "pr_curve.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Evaluator] PR curve saved to {save_path}")
    
    def plot_hallucination_risk_distribution(
        self,
        risks: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """绘制CTM幻觉风险分布直方图"""
        if not HAS_MATPLOTLIB:
            print("[Evaluator] matplotlib not available, skipping risk distribution plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax1 = axes[0]
        ax1.hist(risks, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(np.mean(risks), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(risks):.3f}')
        ax1.axvline(np.median(risks), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(risks):.3f}')
        ax1.set_xlabel('Hallucination Risk')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Hallucination Risk Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2 = axes[1]
        bp = ax2.boxplot(risks, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax2.set_ylabel('Hallucination Risk')
        ax2.set_title('Hallucination Risk Box Plot')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加统计信息
        stats_text = f"Mean: {np.mean(risks):.4f}\nStd: {np.std(risks):.4f}\nMax: {np.max(risks):.4f}\nMin: {np.min(risks):.4f}"
        ax2.text(0.65, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "hallucination_risk_distribution.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Evaluator] Hallucination risk distribution saved to {save_path}")
    
    def plot_risk_error_analysis(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ):
        """绘制风险-错误分析图"""
        if not HAS_MATPLOTLIB:
            print("[Evaluator] matplotlib not available, skipping risk-error analysis plot")
            return
        
        if 'risk_detection_auroc' not in results:
            return
        
        # 获取风险数据
        r_total = np.array(self.all_risk_components['r_total'])
        if len(r_total) == 0:
            return
        
        all_labels = np.concatenate(self.all_labels)
        all_preds = np.concatenate(self.all_preds)
        is_error = (all_preds != all_labels).astype(int)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 正确 vs 错误样本的R_total分布
        ax1 = axes[0]
        correct_mask = is_error == 0
        incorrect_mask = is_error == 1
        
        if correct_mask.sum() > 0:
            ax1.hist(r_total[correct_mask], bins=30, alpha=0.5, label='Correct', color='green')
        if incorrect_mask.sum() > 0:
            ax1.hist(r_total[incorrect_mask], bins=30, alpha=0.5, label='Incorrect', color='red')
        ax1.set_xlabel('R_total (Risk Score)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('R_total Distribution: Correct vs Incorrect')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC曲线 (错误检测)
        ax2 = axes[1]
        if len(np.unique(is_error)) > 1:
            fpr, tpr, _ = roc_curve(is_error, r_total)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUROC = {roc_auc:.3f})')
            ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Risk-Based Error Detection ROC')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)
        
        # 3. PR曲线 (错误检测)
        ax3 = axes[2]
        if len(np.unique(is_error)) > 1:
            precision, recall, _ = precision_recall_curve(is_error, r_total)
            ap = average_precision_score(is_error, r_total)
            ax3.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {ap:.3f})')
            ax3.set_xlabel('Recall')
            ax3.set_ylabel('Precision')
            ax3.set_title('Risk-Based Error Detection PR Curve')
            ax3.legend(loc='lower left')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "risk_error_analysis.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Evaluator] Risk-error analysis plot saved to {save_path}")
    
    def save_predictions(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ):
        """保存预测结果到JSON"""
        if save_path is None:
            save_path = self.save_dir / "predictions.json"
        
        predictions = []
        
        for i, (pred, label, prob, path) in enumerate(zip(
            np.concatenate(self.all_preds),
            np.concatenate(self.all_labels),
            np.concatenate(self.all_probs),
            self.all_paths,
        )):
            pred_dict = {
                'index': i,
                'path': path,
                'true_label': int(label) if not self.config.multi_label else label.tolist(),
                'pred_label': int(pred) if not self.config.multi_label else pred.tolist(),
                'probabilities': prob.tolist() if isinstance(prob, np.ndarray) else prob,
            }
            
            # 添加幻觉风险 (如果有)
            if i < len(self.all_hallucination_risks):
                pred_dict['hallucination_risk'] = float(self.all_hallucination_risks[i])
            
            # 添加风险组件 (如果有)
            if i < len(self.all_risk_components['r_total']):
                pred_dict['r_total'] = float(self.all_risk_components['r_total'][i])
                if i < len(self.all_risk_components['r_state']):
                    pred_dict['r_state'] = float(self.all_risk_components['r_state'][i])
                if i < len(self.all_risk_components['r_scan']):
                    pred_dict['r_scan'] = float(self.all_risk_components['r_scan'][i])
                if i < len(self.all_risk_components['r_task']):
                    pred_dict['r_task'] = float(self.all_risk_components['r_task'][i])
            
            predictions.append(pred_dict)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        print(f"[Evaluator] Predictions saved to {save_path}")
    
    def run_evaluation(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        运行完整评估流程
        
        Args:
            save_plots: 是否保存可视化图表
        
        Returns:
            results: 评估结果
        """
        print("\n" + "=" * 60)
        print("Starting Evaluation")
        print("=" * 60)
        
        # 执行评估
        results = self.evaluate()
        
        # 生成报告
        report = self.generate_report(results)
        print(report)
        
        # 保存报告
        report_path = self.save_dir / "evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[Evaluator] Report saved to {report_path}")
        
        # 生成风险分析报告
        if self.config.compute_risk_metrics:
            risk_report_path = self.save_dir / "risk_analysis_report.md"
            self.generate_risk_analysis_report(results, risk_report_path)
        
        # 可视化
        if save_plots:
            all_labels = np.concatenate(self.all_labels)
            all_probs = np.concatenate(self.all_probs)
            
            # 混淆矩阵
            if self.config.plot_confusion_matrix:
                self.plot_confusion_matrix(results)
            
            # ROC曲线
            if self.config.plot_roc_curve:
                self.plot_roc_curve(all_labels, all_probs)
            
            # PR曲线
            if self.config.plot_pr_curve:
                self.plot_pr_curve(all_labels, all_probs)
            
            # 幻觉风险分布
            if len(self.all_hallucination_risks) > 0:
                risks = np.concatenate(self.all_hallucination_risks)
                self.plot_hallucination_risk_distribution(risks)
            
            # 风险-错误分析图
            if self.config.compute_risk_metrics:
                self.plot_risk_error_analysis(results)
        
        # 保存预测结果
        if self.config.save_predictions:
            self.save_predictions(results)
        
        # 保存JSON结果
        results_json_path = self.save_dir / "evaluation_results.json"
        
        # 转换为可JSON序列化的格式
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {k: v for k, v in value.items() 
                                             if isinstance(v, (int, float, str))}
            else:
                try:
                    serializable_results[key] = value
                except:
                    pass
        
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"[Evaluator] Results saved to {results_json_path}")
        print("=" * 60 + "\n")
        
        return results


# =============================================================================
# 评估工厂
# =============================================================================

class EvaluatorFactory:
    """评估器工厂"""
    
    @staticmethod
    def create_evaluator(
        model: nn.Module,
        data_loader: DataLoader,
        task: str = "classification",
        num_classes: int = 2,
        save_dir: str = "./results",
        compute_risk_metrics: bool = False,
        **kwargs,
    ) -> 'Evaluator':
        """
        创建评估器
        
        Args:
            model: 模型
            data_loader: 数据加载器
            task: 任务类型
            num_classes: 类别数
            save_dir: 保存路径
            compute_risk_metrics: 是否计算风险指标
            **kwargs: 其他参数
        
        Returns:
            evaluator: Evaluator实例
        """
        config = EvaluatorConfig(
            num_classes=num_classes,
            task=task,
            save_dir=save_dir,
            compute_risk_metrics=compute_risk_metrics,
            **kwargs,
        )
        
        return Evaluator(
            model=model,
            data_loader=data_loader,
            config=config,
            **kwargs,
        )


def main():
    """主函数 - 演示用法"""
    print("[Evaluator] This module provides the Evaluator class.")
    print("[Evaluator] Usage: from src.evaluator import Evaluator, EvaluatorConfig")


if __name__ == "__main__":
    main()