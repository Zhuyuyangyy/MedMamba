"""
MedMamba-Guard Training Script

支持MedMamba-Guard框架训练:
- MedMamba-Guard: CNN-SSM双分支 + CTM状态轨迹监控 + Cross-Scan一致性分析 + 任务冲突验证
- CTMMonitor可选训练 (如果CTM可学习)
- 训练时记录R_state, R_scan, R_task
- WandB扩展日志: ctm_metrics, risk_metrics, conflict_rate
- 早停条件: 基于val_risk_total (如果配置了)
- 保存best模型时同时保存CTM校准参数

使用方法:
    python train_medmamba_guard.py --model_type medmamba_guard --data_root ./data
    python train_medmamba_guard.py --model_type medmamba_guard --ctm_trainable --log_ctm_metrics
    python train_medmamba_guard.py --resume ./checkpoints/MedMambaGuard_best.pt

作者: MedMamba Team
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, asdict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR
from torch.amp import autocast
import torch.distributed as dist
import numpy as np

# 尝试导入tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# 尝试导入wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Set --use_wandb to enable.")

# 项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.medmamba import MedMambaV2, MedMambaV3, create_medmamba
from src.models.medmamba_guard import MedMambaGuard, create_medmamba_guard, LightMedMambaGuard, FullMedMambaGuard
from src.models.home_moe import ExpertDiversityLoss
from src.data.dataset import MedMambaDataset, DataLoaderFactory
from src.data.dataset_guard import create_dataset, ISIC2018Dataset, MedMNISTDataset
from src.data.augmentation import Mixup, Cutmix, get_augmentation_pipeline
from src.trainer import Trainer, TrainerConfig
from src.evaluator import Evaluator, EvaluatorConfig


# =============================================================================
# 配置
# =============================================================================

@dataclass
class TrainConfig:
    """训练配置"""
    # 模型
    model_type: str = "medmamba_guard"  # medmamba / medmamba_guard
    model_version: str = "v2"
    model_name: str = "MedMamba-Guard"
    d_model: int = 384
    n_layers: int = 12
    d_state: int = 16
    dropout: float = 0.1
    
    # CTM配置
    ctm_trainable: bool = False  # CTM模块是否可训练
    trajectory_window: int = 5
    delta_window: int = 3
    
    # Cross-Scan配置
    lambda_l2: float = 0.5
    lambda_cos: float = 0.5
    
    # 任务冲突配置
    enable_task_conflict_loss: bool = False
    task_conflict_weight: float = 0.1
    conflict_threshold: float = 0.4
    
    # 风险阈值
    risk_threshold: float = 0.7
    
    # 数据
    dataset_name: str = "isic2018"  # isic2018 / medmnist / synthetic
    data_root: str = "./data"
    img_size: int = 224
    in_channels: int = 3
    batch_size: int = 16
    num_workers: int = 4
    
    # 任务
    task: str = "classification"  # classification / segmentation
    num_classes: int = 2
    multi_label: bool = False
    
    # 训练
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    
    # 早停
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_loss"  # val_loss / val_risk_total / val_accuracy
    
    # 日志
    log_ctm_metrics: bool = True
    log_interval: int = 10
    save_dir: str = "./outputs"
    save_predictions: bool = True
    
    # WandB
    use_wandb: bool = False
    wandb_project: str = "MedMamba-Guard"
    wandb_run_name: Optional[str] = None
    
    # 分布式
    distributed: bool = False
    local_rank: int = 0
    
    # 恢复
    resume: Optional[str] = None


# =============================================================================
# 训练器
# =============================================================================

class MedMambaGuardTrainer:
    """MedMamba-Guard训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 模型类型
        self.is_guard = config.model_type == "medmamba_guard"
        
        # 优化器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 任务冲突损失
        self.task_conflict_criterion = nn.MSELoss() if config.enable_task_conflict_loss else None
        
        # WandB
        self.writer = None
        if config.use_wandb and HAS_WANDB:
            self._init_wandb()
        
        # 状态
        self.current_epoch = 0
        self.best_metric = float('inf') if "loss" in config.early_stopping_metric else 0.0
        self.best_risk_total = float('inf')
        self.patience_counter = 0
        self.history = {'train': [], 'val': []}
        
        # 保存路径
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # CTM可训练模式
        if config.ctm_trainable and self.is_guard:
            self._enable_ctm_training()
    
    def _create_optimizer(self) -> Optimizer:
        """创建优化器"""
        params = self.model.parameters()
        
        # 如果CTM可训练，单独设置学习率
        if self.config.ctm_trainable and self.is_guard:
            ctm_params = []
            non_ctm_params = []
            for name, p in self.model.named_parameters():
                if 'ctm' in name.lower():
                    ctm_params.append(p)
                else:
                    non_ctm_params.append(p)
            
            params = [
                {'params': non_ctm_params, 'lr': self.config.learning_rate},
                {'params': ctm_params, 'lr': self.config.learning_rate * 0.1},
            ]
        
        if self.config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """创建学习率调度器"""
        if self.config.scheduler.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler.lower() == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1,
            )
        elif self.config.scheduler.lower() == "none":
            return None
        else:
            return None
    
    def _init_wandb(self):
        """初始化WandB"""
        run_name = self.config.wandb_run_name or f"{self.config.model_type}_{int(time.time())}"
        
        wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config=asdict(self.config),
        )
        
        self.writer = wandb
    
    def _enable_ctm_training(self):
        """启用CTM模块训练"""
        if hasattr(self.model, 'ctm_monitor'):
            for param in self.model.ctm_monitor.parameters():
                param.requires_grad = True
            print("[Trainer] CTM module training enabled")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_conflict_loss = 0.0
        correct = 0
        total = 0
        
        # CTM指标累积
        ctm_metrics_sum = {
            'r_state': 0.0,
            'r_scan': 0.0,
            'r_task': 0.0,
            'r_entropy': 0.0,
            'r_total': 0.0,
        }
        ctm_count = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            with autocast(enabled=self.device == "cuda"):
                if self.is_guard:
                    outputs = self.model(images, mode='train', return_risk=True)
                    cls_logits = outputs['cls_logits']
                    
                    # 分类损失
                    cls_loss = self.criterion(cls_logits, labels)
                    loss = cls_loss
                    
                    # 任务冲突损失
                    if self.config.enable_task_conflict_loss and 'conflict_loss' in outputs:
                        conflict_loss = outputs['conflict_loss']
                        loss = loss + self.config.task_conflict_weight * conflict_loss
                        total_conflict_loss += conflict_loss.item()
                    
                    # 提取CTM指标
                    if self.config.log_ctm_metrics:
                        ctm_m = outputs.get('ctm_metrics', {})
                        if ctm_m:
                            for k in ['r_state', 'r_scan', 'r_task', 'r_entropy']:
                                if k in ctm_m:
                                    v = ctm_m[k]
                                    if isinstance(v, torch.Tensor):
                                        v = v.item()
                                    ctm_metrics_sum[k] += v
                            
                            # 计算R_total
                            r_total = self.model.hard_gating.compute_total_risk(
                                r_state=ctm_m.get('r_state', 0),
                                r_scan=ctm_m.get('r_scan', 0),
                                r_task=ctm_m.get('r_task', 0),
                                r_entropy=ctm_m.get('r_entropy', 0),
                            )
                            ctm_metrics_sum['r_total'] += r_total
                            ctm_count += 1
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        cls_logits = outputs.get('logits') or outputs.get('cls_logits', outputs.get('output'))
                    else:
                        cls_logits = outputs
                    loss = self.criterion(cls_logits, labels)
            
            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            
            preds = cls_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 日志
            if batch_idx % self.config.log_interval == 0:
                log_str = f"[Epoch {self.current_epoch}] [{batch_idx}/{len(self.train_loader)}] "
                log_str += f"Loss: {loss.item():.4f} "
                if self.is_guard and self.config.log_ctm_metrics:
                    log_str += f"R_total: {ctm_metrics_sum['r_total'] / max(ctm_count, 1):.4f}"
                print(log_str)
        
        # Epoch统计
        n = len(self.train_loader)
        avg_loss = total_loss / n
        avg_cls_loss = total_cls_loss / n
        avg_conflict_loss = total_conflict_loss / n if self.config.enable_task_conflict_loss else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        # CTM平均
        avg_ctm = {}
        if ctm_count > 0:
            for k, v in ctm_metrics_sum.items():
                avg_ctm[f'ctm_{k}'] = v / ctm_count
        
        # WandB日志
        if self.writer:
            self.writer.log({
                'train/loss': avg_loss,
                'train/cls_loss': avg_cls_loss,
                'train/accuracy': accuracy,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                **avg_ctm,
            })
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'conflict_loss': avg_conflict_loss,
            'accuracy': accuracy,
            **avg_ctm,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # CTM指标累积
        ctm_metrics_sum = {
            'r_state': 0.0,
            'r_scan': 0.0,
            'r_task': 0.0,
            'r_entropy': 0.0,
            'r_total': 0.0,
        }
        ctm_count = 0
        
        all_preds = []
        all_labels = []
        all_risks = []
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            if self.is_guard:
                outputs = self.model(images, mode='eval', return_risk=True)
                cls_logits = outputs['cls_logits']
                loss = self.criterion(cls_logits, labels)
                
                # CTM指标
                ctm_m = outputs.get('ctm_metrics', {})
                if ctm_m:
                    for k in ['r_state', 'r_scan', 'r_task', 'r_entropy']:
                        if k in ctm_m:
                            v = ctm_m[k]
                            if isinstance(v, torch.Tensor):
                                v = v.item()
                            ctm_metrics_sum[k] += v
                    
                    r_total = self.model.hard_gating.compute_total_risk(
                        r_state=ctm_m.get('r_state', 0),
                        r_scan=ctm_m.get('r_scan', 0),
                        r_task=ctm_m.get('r_task', 0),
                        r_entropy=ctm_m.get('r_entropy', 0),
                    )
                    ctm_metrics_sum['r_total'] += r_total
                    all_risks.append(r_total)
                    ctm_count += 1
            else:
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    cls_logits = outputs.get('logits') or outputs.get('cls_logits', outputs.get('output'))
                else:
                    cls_logits = outputs
                loss = self.criterion(cls_logits, labels)
                all_risks.append(0.0)
            
            total_loss += loss.item()
            preds = cls_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        # 计算指标
        n = len(self.val_loader)
        avg_loss = total_loss / n
        accuracy = correct / total if total > 0 else 0.0
        
        # CTM平均
        avg_ctm = {}
        if ctm_count > 0:
            for k, v in ctm_metrics_sum.items():
                avg_ctm[f'val_ctm_{k}'] = v / ctm_count
        
        # 计算AUC
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_risks_np = np.array(all_risks)
        
        try:
            from sklearn.metrics import roc_auc_score
            if self.config.num_classes == 2:
                probs = F.softmax(torch.from_numpy(all_preds.astype(np.float32)), dim=-1).numpy()
                if probs.ndim > 1:
                    probs = probs[:, 1]
                auc = roc_auc_score(all_labels, probs)
            else:
                auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
        except:
            auc = 0.0
        
        # WandB日志
        if self.writer:
            self.writer.log({
                'val/loss': avg_loss,
                'val/accuracy': accuracy,
                'val/auc': auc,
                **avg_ctm,
            })
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'risk_total': avg_ctm.get('val_ctm_r_total', 0.0),
            **avg_ctm,
        }
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': asdict(self.config),
            'history': self.history,
        }
        
        # 如果是best模型，保存CTM校准参数
        if is_best and self.is_guard:
            if hasattr(self.model, 'ctm_monitor'):
                checkpoint['ctm_calibration'] = {
                    'calibration_params': self.model.ctm_monitor.get_calibration_params() 
                    if hasattr(self.model.ctm_monitor, 'get_calibration_params') else {},
                }
        
        torch.save(checkpoint, path)
        print(f"[Trainer] Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint.get('history', {'train': [], 'val': []})
        
        print(f"[Trainer] Checkpoint loaded from {path}")
        print(f"[Trainer] Resuming from epoch {self.current_epoch}")
    
    def train(self) -> Dict[str, Any]:
        """执行完整训练流程"""
        print("\n" + "=" * 60)
        print(f"Training {self.config.model_type}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Val samples: {len(self.val_loader.dataset)}")
        print("=" * 60 + "\n")
        
        # 恢复训练
        if self.config.resume:
            self.load_checkpoint(self.config.resume)
        
        # 早停
        early_stop = False
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            self.history['train'].append(train_metrics)
            
            # 验证
            val_metrics = {}
            if self.val_loader:
                val_metrics = self.validate()
                self.history['val'].append(val_metrics)
                
                # 打印进度
                print(f"\n[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
                print(f"[Epoch {epoch}] Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
                
                if self.is_guard and self.config.log_ctm_metrics:
                    print(f"[Epoch {epoch}] R_state: {train_metrics.get('ctm_r_state', 0):.4f} | R_scan: {train_metrics.get('ctm_r_scan', 0):.4f} | R_task: {train_metrics.get('ctm_r_task', 0):.4f}")
                    print(f"[Epoch {epoch}] Val R_total: {val_metrics.get('val_ctm_r_total', 0):.4f}")
            else:
                print(f"\n[Epoch {epoch}] Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            
            # 学习率调整
            if self.scheduler:
                self.scheduler.step()
            
            # 早停检查
            if self.val_loader:
                if self.config.early_stopping_metric == "val_risk_total":
                    current_metric = val_metrics.get('val_ctm_r_total', float('inf'))
                    check_best = current_metric < self.best_metric
                    metric_name = "risk_total"
                elif self.config.early_stopping_metric == "val_accuracy":
                    current_metric = val_metrics.get('accuracy', 0.0)
                    check_best = current_metric > self.best_metric
                    metric_name = "accuracy"
                else:
                    current_metric = val_metrics.get('loss', float('inf'))
                    check_best = current_metric < self.best_metric
                    metric_name = "loss"
                
                if check_best:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    
                    # 保存best模型
                    best_path = self.save_dir / f"{self.config.model_type}_best.pt"
                    self.save_checkpoint(str(best_path), is_best=True)
                    print(f"[Epoch {epoch}] New best {metric_name}: {current_metric:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"[Epoch {epoch}] No improvement. Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\n[Trainer] Early stopping triggered at epoch {epoch}")
                    early_stop = True
            else:
                # 无验证集时每个epoch都保存
                if (epoch + 1) % 10 == 0:
                    ckpt_path = self.save_dir / f"{self.config.model_type}_epoch{epoch+1}.pt"
                    self.save_checkpoint(str(ckpt_path))
            
            # 保存最后一个epoch
            if epoch == self.config.epochs - 1:
                last_path = self.save_dir / f"{self.config.model_type}_last.pt"
                self.save_checkpoint(str(last_path))
            
            if early_stop:
                break
            
            print()
        
        # 保存训练历史
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"[Trainer] Training history saved to {history_path}")
        
        # WandB关闭
        if self.writer:
            self.writer.finish()
        
        return {
            'best_metric': self.best_metric,
            'history': self.history,
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_model(config: TrainConfig) -> nn.Module:
    """创建模型"""
    if config.model_type == "medmamba_guard":
        model = create_medmamba_guard(
            d_model=config.d_model,
            n_layers=config.n_layers,
            d_state=config.d_state,
            num_classes=config.num_classes,
            dropout=config.dropout,
            trajectory_window=config.trajectory_window,
            delta_window=config.delta_window,
            lambda_l2=config.lambda_l2,
            lambda_cos=config.lambda_cos,
            conflict_threshold=config.conflict_threshold,
            theta_high=config.risk_threshold,
            use_ctm=True,
            use_cross_scan=True,
            use_task_validator=config.enable_task_conflict_loss,
        )
    elif config.model_type == "medmamba":
        model = create_medmamba(
            d_model=config.d_model,
            n_layers=config.n_layers,
            d_state=config.d_state,
            num_classes=config.num_classes,
            dropout=config.dropout,
            version=config.model_version,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    return model


def create_dataloaders(config: TrainConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    """创建数据加载器"""
    # 创建数据集
    train_dataset = create_dataset(
        dataset_name=config.dataset_name,
        data_root=config.data_root,
        split='train',
        img_size=config.img_size,
        transform=None,
    )
    
    val_dataset = create_dataset(
        dataset_name=config.dataset_name,
        data_root=config.data_root,
        split='val',
        img_size=config.img_size,
        transform=None,
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MedMamba-Guard Training")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="medmamba_guard",
                        choices=["medmamba", "medmamba_guard"],
                        help="Model type")
    parser.add_argument("--d_model", type=int, default=384, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--d_state", type=int, default=16, help="State dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # CTM参数
    parser.add_argument("--ctm_trainable", action="store_true",
                        help="Enable CTM module training")
    parser.add_argument("--trajectory_window", type=int, default=5,
                        help="CTM trajectory window")
    parser.add_argument("--delta_window", type=int, default=3,
                        help="CTM delta window")
    
    # Cross-Scan参数
    parser.add_argument("--lambda_l2", type=float, default=0.5,
                        help="Cross-Scan L2 weight")
    parser.add_argument("--lambda_cos", type=float, default=0.5,
                        help="Cross-Scan cosine weight")
    
    # 任务冲突参数
    parser.add_argument("--enable_task_conflict_loss", action="store_true",
                        help="Enable task conflict loss")
    parser.add_argument("--task_conflict_weight", type=float, default=0.1,
                        help="Task conflict loss weight")
    parser.add_argument("--conflict_threshold", type=float, default=0.4,
                        help="Conflict threshold")
    
    # 风险阈值
    parser.add_argument("--risk_threshold", type=float, default=0.7,
                        help="Hard gating risk threshold")
    
    # 数据参数
    parser.add_argument("--dataset_name", type=str, default="isic2018",
                        choices=["isic2018", "medmnist", "synthetic"],
                        help="Dataset name")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Data root directory")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers")
    
    # 任务参数
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "segmentation"],
                        help="Task type")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adam", "sgd"],
                        help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "none"],
                        help="Scheduler")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Gradient clipping")
    
    # 早停参数
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--early_stopping_metric", type=str, default="val_loss",
                        choices=["val_loss", "val_risk_total", "val_accuracy"],
                        help="Early stopping metric")
    
    # 日志参数
    parser.add_argument("--log_ctm_metrics", action="store_true",
                        help="Log CTM metrics to WandB")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log interval")
    parser.add_argument("--save_dir", type=str, default="./outputs",
                        help="Save directory")
    
    # WandB参数
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use WandB for logging")
    parser.add_argument("--wandb_project", type=str, default="MedMamba-Guard",
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")
    
    # 恢复参数
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 转换为配置
    config = TrainConfig(
        model_type=args.model_type,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        dropout=args.dropout,
        ctm_trainable=args.ctm_trainable,
        trajectory_window=args.trajectory_window,
        delta_window=args.delta_window,
        lambda_l2=args.lambda_l2,
        lambda_cos=args.lambda_cos,
        enable_task_conflict_loss=args.enable_task_conflict_loss,
        task_conflict_weight=args.task_conflict_weight,
        conflict_threshold=args.conflict_threshold,
        risk_threshold=args.risk_threshold,
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task=args.task,
        num_classes=args.num_classes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        gradient_clip=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        log_ctm_metrics=args.log_ctm_metrics,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        resume=args.resume,
    )
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Using device: {device}")
    
    # 创建模型
    print(f"[Main] Creating {config.model_type} model...")
    model = create_model(config)
    model = model.to(device)
    
    # 打印模型参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Main] Model parameters: {n_params:,}")
    
    # 创建数据加载器
    print(f"[Main] Loading dataset: {config.dataset_name}")
    train_loader, val_loader = create_dataloaders(config)
    
    # 创建训练器
    trainer = MedMambaGuardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    
    # 训练
    results = trainer.train()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best metric: {results['best_metric']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()