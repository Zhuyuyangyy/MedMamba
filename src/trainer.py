"""
MedMamba训练器模块

功能:
- 支持多GPU分布式训练 (DDP)
- 早停机制
- 学习率调度
- 训练指标: loss, accuracy, F1, AUC
- Checkpoint保存 (best/last)

作者: MedMamba Team
"""

import os
import time
import json
import copy
import random
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR
from torch.amp import autocast
import torch.distributed as dist

# 尝试导入wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# 尝试导入tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# =============================================================================
# 配置
# =============================================================================

@dataclass
class TrainerConfig:
    """训练器配置"""
    # 模型
    model_name: str = "MedMamba"
    model_version: str = "v2"
    
    # 训练
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # 优化
    optimizer: str = "AdamW"
    scheduler: str = "cosine"  # cosine / step / plateau
    step_size: int = 30  # for StepLR
    gamma: float = 0.1  # for StepLR
    plateau_patience: int = 10  # for ReduceLROnPlateau
    
    # AMP
    use_amp: bool = True
    gradient_clip: float = 1.0
    
    # 早停
    early_stop_patience: int = 15
    early_stop_min_delta: float = 0.001
    
    # 正则化
    dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Mixup/Cutmix
    use_mixup: bool = False
    mixup_alpha: float = 1.0
    cutmix_alpha: float = 1.0
    
    # 日志
    log_interval: int = 10
    eval_interval: int = 1
    save_dir: str = "./checkpoints"
    
    # 分布式
    local_rank: int = 0
    world_size: int = 1
    
    # wandb
    use_wandb: bool = False
    wandb_project: str = "MedMamba"
    wandb_run_name: Optional[str] = None


# =============================================================================
# 指标计算
# =============================================================================

class MetricsCalculator:
    """
    训练指标计算器
    
    计算: loss, accuracy, precision, recall, F1, AUC
    """
    
    def __init__(self, num_classes: int = 2, multi_label: bool = False):
        """
        Args:
            num_classes: 类别数
            multi_label: 是否多标签分类
        """
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.reset()
    
    def reset(self):
        """重置所有统计量"""
        self.total_loss = 0.0
        self.count = 0
        self.correct = 0
        self.total = 0
        
        # 用于F1/AUC计算
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor):
        """
        更新统计量
        
        Args:
            loss: 批次损失
            logits: 模型输出 [B, num_classes]
            labels: 标签 [B]
        """
        self.total_loss += loss * logits.size(0)
        self.count += logits.size(0)
        
        # 准确率
        if self.multi_label:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            self.correct += (preds == labels).all(dim=1).sum().item()
        else:
            preds = logits.argmax(dim=-1)
            self.correct += (preds == labels).sum().item()
        
        self.total += labels.size(0)
        
        # 保存预测和标签用于F1/AUC
        if self.multi_label:
            probs = torch.sigmoid(logits)
            self.all_probs.append(probs.detach().cpu())
            self.all_labels.append(labels.detach().cpu())
        else:
            self.all_probs.append(F.softmax(logits, dim=-1).detach().cpu())
            self.all_labels.append(labels.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            metrics: 指标字典
        """
        metrics = {}
        
        # Loss
        metrics['loss'] = self.total_loss / self.count if self.count > 0 else 0
        
        # Accuracy
        metrics['accuracy'] = self.correct / self.total if self.total > 0 else 0
        
        # F1/AUC (需要足够的数据)
        if len(self.all_labels) > 0:
            all_labels = torch.cat(self.all_labels)
            all_probs = torch.cat(self.all_probs)
            
            if not self.multi_label and self.num_classes == 2:
                # 二分类
                probs_pos = all_probs[:, 1]
                
                # AUC
                try:
                    from sklearn.metrics import roc_auc_score
                    if len(all_labels.unique()) > 1:
                        metrics['auc'] = roc_auc_score(all_labels, probs_pos)
                    else:
                        metrics['auc'] = 0.5
                except ImportError:
                    metrics['auc'] = 0.0
                
                # F1
                try:
                    from sklearn.metrics import f1_score
                    preds = (probs_pos > 0.5).long()
                    metrics['f1'] = f1_score(all_labels, preds, average='binary')
                except ImportError:
                    metrics['f1'] = 0.0
            else:
                # 多分类/多标签
                try:
                    from sklearn.metrics import roc_auc_score, f1_score
                    
                    if self.multi_label:
                        # 多标签
                        metrics['auc'] = roc_auc_score(
                            all_labels, all_probs, average='macro', multi_label='average'
                        )
                        preds = (all_probs > 0.5).long()
                        metrics['f1'] = f1_score(
                            all_labels, preds, average='macro', zero_division=0
                        )
                    else:
                        # 多分类
                        preds = all_probs.argmax(dim=-1)
                        metrics['auc'] = roc_auc_score(
                            all_labels, all_probs, multi_class='ovr', average='macro'
                        )
                        metrics['f1'] = f1_score(
                            all_labels, preds, average='macro', zero_division=0
                        )
                except ImportError:
                    metrics['auc'] = 0.0
                    metrics['f1'] = 0.0
        
        return metrics


# =============================================================================
# 训练器
# =============================================================================

class Trainer:
    """
    MedMamba训练器
    
    支持:
    - 单GPU训练
    - 多GPU分布式训练 (DDP)
    - 早停
    - 学习率调度
    - AMP混合精度训练
    - Mixup/Cutmix增强
    - Wandb/Tensorboard日志
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainerConfig,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        
        # 多GPU设置
        self.is_distributed = config.world_size > 1
        self.local_rank = config.local_rank
        
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank]
            )
        
        # 优化器
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        # 调度器
        if scheduler is None:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = scheduler
        
        # AMP
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None
        
        # Mixup/Cutmix
        if config.use_mixup:
            from .augmentation import Mixup, Cutmix
            self.mixup = Mixup(alpha=config.mixup_alpha, prob=0.5)
            self.cutmix = Cutmix(alpha=config.cutmix_alpha, prob=0.5)
        else:
            self.mixup = None
            self.cutmix = None
        
        # 指标
        self.train_metrics = MetricsCalculator()
        self.val_metrics = MetricsCalculator()
        
        # 早停
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # 日志
        self.writer = None
        if HAS_TENSORBOARD and config.local_rank == 0:
            os.makedirs(config.save_dir, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(config.save_dir, "runs"))
        
        # Wandb
        self.wandb_run = None
        if HAS_WANDB and config.use_wandb and config.local_rank == 0:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"{config.model_name}_{config.model_version}",
                config=asdict(config),
            )
            self.wandb_run = wandb
        
        # Epoch计数
        self.current_epoch = 0
        self.global_step = 0
        
        # 保存路径
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': [],
            'lr': [],
        }
    
    def _create_optimizer(self) -> Optimizer:
        """创建优化器"""
        if self.config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> _LRScheduler:
        """创建学习率调度器"""
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma,
            )
        else:
            # 默认使用CosineAnnealing
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            metrics: 训练指标
        """
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Mixup/Cutmix
            if self.mixup is not None and random.random() < 0.5:
                images, labels_a, labels_b, lam = self.mixup(images, labels)

                # 前向传播
                self.optimizer.zero_grad()

                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(images)
                        if isinstance(outputs, dict):
                            logits = outputs.get('logits') or outputs.get('cls_logits')
                        else:
                            logits = outputs
                        loss_a = self.criterion(logits, labels_a)
                        loss_b = self.criterion(logits, labels_b)
                        loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits') or outputs.get('cls_logits')
                    else:
                        logits = outputs
                    loss_a = self.criterion(logits, labels_a)
                    loss_b = self.criterion(logits, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    self.optimizer.step()

                # 更新指标
                self.train_metrics.update(loss.item(), logits, labels_a)
            else:
                # 标准训练
                self.optimizer.zero_grad()
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(images)
                        
                        if isinstance(outputs, dict):
                            logits = outputs.get('logits') or outputs.get('cls_logits')
                        else:
                            logits = outputs
                        
                        loss = self.criterion(logits, labels)
                else:
                    outputs = self.model(images)
                    
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits') or outputs.get('cls_logits')
                    else:
                        logits = outputs
                    
                    loss = self.criterion(logits, labels)
                
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                    self.optimizer.step()
                
                # 更新指标
                if isinstance(outputs, dict):
                    logits = outputs.get('logits') or outputs.get('cls_logits')
                else:
                    logits = outputs
                self.train_metrics.update(loss.item(), logits, labels)
            
            epoch_loss += loss.item()
            
            # 日志
            if batch_idx % self.config.log_interval == 0 and self.config.local_rank == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.4f}")
            
            self.global_step += 1
        
        # 计算epoch指标
        metrics = self.train_metrics.compute()
        metrics['loss'] = epoch_loss / len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型
        
        Returns:
            metrics: 验证指标
        """
        self.model.eval()
        self.val_metrics.reset()
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
            else:
                outputs = self.model(images)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits') or outputs.get('cls_logits')
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
            
            self.val_metrics.update(loss.item(), logits, labels)
        
        return self.val_metrics.compute()
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        is_last: bool = False,
    ):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            is_best: 是否为最佳模型
            is_last: 是否为最新模型
        """
        if self.config.local_rank != 0:
            return
        
        # 获取模型状态
        if self.is_distributed:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'history': self.history,
        }
        
        # 保存最新检查点
        if is_last:
            last_path = self.save_dir / f"{self.config.model_name}_last.pt"
            torch.save(checkpoint, last_path)
            print(f"[Trainer] Last checkpoint saved: {last_path}")
        
        # 保存最佳检查点
        if is_best:
            best_path = self.save_dir / f"{self.config.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"[Trainer] Best checkpoint saved: {best_path}")
        
        # 按epoch保存
        epoch_path = self.save_dir / f"{self.config.model_name}_epoch{epoch}.pt"
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        
        Returns:
            epoch: 恢复的epoch
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.history = checkpoint.get('history', self.history)
        
        print(f"[Trainer] Loaded checkpoint from epoch {self.current_epoch}")
        
        return self.current_epoch
    
    def early_stopping(self, val_loss: float) -> bool:
        """
        检查早停条件
        
        Args:
            val_loss: 验证损失
        
        Returns:
            should_stop: 是否应该停止训练
        """
        if val_loss < self.best_val_loss - self.config.early_stop_min_delta:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            
            # 保存最佳模型
            if self.is_distributed:
                self.best_model_state = copy.deepcopy(self.model.module.state_dict())
            else:
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            return False
        else:
            self.early_stop_counter += 1
            
            if self.early_stop_counter >= self.config.early_stop_patience:
                print(f"[Trainer] Early stopping triggered after {self.early_stop_counter} epochs without improvement")
                return True
            
            return False
    
    def log_metrics(self, phase: str, metrics: Dict[str, float], epoch: int):
        """
        记录指标到日志
        
        Args:
            phase: 阶段 (train/val)
            metrics: 指标字典
            epoch: 当前epoch
        """
        if self.config.local_rank != 0:
            return
        
        # Tensorboard
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{phase}/{name}", value, epoch)
        
        # Wandb
        if self.wandb_run:
            log_dict = {f"{phase}_{k}": v for k, v in metrics.items()}
            log_dict['epoch'] = epoch
            self.wandb_run.log(log_dict)
    
    def train(self, start_epoch: int = 0) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            start_epoch: 起始epoch (用于恢复训练)
        
        Returns:
            history: 训练历史
        """
        print("\n" + "=" * 60)
        print(f"Starting Training: {self.config.model_name} {self.config.model_version}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Learning Rate: {self.config.learning_rate}")
        print(f"AMP: {self.config.use_amp}")
        print(f"Early Stop: {self.config.early_stop_patience}")
        print("=" * 60 + "\n")
        
        for epoch in range(start_epoch, self.config.epochs):
            epoch_start = time.time()
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            if self.val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.evaluate()
                
                # 学习率调度
                if self.config.scheduler == "plateau":
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 记录历史
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics.get('accuracy', 0))
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics.get('accuracy', 0))
                self.history['val_f1'].append(val_metrics.get('f1', 0))
                self.history['val_auc'].append(val_metrics.get('auc', 0))
                self.history['lr'].append(current_lr)
                
                # 日志
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch+1}/{self.config.epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics.get('accuracy', 0):.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics.get('accuracy', 0):.4f}")
                print(f"  Val F1: {val_metrics.get('f1', 0):.4f}, AUC: {val_metrics.get('auc', 0):.4f}")
                print(f"  LR: {current_lr:.6f}")
                
                # 记录指标
                self.log_metrics("train", train_metrics, epoch)
                self.log_metrics("val", val_metrics, epoch)
                
                # 保存检查点
                is_best = val_metrics['loss'] < self.best_val_loss
                is_last = (epoch + 1) == self.config.epochs
                self.save_checkpoint(epoch, is_best=is_best, is_last=is_last)
                
                # 早停检查
                if self.early_stopping(val_metrics['loss']):
                    print(f"[Trainer] Stopping at epoch {epoch+1}")
                    break
            else:
                # 无验证集
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics.get('accuracy', 0))
                self.history['lr'].append(current_lr)
                
                # 调度
                self.scheduler.step()
                
                print(f"\nEpoch {epoch+1}/{self.config.epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  LR: {current_lr:.6f}")
                
                self.log_metrics("train", train_metrics, epoch)
                
                if (epoch + 1) % self.config.eval_interval == 0:
                    is_last = (epoch + 1) == self.config.epochs
                    self.save_checkpoint(epoch, is_best=False, is_last=is_last)
            
            self.current_epoch = epoch + 1
        
        # 保存训练历史
        history_path = self.save_dir / f"{self.config.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 清理
        if self.writer:
            self.writer.close()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"History saved to {history_path}")
        print("=" * 60)
        
        return self.history


# =============================================================================
# 分布式训练辅助
# =============================================================================

def setup_distributed() -> Tuple[int, int]:
    """
    设置分布式训练环境
    
    Returns:
        (local_rank, world_size)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        local_rank = 0
        world_size = 1
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    
    return local_rank, world_size


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """主函数 - 演示用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MedMamba Trainer")
    parser.add_argument("--config", type=str, default="", help="Config file path")
    args = parser.parse_args()
    
    print("[Trainer] This module provides the Trainer class.")
    print("[Trainer] Usage: from src.trainer import Trainer, TrainerConfig")


if __name__ == "__main__":
    main()