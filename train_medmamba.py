"""
MedMamba Training Script - 完整训练脚手架

支持:
- MedMamba V1/V2/V3
- 分类任务 / 分割任务
- 多标签分类
- CTM幻觉检测
- HoME-MoE专家多样性训练
- 多GPU分布式训练 (DDP)
- 早停机制
- Wandb/Tensorboard日志
- 从checkpoint恢复训练

使用方法:
    # 单GPU训练
    python train_medmamba.py --model v2 --task classification --data_root ./data
    
    # 多GPU分布式训练
    python -m torch.distributed.launch --nproc_per_node=4 train_medmamba.py --model v2
    
    # 从checkpoint恢复
    python train_medmamba.py --resume ./checkpoints/MedMamba_best.pt
    
    # 使用Wandb
    python train_medmamba.py --use_wandb --wandb_project MedMamba

作者: MedMamba Team
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
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
from src.models.home_moe import ExpertDiversityLoss
from src.data.dataset import MedMambaDataset, DataLoaderFactory
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
    model_version: str = "v2"
    model_name: str = "MedMamba"
    d_model: int = 384
    n_layers: int = 12
    d_state: int = 16
    dropout: float = 0.1
    
    # 数据
    data_root: str = "./data"
    img_size: int = 224
    in_channels: int = 3
    batch_size: int = 16
    num_workers: int = 4
    
    # 任务
    task: str = "classification"
    num_classes: int = 2
    num_diseases: int = 8
    multi_label: bool = False
    class_names: Optional[List[str]] = None
    
    # 训练
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # 优化
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    step_size: int = 30
    gamma: float = 0.1
    
    # AMP
    use_amp: bool = True
    gradient_clip: float = 1.0
    
    # 早停
    early_stop_patience: int = 15
    early_stop_min_delta: float = 0.001
    
    # Mixup/Cutmix
    use_mixup: bool = False
    mixup_alpha: float = 1.0
    cutmix_alpha: float = 1.0
    
    # MoE
    use_moe: bool = True
    num_experts: int = 4
    moe_loss_weight: float = 0.1
    
    # CTM
    use_ctm: bool = True
    ctm_loss_weight: float = 0.05
    
    # CT窗宽窗位
    use_ct_window: bool = False
    
    # 日志
    log_interval: int = 10
    eval_interval: int = 1
    save_dir: str = "./checkpoints"
    
    # 分布式
    local_rank: int = 0
    world_size: int = 1
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = "MedMamba"
    wandb_run_name: Optional[str] = None
    
    # 恢复
    resume: Optional[str] = None
    
    # 设备
    device: str = "cuda"
    seed: int = 42


# =============================================================================
# 医学影像数据集
# =============================================================================

class MedicalImageDataset(Dataset):
    """
    医学影像数据集
    
    从本地文件夹加载CT/MRI/X-ray图像
    目录结构: data_root/class_name/image.png
    """
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.nii', '.nii.gz'}
    
    def __init__(
        self,
        data_root: str,
        img_size: int = 224,
        task: str = "classification",
        num_classes: int = 2,
        mode: str = "train",
        transform: Optional[callable] = None,
        cache_data: bool = False,
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.task = task
        self.num_classes = num_classes
        self.mode = mode
        self.transform = transform
        self.cache_data = cache_data
        self.cache = {}
        
        # 加载样本
        self.samples = self._load_samples()
        
        # 默认变换
        self.default_transform = self._get_default_transform()
    
    def _load_samples(self) -> List[Dict]:
        """加载数据样本"""
        samples = []
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        # 遍历类别目录
        for class_idx, class_dir in enumerate(sorted(self.data_root.iterdir())):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    samples.append({
                        'path': str(img_path),
                        'label': class_idx,
                        'class_name': class_name,
                    })
        
        return samples
    
    def _get_default_transform(self):
        """获取默认变换"""
        if self.mode == "train":
            pipeline = get_augmentation_pipeline(mode="train")
            return pipeline
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self.cache:
            return self.cache[idx]
        
        sample = self.samples[idx]
        
        # 加载图像
        try:
            import cv2
            image = cv2.imread(sample['path'])
            if image is None:
                image = cv2.imread(sample['path'], cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            image = torch.zeros(3, self.img_size, self.img_size)
            return {'image': image, 'label': torch.tensor(sample['label'])}
        
        # 调整尺寸
        import cv2
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # 归一化到[0, 1]
        image = image.astype(np.float32) / 255.0
        
        # 应用变换
        if self.default_transform:
            image = self.default_transform(image)
        elif self.transform:
            image = self.transform(image)
        
        result = {
            'image': image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'path': sample['path'],
        }
        
        if self.cache_data:
            self.cache[idx] = result
        
        return result


# 为了兼容性保留别名
class MockMedicalDataset(MedicalImageDataset):
    """模拟数据集 (用于演示)"""
    
    def __init__(self, num_samples: int = 1000, img_size: int = 224, task: str = "classification"):
        self.num_samples = num_samples
        self.img_size = img_size
        self.task = task
        self.cache = {}
        
        # 随机生成样本
        self.samples = [
            {'path': f'fake_{i}.png', 'label': i % 2, 'class_name': f'class_{i % 2}'}
            for i in range(num_samples)
        ]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self.cache:
            return self.cache[idx]
        
        sample = self.samples[idx]
        
        # 随机图像
        image = torch.randn(3, self.img_size, self.img_size)
        
        if self.task == "classification":
            label = torch.randint(0, 2, (1,)).squeeze(0)
        else:
            label = torch.randint(0, 8, (self.img_size, self.img_size))
            label = label.long()
        
        result = {'image': image, 'label': label}
        
        if self.cache_data:
            self.cache[idx] = result
        
        return result


# =============================================================================
# 损失函数
# =============================================================================

class MedMambaLoss(nn.Module):
    """
    MedMamba统一损失函数
    
    组合:
    - 任务损失 (交叉熵/Dice)
    - CTM幻觉损失 (可选)
    - MoE专家多样性损失 (可选)
    """
    
    def __init__(
        self,
        task: str = "classification",
        num_classes: int = 2,
        multi_label: bool = False,
        use_ctm: bool = False,
        ctm_weight: float = 0.05,
        use_moe: bool = False,
        moe_weight: float = 0.1,
    ):
        super().__init__()
        self.task = task
        self.use_ctm = use_ctm
        self.use_moe = use_moe
        self.ctm_weight = ctm_weight
        self.moe_weight = moe_weight
        
        if task == "classification":
            if multi_label:
                self.task_loss = nn.BCEWithLogitsLoss()
            else:
                self.task_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            self.task_loss = DiceLoss(num_classes)
        
        if use_moe:
            self.moe_loss = ExpertDiversityLoss(target_entropy=0.9)
        else:
            self.moe_loss = None
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算损失"""
        losses = {}
        
        # 任务损失
        if self.task == "classification":
            logits = predictions.get("logits") or predictions.get("cls_logits")
            task_loss = self.task_loss(logits, targets)
        else:
            seg_logits = predictions.get("seg_logits")
            task_loss = self.task_loss(seg_logits, targets)
        
        losses["task"] = task_loss
        
        # CTM损失
        if self.use_ctm and "hallucination_risk" in predictions:
            h_risk = predictions["hallucination_risk"]
            ctm_loss = h_risk.mean()
            losses["ctm"] = ctm_loss * self.ctm_weight
        
        # MoE多样性损失
        if self.use_moe and routing_weights is not None:
            moe_loss = self.moe_loss(routing_weights)
            losses["moe_diversity"] = moe_loss * self.moe_weight
        
        # 总损失
        total_loss = sum(losses.values())
        losses["total"] = total_loss
        
        return total_loss, losses


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算Dice损失"""
        probs = F.softmax(logits, dim=1)
        
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


# =============================================================================
# 训练器
# =============================================================================

class MedMambaTrainer:
    """
    MedMamba训练器
    
    支持:
    - 单GPU/多GPU训练 (DDP)
    - 早停机制
    - 学习率调度
    - AMP混合精度训练
    - Mixup/Cutmix增强
    - Wandb/Tensorboard日志
    """
    
    def __init__(
        self,
        config: TrainConfig,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        
        # 设备
        self.device = config.device
        
        # 创建模型
        if model is None:
            self.model = create_medmamba(
                version=config.model_version,
                img_size=config.img_size,
                in_channels=config.in_channels,
                num_classes=config.num_classes,
                num_diseases=config.num_diseases,
                d_model=config.d_model,
                n_layers=config.n_layers,
                d_state=config.d_state,
                dropout=config.dropout,
                use_ctm=config.use_ctm,
                use_moe=config.use_moe,
                num_experts=config.num_experts,
            ).to(self.device)
        else:
            self.model = model
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 损失函数
        self.criterion = MedMambaLoss(
            task=config.task,
            num_classes=config.num_classes,
            multi_label=config.multi_label,
            use_ctm=config.use_ctm,
            ctm_weight=config.ctm_loss_weight,
            use_moe=config.use_moe,
            moe_weight=config.moe_loss_weight,
        )
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度
        self.scheduler = self._create_scheduler()
        
        # AMP
        self.scaler = torch.amp.GradScaler('cuda') if config.use_amp else None
        
        # Mixup/Cutmix
        self.mixup = None
        self.cutmix = None
        if config.use_mixup:
            self.mixup = Mixup(alpha=config.mixup_alpha, prob=0.5)
            self.cutmix = Cutmix(alpha=config.cutmix_alpha, prob=0.5)
        
        # 分布式
        self.is_distributed = config.world_size > 1
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[config.local_rank]
            )
        
        # 指标
        self.train_metrics = MetricsCalculator(config.num_classes, config.multi_label)
        self.val_metrics = MetricsCalculator(config.num_classes, config.multi_label)
        
        # 早停
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')
        
        # 日志
        self.writer = None
        if HAS_TENSORBOARD and config.local_rank == 0:
            os.makedirs(config.save_dir, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(config.save_dir, "runs"))
        
        self.wandb_run = None
        if HAS_WANDB and config.use_wandb and config.local_rank == 0:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"{config.model_name}_{config.model_version}",
                config=asdict(config),
            )
            self.wandb_run = wandb
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [],
            'lr': [],
        }
        
        # 保存路径
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> Optimizer:
        """创建优化器"""
        if self.config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
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
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
    
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
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        if self.is_distributed:
            self.model.train()
        else:
            self.model.train()
        
        self.train_metrics.reset()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Mixup/Cutmix
            use_mix = (self.mixup is not None) and (random.random() < 0.5)
            use_cut = (self.cutmix is not None) and not use_mix and (random.random() < 0.5)
            
            if use_mix:
                images, labels_a, labels_b, lam = self.mixup(images, labels)
            elif use_cut:
                images, labels_a, labels_b, lam = self.cutmix(images, labels)
            else:
                labels_a, labels_b, lam = labels, labels, 1.0
            
            self.optimizer.zero_grad()

            # 前向传播
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)

                    if isinstance(outputs, dict):
                        logits = outputs.get('logits') or outputs.get('cls_logits')
                    else:
                        logits = outputs

                    if use_mix or use_cut:
                        # Mixup/Cutmix只用任务损失, 不用CTM/MoE辅助损失
                        loss_a = self.criterion.task_loss(logits, labels_a)
                        loss_b = self.criterion.task_loss(logits, labels_b)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss, _ = self.criterion(outputs, labels, None)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)

                if isinstance(outputs, dict):
                    logits = outputs.get('logits') or outputs.get('cls_logits')
                else:
                    logits = outputs

                if use_mix or use_cut:
                    # Mixup/Cutmix只用任务损失, 不用CTM/MoE辅助损失
                    loss_a = self.criterion.task_loss(logits, labels_a)
                    loss_b = self.criterion.task_loss(logits, labels_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss, _ = self.criterion(outputs, labels, None)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            # 更新指标
            if isinstance(outputs, dict):
                logits = outputs.get('logits') or outputs.get('cls_logits')
            else:
                logits = outputs
            
            self.train_metrics.update(loss.item(), logits, labels_a)
            epoch_loss += loss.item()
            
            if batch_idx % self.config.log_interval == 0 and self.config.local_rank == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: Loss={loss.item():.4f}")
            
            self.global_step += 1
        
        metrics = self.train_metrics.compute()
        metrics['loss'] = epoch_loss / len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
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
            
            loss, _ = self.criterion(outputs, labels, None)
            self.val_metrics.update(loss.item(), logits, labels)
        
        return self.val_metrics.compute()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_last: bool = False):
        """保存检查点"""
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
        
        # 保存
        if is_last or is_best:
            name = "best" if is_best else "last"
            path = self.save_dir / f"{self.config.model_name}_{name}.pt"
            torch.save(checkpoint, path)
            print(f"[Trainer] {name.capitalize()} checkpoint saved: {path}")
        
        # 按epoch保存
        if (epoch + 1) % self.config.eval_interval == 0:
            epoch_path = self.save_dir / f"{self.config.model_name}_epoch{epoch+1}.pt"
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载检查点"""
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
        """早停检查"""
        if val_loss < self.best_val_loss - self.config.early_stop_min_delta:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.config.early_stop_patience:
                print(f"[Trainer] Early stopping triggered")
                return True
            return False
    
    def log_metrics(self, phase: str, metrics: Dict[str, float], epoch: int):
        """记录指标"""
        if self.config.local_rank != 0:
            return
        
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{phase}/{name}", value, epoch)
        
        if self.wandb_run:
            log_dict = {f"{phase}_{k}": v for k, v in metrics.items()}
            log_dict['epoch'] = epoch
            self.wandb_run.log(log_dict)
    
    def train(self, start_epoch: int = 0) -> Dict[str, List[float]]:
        """完整训练流程"""
        if self.config.local_rank == 0:
            print("\n" + "=" * 60)
            print(f"MedMamba Training: {self.config.model_version}")
            print(f"Task: {self.config.task}")
            print(f"Device: {self.device}")
            print(f"World Size: {self.config.world_size}")
            print(f"Epochs: {self.config.epochs}")
            print(f"Batch Size: {self.config.batch_size}")
            print(f"Learning Rate: {self.config.learning_rate}")
            print(f"Use MoE: {self.config.use_moe}, Use CTM: {self.config.use_ctm}")
            print("=" * 60 + "\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, self.config.epochs):
            epoch_start = time.time()
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.evaluate()
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_metrics.get('loss', 0))
            self.history['train_acc'].append(train_metrics.get('accuracy', 0))
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['val_acc'].append(val_metrics.get('accuracy', 0))
            self.history['val_f1'].append(val_metrics.get('f1', 0))
            self.history['val_auc'].append(val_metrics.get('auc', 0))
            self.history['lr'].append(current_lr)
            
            # 日志
            if self.config.local_rank == 0:
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch+1}/{self.config.epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics.get('loss', 0):.4f}, Acc: {train_metrics.get('accuracy', 0):.4f}")
                print(f"  Val Loss: {val_metrics.get('loss', 0):.4f}, Acc: {val_metrics.get('accuracy', 0):.4f}")
                print(f"  Val F1: {val_metrics.get('f1', 0):.4f}, AUC: {val_metrics.get('auc', 0):.4f}")
                print(f"  LR: {current_lr:.6f}")
                
                self.log_metrics("train", train_metrics, epoch)
                self.log_metrics("val", val_metrics, epoch)
                
                # 保存检查点
                is_best = val_metrics.get('loss', float('inf')) < best_val_loss
                if is_best:
                    best_val_loss = val_metrics.get('loss', float('inf'))
                
                is_last = (epoch + 1) == self.config.epochs
                self.save_checkpoint(epoch, is_best=is_best, is_last=is_last)
                
                # 早停
                if self.early_stopping(val_metrics.get('loss', float('inf'))):
                    print(f"[Trainer] Stopping at epoch {epoch+1}")
                    break
            
            self.current_epoch = epoch + 1
        
        # 保存历史
        if self.config.local_rank == 0:
            history_path = self.save_dir / f"{self.config.model_name}_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            if self.writer:
                self.writer.close()
            
            print("\n" + "=" * 60)
            print("Training Complete!")
            print(f"Results saved to {self.save_dir}")
            print("=" * 60)
        
        return self.history


class MetricsCalculator:
    """训练指标计算器"""
    
    def __init__(self, num_classes: int = 2, multi_label: bool = False):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.count = 0
        self.correct = 0
        self.total = 0
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor):
        self.total_loss += loss * logits.size(0)
        self.count += logits.size(0)
        
        if self.multi_label:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            self.correct += (preds == labels).all(dim=1).sum().item()
        else:
            preds = logits.argmax(dim=-1)
            self.correct += (preds == labels).sum().item()
        
        self.total += labels.size(0)
        
        self.all_probs.append(F.softmax(logits, dim=-1).detach().cpu())
        self.all_labels.append(labels.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        metrics = {}
        metrics['loss'] = self.total_loss / self.count if self.count > 0 else 0
        metrics['accuracy'] = self.correct / self.total if self.total > 0 else 0
        
        if len(self.all_labels) > 0:
            all_labels = torch.cat(self.all_labels)
            all_probs = torch.cat(self.all_probs)
            
            try:
                from sklearn.metrics import roc_auc_score, f1_score
                
                if self.num_classes == 2:
                    metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
                    metrics['f1'] = f1_score(all_labels, all_probs.argmax(dim=-1), average='binary')
                else:
                    metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
                    metrics['f1'] = f1_score(all_labels, all_probs.argmax(dim=-1), average='macro', zero_division=0)
            except ImportError:
                metrics['auc'] = 0.0
                metrics['f1'] = 0.0
        
        return metrics


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MedMamba Training")
    
    # 模型
    parser.add_argument("--model", type=str, default="v2", choices=["v1", "v2", "v3"],
                        help="Model version")
    parser.add_argument("--model_name", type=str, default="MedMamba", help="Model name")
    parser.add_argument("--d_model", type=int, default=384, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--d_state", type=int, default=16, help="SSM state dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # 数据
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--use_ct_window", action="store_true", help="Use CT window adjustment")
    
    # 任务
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "segmentation"],
                        help="Task type")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--num_diseases", type=int, default=8, help="Number of diseases")
    parser.add_argument("--multi_label", action="store_true", help="Multi-label classification")
    parser.add_argument("--class_names", type=str, nargs='+', default=None, help="Class names")
    
    # 训练
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adam", "SGD"],
                        help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"],
                        help="Scheduler")
    parser.add_argument("--step_size", type=int, default=30, help="Step LR step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="Step LR gamma")
    
    # AMP
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use AMP")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clip value")
    
    # 早停
    parser.add_argument("--early_stop_patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--early_stop_delta", type=float, default=0.001, help="Early stopping delta")
    
    # Mixup/Cutmix
    parser.add_argument("--use_mixup", action="store_true", help="Use Mixup")
    parser.add_argument("--mixup_alpha", type=float, default=1.0, help="Mixup alpha")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0, help="Cutmix alpha")
    
    # MoE
    parser.add_argument("--use_moe", action="store_true", help="Use MoE")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts")
    parser.add_argument("--moe_weight", type=float, default=0.1, help="MoE loss weight")
    
    # CTM
    parser.add_argument("--use_ctm", action="store_true", help="Use CTM")
    parser.add_argument("--ctm_weight", type=float, default=0.05, help="CTM loss weight")
    
    # 日志
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save directory")
    
    # Wandb/Tensorboard
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb")
    parser.add_argument("--wandb_project", type=str, default="MedMamba", help="Wandb project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--use_tensorboard", action="store_true", help="Use Tensorboard")
    
    # 恢复
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # 分布式
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for DDP")
    parser.add_argument("--world_size", type=int, default=1, help="World size for DDP")
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples (for mock data)")
    parser.add_argument("--use_mock_data", action="store_true", help="Use mock data for testing")
    
    return parser.parse_args()


def setup_distributed():
    """设置分布式训练"""
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
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    
    # 设置分布式
    local_rank, world_size = setup_distributed()
    args.local_rank = local_rank
    args.world_size = world_size
    
    # 设置种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 配置
    config = TrainConfig(
        model_version=args.model,
        model_name=args.model_name,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        dropout=args.dropout,
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task=args.task,
        num_classes=args.num_classes,
        num_diseases=args.num_diseases,
        multi_label=args.multi_label,
        class_names=args.class_names,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        use_amp=not args.no_amp and args.use_amp,
        gradient_clip=args.gradient_clip,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_delta,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        moe_loss_weight=args.moe_weight,
        use_ctm=args.use_ctm,
        ctm_loss_weight=args.ctm_weight,
        use_ct_window=args.use_ct_window,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        resume=args.resume,
        local_rank=local_rank,
        world_size=world_size,
        device=args.device,
        seed=args.seed,
    )
    
    # 设备
    if torch.cuda.is_available() and args.device == "cuda":
        device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
    else:
        device = "cpu"
    config.device = device
    
    if local_rank == 0:
        print(f"Using device: {device}")
    
    # 数据
    if args.use_mock_data:
        if local_rank == 0:
            print("Using mock data for testing")
        train_dataset = MockMedicalDataset(args.num_samples, args.img_size, args.task)
        val_dataset = MockMedicalDataset(min(100, args.num_samples // 10), args.img_size, args.task)
    else:
        # 真实数据集
        train_dataset = MedMambaDataset(
            data_root=args.data_root,
            img_size=args.img_size,
            task=args.task,
            num_classes=args.num_classes,
            mode="train",
            use_ct_window=args.use_ct_window,
        )
        val_dataset = None  # 需要时可以从DataLoaderFactory分割
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    # 创建模型
    model = create_medmamba(
        version=config.model_version,
        img_size=config.img_size,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        num_diseases=config.num_diseases,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        dropout=config.dropout,
        use_ctm=config.use_ctm,
        use_moe=config.use_moe,
        num_experts=config.num_experts,
    )
    
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    model = model.to(device)
    
    # 创建训练器
    trainer = MedMambaTrainer(config, model, train_loader, val_loader)
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        if local_rank == 0:
            print(f"Resumed from epoch {start_epoch}")
    
    # 训练
    history = trainer.train(start_epoch=start_epoch)
    
    # 清理
    cleanup_distributed()
    
    if local_rank == 0:
        print("\nTraining completed!")
        print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()