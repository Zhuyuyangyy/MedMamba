"""
MedMamba数据集模块
支持CT/MRI医学影像数据加载和数据增强

作者: MedMamba Team
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .augmentation import (
    RandomFlip, RandomCrop, ColorJitter, 
    Normalize, CTWindowAdjustment,
    Mixup, Cutmix
)


# =============================================================================
# 医学影像数据集
# =============================================================================

class MedMambaDataset(Dataset):
    """
    MedMamba医学影像数据集
    
    支持:
    - 本地文件夹结构 (按类别组织)
    - CT/MRI图像读取 (NIfTI, DICOM, PNG, JPG)
    - 数据增强 (flip, crop, normalize, window/level调整)
    
    目录结构:
        data_root/
            class1/
                img1.png
                img2.dcm
            class2/
                ...
    """
    
    SUPPORTED_IMAGE_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif',
        '.nii', '.nifti', '.nii.gz',  # NIfTI格式
        '.dcm', '.dicom',             # DICOM格式
    }
    
    def __init__(
        self,
        data_root: str,
        img_size: int = 224,
        task: str = "classification",
        num_classes: int = 2,
        mode: str = "train",
        transform: Optional[Callable] = None,
        use_ct_window: bool = False,
        window_presets: Optional[Dict[str, Tuple[int, int]]] = None,
        cache_data: bool = False,
        multi_label: bool = False,
    ):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            img_size: 图像尺寸
            task: 任务类型 (classification/segmentation)
            num_classes: 类别数
            mode: 模式 (train/val/test)
            transform: 自定义变换
            use_ct_window: 是否使用CT窗宽窗位调整
            window_presets: CT窗宽窗位预设
            cache_data: 是否缓存数据到内存
            multi_label: 是否多标签分类
        """
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.task = task
        self.num_classes = num_classes
        self.mode = mode
        self.transform = transform
        self.use_ct_window = use_ct_window
        self.cache_data = cache_data
        self.multi_label = multi_label
        
        # 默认CT窗宽窗位预设
        self.window_presets = window_presets or {
            'bone': (2000, 500),      # 骨窗
            'lung': (1500, -600),    # 肺窗
            'soft': (400, 50),       # 软组织窗
            'brain': (80, 40),       # 脑窗
            'liver': (150, 30),      # 肝脏窗
        }
        
        # 加载数据
        self.samples = self._load_data()
        
        # 缓存
        self.cached_data = {} if cache_data else None
        
        # 默认变换 (mode为train时启用)
        self.default_transform = self._get_default_transform()
    
    def _load_data(self) -> List[Dict]:
        """
        加载数据样本
        
        Returns:
            samples: 样本列表, 每个元素为 {path, label, modality}
        """
        samples = []
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        # 遍历类别目录
        class_dirs = [d for d in self.data_root.iterdir() if d.is_dir()]
        
        for class_idx, class_dir in enumerate(sorted(class_dirs)):
            class_name = class_dir.name
            
            # 遍历该类别下所有图像
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_IMAGE_EXTENSIONS:
                    samples.append({
                        'path': str(img_path),
                        'label': class_idx,
                        'modality': self._detect_modality(img_path),
                        'class_name': class_name,
                    })
        
        if len(samples) == 0:
            raise ValueError(f"No images found in {self.data_root}")
        
        print(f"[MedMambaDataset] Loaded {len(samples)} samples from {self.data_root}")
        return samples
    
    def _detect_modality(self, path: Path) -> str:
        """
        检测影像模态 (CT/MRI/X-ray等)
        
        Args:
            path: 图像路径
        
        Returns:
            modality: 模态字符串
        """
        suffix = path.suffix.lower()
        
        if suffix in ['.nii', '.nifti', '.nii.gz']:
            # NIfTI格式通常是CT或MRI
            # 这里简化处理, 实际应用需要读取header信息
            return 'CT' if 'ct' in path.stem.lower() else 'MRI'
        elif suffix in ['.dcm', '.dicom']:
            # DICOM格式需要读取tag信息判断模态
            # 简化处理
            return 'CT' if 'ct' in path.stem.lower() else 'MRI'
        else:
            # 其他格式默认为普通医学图像
            return 'X-ray'
    
    def _get_default_transform(self):
        """
        获取默认数据变换
        
        Returns:
            transform: torchvision transforms组合
        """
        if self.mode == "train":
            transform_list = []
            
            # 基础变换
            transform_list.append(transforms.Resize((self.img_size, self.img_size)))
            
            # 数据增强
            if self.transform:
                transform_list.append(self.transform)
            else:
                # 默认增强
                transform_list.append(RandomFlip(prob=0.5))
                transform_list.append(RandomCrop(crop_size=self.img_size))
                
                # CT窗宽窗位调整 (训练时随机选择)
                if self.use_ct_window:
                    transform_list.append(
                        CTWindowAdjustment(presets=self.window_presets, prob=0.5)
                    )
                
                # 颜色抖动 (适用于X-ray/MRI)
                transform_list.append(ColorJitter(brightness=0.2, contrast=0.2))
            
            # 归一化
            transform_list.append(Normalize())
            
            return transforms.Compose(transform_list)
        else:
            # 验证/测试模式只做基础变换
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                Normalize(),
            ])
    
    def _load_image(self, path: str) -> np.ndarray:
        """
        加载医学图像
        
        Args:
            path: 图像路径
        
        Returns:
            image: numpy数组, shape [H, W] 或 [H, W, C]
        """
        suffix = Path(path).suffix.lower()
        
        if suffix in ['.nii', '.nifti', '.nii.gz']:
            # 加载NIfTI
            try:
                import nibabel as nib
                img = nib.load(path)
                image = img.get_fdata()
                
                # 处理3D图像 (取中间层)
                if len(image.shape) == 3:
                    mid_slice = image.shape[2] // 2
                    image = image[:, :, mid_slice]
                
                # 转换为uint8
                image = self._normalize_to_uint8(image)
                return image
            except ImportError:
                # nibabel未安装, 尝试用其他方式
                print(f"Warning: nibabel not installed, using fallback for {path}")
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                return image if image is not None else np.zeros((self.img_size, self.img_size))
        
        elif suffix in ['.dcm', '.dicom']:
            # 加载DICOM
            try:
                import pydicom
                ds = pydicom.dcmread(path)
                image = ds.pixel_array
                
                # 应用窗外设置 (如果存在)
                if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                    # 简单处理
                    pass
                
                image = self._normalize_to_uint8(image)
                return image
            except ImportError:
                print(f"Warning: pydicom not installed, using fallback for {path}")
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                return image if image is not None else np.zeros((self.img_size, self.img_size))
        
        else:
            # 普通图像格式
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                # 尝试RGB
                image_rgb = cv2.imread(path, cv2.IMREAD_COLOR)
                if image_rgb is not None:
                    return cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
                return np.zeros((self.img_size, self.img_size))
            
            # 转换为RGB (统一处理)
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    def _normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        """
        将图像归一化到uint8
        
        Args:
            image: 输入图像
        
        Returns:
            normalized: uint8图像
        """
        # 线性归一化
        img_min = image.min()
        img_max = image.max()
        
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min) * 255
        else:
            image = np.zeros_like(image)
        
        return image.astype(np.uint8)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取样本
        
        Args:
            idx: 样本索引
        
        Returns:
            sample: 字典 {image, label, ...}
        """
        # 检查缓存
        if self.cached_data is not None and idx in self.cached_data:
            return self.cached_data[idx]
        
        sample_info = self.samples[idx]
        
        # 加载图像
        image = self._load_image(sample_info['path'])
        
        # 调整尺寸
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # 应用默认变换
        if self.default_transform:
            image = self.default_transform(image)
        
        # 获取标签
        label = sample_info['label']
        
        # 构建返回字典
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': sample_info['path'],
            'modality': sample_info['modality'],
        }
        
        # 多标签情况
        if self.multi_label:
            # 假设multi_label使用one-hot编码
            one_hot = torch.zeros(self.num_classes)
            one_hot[label] = 1
            sample['label'] = one_hot
        
        # 分割任务
        if self.task == "segmentation":
            # 分割任务的label处理可能需要加载mask
            # 这里简化处理, 假设有对应的mask文件
            pass
        
        # 缓存
        if self.cached_data is not None:
            self.cached_data[idx] = sample
        
        return sample


# =============================================================================
# 切片数据集 (用于3D医学影像)
# =============================================================================

class MedicalVolumeDataset(Dataset):
    """
    3D医学影像体积数据集
    
    将3D体积数据(如CT/MRI)切片为2D图像进行处理
    """
    
    def __init__(
        self,
        volume_paths: List[str],
        img_size: int = 224,
        num_classes: int = 2,
        mode: str = "train",
        transform: Optional[Callable] = None,
        slice_step: int = 1,  # 切片间隔
    ):
        """
        初始化体积数据集
        
        Args:
            volume_paths: 3D体积文件路径列表
            img_size: 输出图像尺寸
            num_classes: 类别数
            mode: 模式
            transform: 变换
            slice_step: 切片间隔 (每slice_step层取一层)
        """
        self.volume_paths = volume_paths
        self.img_size = img_size
        self.num_classes = num_classes
        self.mode = mode
        self.transform = transform
        self.slice_step = slice_step
        
        # 预处理: 获取所有切片
        self.slices = self._prepare_slices()
    
    def _prepare_slices(self) -> List[Tuple[str, int]]:
        """
        准备所有切片信息
        
        Returns:
            slices: [(volume_path, slice_idx), ...]
        """
        slices = []
        
        for vol_path in self.volume_paths:
            try:
                import nibabel as nib
                img = nib.load(vol_path)
                volume = img.get_fdata()
                
                # 按间隔提取切片
                for i in range(0, volume.shape[2], self.slice_step):
                    slices.append((vol_path, i))
            except ImportError:
                print(f"Warning: Cannot load {vol_path}")
        
        print(f"[MedicalVolumeDataset] Prepared {len(slices)} slices from {len(self.volume_paths)} volumes")
        return slices
    
    def __len__(self) -> int:
        return len(self.slices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_path, slice_idx = self.slices[idx]
        
        # 加载体积
        import nibabel as nib
        img = nib.load(vol_path)
        volume = img.get_fdata()
        
        # 提取切片
        slice_2d = volume[:, :, slice_idx]
        
        # 归一化
        slice_2d = self._normalize_to_uint8(slice_2d)
        
        # 转为RGB
        slice_rgb = np.stack([slice_2d] * 3, axis=-1)
        
        # 调整尺寸
        slice_rgb = cv2.resize(slice_rgb, (self.img_size, self.img_size))
        
        # 变换
        if self.transform:
            slice_rgb = self.transform(slice_rgb)
        
        # 标签 (从路径推断)
        label = 0  # 默认值, 实际应根据数据集设置
        
        return {
            'image': slice_rgb,
            'label': torch.tensor(label, dtype=torch.long),
            'volume_path': vol_path,
            'slice_idx': slice_idx,
        }
    
    def _normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min) * 255
        return image.astype(np.uint8)


# =============================================================================
# 数据加载器工厂
# =============================================================================

class DataLoaderFactory:
    """
    数据加载器工厂
    
    创建训练/验证/测试数据加载器
    """
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        """
        创建数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批量大小
            shuffle: 是否打乱
            num_workers: 工作进程数
            pin_memory: 是否固定内存
            drop_last: 是否丢弃最后不完整批次
        
        Returns:
            dataloader: DataLoader实例
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    
    @staticmethod
    def create_split_dataloaders(
        data_root: str,
        img_size: int = 224,
        batch_size: int = 16,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        num_workers: int = 4,
        **dataset_kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建训练/验证/测试数据加载器
        
        Args:
            data_root: 数据根目录
            img_size: 图像尺寸
            batch_size: 批量大小
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            num_workers: 工作进程数
            **dataset_kwargs: 其他数据集参数
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        from torch.utils.data import random_split
        
        # 创建完整数据集
        full_dataset = MedMambaDataset(
            data_root=data_root,
            img_size=img_size,
            mode="train",
            **dataset_kwargs
        )
        
        total_size = len(full_dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        # 分割
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 设置模式
        train_dataset.dataset.mode = "train"
        val_dataset.dataset.mode = "val"
        test_dataset.dataset.mode = "test"
        
        # 创建加载器
        train_loader = DataLoaderFactory.create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
        
        val_loader = DataLoaderFactory.create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        
        test_loader = DataLoaderFactory.create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        
        return train_loader, val_loader, test_loader


# Public API alias used by tests / external scripts.
# The canonical class is MedMambaDataset; MedicalImageDataset is provided for
# naming consistency with the medical-imaging literature.
MedicalImageDataset = MedMambaDataset