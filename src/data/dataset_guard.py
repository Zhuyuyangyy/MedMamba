"""
MedMamba-Guard数据集模块

支持:
- ISIC 2018皮肤病变数据集
- MedMNIST标准化数据集
- 合成医学影像数据集
- 数据增强: RandomFlip, RandomRotate, ColorJitter, Mixup, Cutmix

返回格式:
- image: (3, H, W) tensor
- label: (1,) tensor
- segmentation_mask: (1, H, W) tensor (如果可用)
- metadata: dict

作者: MedMamba Team
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 尝试导入medmnist
try:
    import medmnist
    from medmnist import INFO
    HAS_MEDMNIST = True
except ImportError:
    HAS_MEDMNIST = False
    print("Warning: medmnist not installed. MedMNIST dataset will not be available.")

# 导入数据增强
from .augmentation import (
    RandomFlip, RandomCrop, ColorJitter, 
    Normalize, CTWindowAdjustment,
    Mixup, Cutmix
)


# =============================================================================
# ISIC 2018 数据集
# =============================================================================

class ISIC2018Dataset(Dataset):
    """
    ISIC 2018皮肤病变数据集
    
    目录结构:
        data_root/ISIC2018/
            images/
                ISIC_001.jpg
                ISIC_002.jpg
                ...
            masks/
                ISIC_001_segmentation.png
                ISIC_002_segmentation.png
                ...
    
    或:
        data_root/ISIC2018/
            train/
                images/
                masks/
            val/
                images/
                masks/
            test/
                images/
                masks/
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 224,
        transform: Optional[Callable] = None,
        mask_dir: Optional[str] = None,
        return_mask: bool = True,
    ):
        """
        初始化ISIC2018数据集
        
        Args:
            data_root: 数据根目录
            split: 数据集划分 (train/val/test)
            img_size: 图像尺寸
            transform: 数据变换
            mask_dir: mask目录路径 (如果为None则自动查找)
            return_mask: 是否返回分割mask
        """
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.return_mask = return_mask
        
        # 确定路径
        if (self.data_root / "train").exists() or (self.data_root / "val").exists():
            # 嵌套结构
            self.split_root = self.data_root / split
            self.image_dir = self.split_root / "images"
            self.mask_dir = self.split_root / "masks" if mask_dir is None else Path(mask_dir)
        else:
            # 扁平结构
            self.split_root = self.data_root / "ISIC2018"
            self.image_dir = self.split_root / "images"
            self.mask_dir = self.split_root / "masks" if mask_dir is None else Path(mask_dir)
        
        # 加载样本列表
        self.samples = self._load_samples()
        
        # 默认变换
        self.default_transform = self._get_default_transform()
    
    def _load_samples(self) -> List[Dict]:
        """加载样本列表"""
        samples = []
        
        if not self.image_dir.exists():
            # 尝试不同的路径
            alt_paths = [
                self.data_root / "ISIC2018" / "images",
                self.data_root / "images",
            ]
            for p in alt_paths:
                if p.exists():
                    self.image_dir = p
                    break
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # 遍历图像
        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            
            # 对应mask路径
            stem = img_path.stem
            mask_path = None
            if self.return_mask and self.mask_dir.exists():
                # 尝试不同mask文件名模式
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_mask = self.mask_dir / f"{stem}{ext}"
                    if potential_mask.exists():
                        mask_path = potential_mask
                        break
                    potential_mask = self.mask_dir / f"{stem}_segmentation{ext}"
                    if potential_mask.exists():
                        mask_path = potential_mask
                        break
            
            # 标签 (从文件名或目录结构推断)
            # ISIC2018: 0=正常, 1=恶性/病变
            # 这里简化处理，使用文件名的某些模式来模拟标签
            label = 1 if 'melanoma' in stem.lower() or 'malignant' in stem.lower() else 0
            
            samples.append({
                'path': str(img_path),
                'mask_path': str(mask_path) if mask_path else None,
                'label': label,
                'stem': stem,
            })
        
        print(f"[ISIC2018Dataset] Loaded {len(samples)} samples from {self.split} split")
        return samples
    
    def _get_default_transform(self):
        """获取默认变换"""
        if self.split == "train":
            transform_list = [
                transforms.Resize((self.img_size, self.img_size)),
                RandomFlip(prob=0.5),
                RandomRotate(degrees=15),
                ColorJitter(brightness=0.2, contrast=0.2),
            ]
        else:
            transform_list = [
                transforms.Resize((self.img_size, self.img_size)),
            ]
        
        transform_list.append(Normalize())
        
        return transforms.Compose(transform_list)
    
    def _load_image(self, path: str) -> np.ndarray:
        """加载图像"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_mask(self, path: str) -> np.ndarray:
        """加载mask"""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # 返回全零mask
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        
        # 加载图像
        image = self._load_image(sample_info['path'])
        
        # 加载mask
        mask = None
        if self.return_mask and sample_info['mask_path']:
            mask = self._load_mask(sample_info['mask_path'])
        elif self.return_mask:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # 应用变换
        if self.default_transform:
            image = self.default_transform(image)
        
        if mask is not None:
            mask = torch.from_numpy(mask).float() / 255.0
            mask = mask.unsqueeze(0)  # (1, H, W)
        
        return {
            'image': image,
            'label': torch.tensor(sample_info['label'], dtype=torch.long),
            'segmentation_mask': mask,
            'path': sample_info['path'],
            'stem': sample_info['stem'],
        }


# =============================================================================
# MedMNIST 数据集
# =============================================================================

class MedMNISTDataset(Dataset):
    """
    MedMNIST标准化医学影像数据集
    
    支持的数据集:
    - breastmnist: 乳腺癌超声图像 (binaries 1)
    - tissueMNIST: 组织病理学图像 (binaries 1-9)
    - organamnist: 器官MRI图像 (11类)
    - pneumoniamnist: 肺炎X光图像 (binaries 1)
    - chestmnist: 胸部X光图像 (binaries 1)
    - dermamnist: 皮肤病图像 (7类)
    - octmnist: OCT图像 (binaries 1)
    - retinaMNIST: 视网膜图像 (5类)
    - vasculatureMNIST: 血管图像 (binaries 1)
    """
    
    DATASET_INFO = {
        'breastmnist': {'n_channels': 1, 'n_classes': 2, 'size': 28},
        'tissuemnist': {'n_channels': 1, 'n_classes': 8, 'size': 28},
        'organamnist': {'n_channels': 1, 'n_classes': 11, 'size': 28},
        'pneumoniamnist': {'n_channels': 1, 'n_classes': 2, 'size': 28},
        'chestmnist': {'n_channels': 1, 'n_classes': 2, 'size': 28},
        'dermamnist': {'n_channels': 3, 'n_classes': 7, 'size': 28},
        'octmnist': {'n_channels': 1, 'n_classes': 4, 'size': 28},
        'retinamnist': {'n_channels': 3, 'n_classes': 5, 'size': 28},
        'vasculaturemnist': {'n_channels': 1, 'n_classes': 2, 'size': 28},
    }
    
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        split: str = "train",
        img_size: int = 224,
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """
        初始化MedMNIST数据集
        
        Args:
            dataset_name: 数据集名称 (如 'breastmnist', 'tissuemnist')
            data_root: 数据根目录
            split: 数据集划分 (train/val/test)
            img_size: 图像尺寸
            transform: 数据变换
            download: 是否下载数据集
        """
        if not HAS_MEDMNIST:
            raise ImportError("medmnist is not installed. Install with: pip install medmnist")
        
        self.dataset_name = dataset_name.lower()
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # 获取数据集信息
        if self.dataset_name not in self.DATASET_INFO:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASET_INFO.keys())}")
        
        self.info = self.DATSET_INFO[self.dataset_name]
        self.n_channels = self.info['n_channels']
        self.n_classes = self.info['n_classes']
        
        # 加载数据集
        self.dataset = self._load_dataset(download)
        
        # 默认变换
        self.default_transform = self._get_default_transform()
    
    def _load_dataset(self, download: bool):
        """加载MedMNIST数据集"""
        # 动态导入medmnist
        import medmnist
        from medmnist import Evaluator
        
        # 获取DataClass
        data_class = getattr(medmnist, self.dataset_name.capitalize())
        
        # 下载并加载
        self.data_flag = data_class.split + f"_{self.dataset_name}"
        
        # 创建数据类
        dataset = data_class(
            root=self.data_root,
            split=self.split,
            download=download,
            transform=None,  # 我们自己处理transform
        )
        
        return dataset
    
    def _get_default_transform(self):
        """获取默认变换"""
        from medmnist import DataEnhancer
        
        if self.split == "train":
            transform_list = [
                transforms.Resize((self.img_size, self.img_size)),
                RandomFlip(prob=0.5),
            ]
        else:
            transform_list = [
                transforms.Resize((self.img_size, self.img_size)),
            ]
        
        transform_list.append(Normalize())
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 获取原始数据
        img, label = self.dataset[idx]
        
        # 转换图像
        if isinstance(img, torch.Tensor):
            image = img.numpy()
        else:
            image = np.array(img)
        
        # 处理通道
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)  # (H, W, 1)
        
        # 转换为RGB (如果需要)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        
        # 调整尺寸
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # 应用变换
        if self.default_transform:
            image = self.default_transform(image)
        
        # 确保label是标量
        if isinstance(label, (list, np.ndarray, torch.Tensor)):
            label = label.item() if label.numel() == 1 else label
        
        return {
            'image': image,
            'label': torch.tensor(label if isinstance(label, int) else int(label), dtype=torch.long),
            'path': f"{self.dataset_name}_{self.split}_{idx}",
            'stem': f"{self.dataset_name}_{self.split}_{idx}",
        }


# =============================================================================
# 合成医学影像数据集
# =============================================================================

class SyntheticMedicalDataset(Dataset):
    """
    合成医学影像数据集
    
    用于快速测试和调试
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        img_size: int = 224,
        num_classes: int = 2,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        """
        初始化合成数据集
        
        Args:
            num_samples: 样本数量
            img_size: 图像尺寸
            num_classes: 类别数
            split: 数据集划分
            transform: 数据变换
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.split = split
        self.transform = transform
        
        # 生成随机标签
        self.labels = np.random.randint(0, num_classes, size=num_samples)
        
        # 默认变换
        self.default_transform = self._get_default_transform()
    
    def _get_default_transform(self):
        """获取默认变换"""
        if self.split == "train":
            transform_list = [
                transforms.Resize((self.img_size, self.img_size)),
                RandomFlip(prob=0.5),
            ]
        else:
            transform_list = [
                transforms.Resize((self.img_size, self.img_size)),
            ]
        
        transform_list.append(Normalize())
        return transforms.Compose(transform_list)
    
    def _generate_synthetic_image(self, label: int) -> np.ndarray:
        """生成合成图像"""
        # 创建基础图像
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        
        # 添加不同的模式用于不同类别
        if label == 0:
            # 正常: 中心高斯斑点
            center = self.img_size // 2
            y, x = np.ogrid[:self.img_size, :self.img_size]
            mask = ((x - center) ** 2 + (y - center) ** 2) < (self.img_size // 4) ** 2
            img[mask] = 0.8
            img[~mask] = 0.2
        else:
            # 病变: 添加多个斑点
            center = self.img_size // 2
            y, x = np.ogrid[:self.img_size, :self.img_size]
            
            # 中心斑点
            mask1 = ((x - center) ** 2 + (y - center) ** 2) < (self.img_size // 4) ** 2
            img[mask1] = 0.9
            
            # 边缘斑点
            for _ in range(3):
                cx = np.random.randint(self.img_size // 4, 3 * self.img_size // 4)
                cy = np.random.randint(self.img_size // 4, 3 * self.img_size // 4)
                r = np.random.randint(self.img_size // 8, self.img_size // 4)
                mask = ((x - cx) ** 2 + (y - cy) ** 2) < r ** 2
                img[mask] = 0.85
        
        # 添加噪声
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        # 转换为uint8
        img = (img * 255).astype(np.uint8)
        
        return img
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label = self.labels[idx]
        image = self._generate_synthetic_image(label)
        
        # 应用变换
        if self.default_transform:
            image = self.default_transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': f"synthetic_{idx}",
            'stem': f"synthetic_{idx}",
        }


# =============================================================================
# RandomRotate (因为augmentation.py可能没有)
# =============================================================================

class RandomRotate:
    """随机旋转增强"""
    
    def __init__(self, degrees: float = 15, prob: float = 0.5):
        self.degrees = degrees
        self.prob = prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        import random
        if random.random() > self.prob:
            return image
        
        import cv2
        h, w = image.shape[:2]
        angle = random.uniform(-self.degrees, self.degrees)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated


# =============================================================================
# 数据集工厂函数
# =============================================================================

def create_dataset(
    dataset_name: str,
    data_root: str,
    split: str = "train",
    transform: Optional[Callable] = None,
    img_size: int = 224,
    **kwargs,
) -> Dataset:
    """
    创建数据集的工厂函数
    
    Args:
        dataset_name: 数据集名称 (isic2018, medmnist, synthetic)
        data_root: 数据根目录
        split: 数据集划分 (train/val/test)
        transform: 数据变换
        img_size: 图像尺寸
        **kwargs: 其他参数
    
    Returns:
        dataset: Dataset实例
    
    Raises:
        ValueError: 未知数据集名称
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'isic2018':
        return ISIC2018Dataset(
            data_root=data_root,
            split=split,
            img_size=img_size,
            transform=transform,
            **kwargs,
        )
    elif dataset_name == 'medmnist':
        return MedMNISTDataset(
            dataset_name=kwargs.get('medmnist_subset', 'breastmnist'),
            data_root=data_root,
            split=split,
            img_size=img_size,
            transform=transform,
            **kwargs,
        )
    elif dataset_name == 'synthetic':
        return SyntheticMedicalDataset(
            num_samples=kwargs.get('num_samples', 1000),
            img_size=img_size,
            num_classes=kwargs.get('num_classes', 2),
            split=split,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: isic2018, medmnist, synthetic")


def create_medical_dataloader(
    dataset_name: str,
    data_root: str,
    split: str = "train",
    batch_size: int = 16,
    img_size: int = 224,
    num_workers: int = 4,
    shuffle: bool = None,
    **kwargs,
) -> DataLoader:
    """
    创建医学影像数据加载器
    
    Args:
        dataset_name: 数据集名称
        data_root: 数据根目录
        split: 数据集划分
        batch_size: 批量大小
        img_size: 图像尺寸
        num_workers: 工作进程数
        shuffle: 是否打乱 (默认: train=True, val/test=False)
        **kwargs: 其他参数
    
    Returns:
        dataloader: DataLoader实例
    """
    # 确定shuffle
    if shuffle is None:
        shuffle = (split == 'train')
    
    # 创建数据集
    dataset = create_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        img_size=img_size,
        **kwargs,
    )
    
    # 创建数据加载器
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )


# =============================================================================
# 分割数据集包装器
# =============================================================================

class SegmentationDatasetWrapper(Dataset):
    """分割任务数据集包装器
    
    将分类数据集包装成分割数据集
    """
    
    def __init__(
        self,
        classification_dataset: Dataset,
        generate_masks: bool = False,
    ):
        """
        Args:
            classification_dataset: 分类数据集
            generate_masks: 是否生成合成mask (True时使用高斯中心作为mask)
        """
        self.dataset = classification_dataset
        self.generate_masks = generate_masks
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        
        # 如果已有mask，直接返回
        if sample.get('segmentation_mask') is not None:
            return sample
        
        # 生成合成mask
        image = sample['image']
        if isinstance(image, torch.Tensor):
            # 从tensor获取尺寸
            if image.dim() == 3:
                _, h, w = image.shape
            else:
                h, w = self.dataset.img_size, self.dataset.img_size
        else:
            h, w = image.shape[:2] if hasattr(image, 'shape') else (self.dataset.img_size, self.dataset.img_size)
        
        # 生成简单的圆形mask (模拟分割标注)
        mask = torch.zeros(1, h, w)
        
        # 中心区域
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist = ((x - cx) ** 2 + (y - cy) ** 2).float()
        mask[0] = (dist < radius ** 2).float()
        
        sample['segmentation_mask'] = mask
        
        return sample


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数 - 演示用法"""
    print("[Dataset] Available datasets:")
    print("  - isic2018: ISIC 2018 skin lesion dataset")
    print("  - medmnist: MedMNIST standardized dataset")
    print("  - synthetic: Synthetic medical images for testing")
    print("")
    print("[Dataset] Usage:")
    print("  from src.data.dataset_guard import create_dataset, create_medical_dataloader")
    print("")
    print("  # Create dataset")
    print("  train_dataset = create_dataset('synthetic', './data', split='train')")
    print("  ")
    print("  # Create dataloader")
    print("  train_loader = create_medical_dataloader('synthetic', './data', split='train', batch_size=16)")


if __name__ == "__main__":
    main()