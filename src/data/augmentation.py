"""
MedMamba数据增强模块

支持:
- 基础增强: RandomFlip, RandomCrop, ColorJitter, Normalize
- 混合增强: Mixup, Cutmix
- 医学影像专用: CT窗宽窗位调整

作者: MedMamba Team
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Callable
import random


# =============================================================================
# 基础增强
# =============================================================================

class RandomFlip:
    """
    随机翻转增强
    
    支持水平翻转和垂直翻转
    """
    
    def __init__(self, prob: float = 0.5, horizontal: bool = True, vertical: bool = True):
        """
        Args:
            prob: 翻转概率
            horizontal: 是否支持水平翻转
            vertical: 是否支持垂直翻转
        """
        self.prob = prob
        self.horizontal = horizontal
        self.vertical = vertical
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        执行翻转
        
        Args:
            image: 输入图像 [H, W] 或 [H, W, C]
        
        Returns:
            image: 增强后图像
        """
        if random.random() < self.prob:
            if self.horizontal and random.random() < 0.5:
                image = np.fliplr(image).copy()
            if self.vertical and random.random() < 0.5:
                image = np.flipud(image).copy()
        
        return image


class RandomCrop:
    """
    随机裁剪增强
    
    从图像中随机裁剪指定大小的区域
    """
    
    def __init__(self, crop_size: int = 224, prob: float = 0.5):
        """
        Args:
            crop_size: 裁剪尺寸
            prob: 裁剪概率
        """
        self.crop_size = crop_size
        self.prob = prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        执行裁剪
        
        Args:
            image: 输入图像 [H, W] 或 [H, W, C]
        
        Returns:
            image: 增强后图像
        """
        if random.random() > self.prob:
            return image
        
        h, w = image.shape[:2]
        
        if h < self.crop_size or w < self.crop_size:
            # 图像小于裁剪尺寸, 进行缩放
            return cv2.resize(image, (self.crop_size, self.crop_size))
        
        # 随机选择裁剪位置
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        
        cropped = image[top:top + self.crop_size, left:left + self.crop_size]
        
        return cropped


class ColorJitter:
    """
    颜色抖动增强
    
    随机调整图像的亮度、对比度、饱和度和色调
    适用于X-ray和MRI图像
    """
    
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        prob: float = 0.5,
    ):
        """
        Args:
            brightness: 亮度调整范围
            contrast: 对比度调整范围
            saturation: 饱和度调整范围
            hue: 色调调整范围
            prob: 应用概率
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        执行颜色抖动
        
        Args:
            image: 输入图像 [H, W, C] uint8
        
        Returns:
            image: 增强后图像
        """
        if random.random() > self.prob:
            return image
        
        # 转换为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 亮度调整
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        
        # 对比度调整
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            hsv[:, :, 2] = np.clip(((hsv[:, :, 2] - 128) * factor + 128), 0, 255)
        
        # 饱和度调整
        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        
        # 色调调整
        if self.hue > 0:
            delta = random.uniform(-self.hue, self.hue) * 180
            hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
        
        # 转回RGB
        hsv = hsv.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return result


class Normalize:
    """
    医学影像归一化
    
    将图像像素值归一化到[-1, 1]或[0, 1]范围
    支持CT/MRI等医学影像的特殊处理
    """
    
    def __init__(
        self,
        mean: Union[float, List[float]] = [0.485, 0.456, 0.406],
        std: Union[float, List[float]] = [0.229, 0.224, 0.225],
        min_val: float = 0.0,
        max_val: float = 1.0,
    ):
        """
        Args:
            mean: 均值 (ImageNet标准)
            std: 标准差
            min_val: 输入最小值
            max_val: 输入最大值
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        执行归一化
        
        Args:
            image: 输入图像 [H, W, C] uint8 或 [H, W] 灰度
        
        Returns:
            tensor: 归一化后的tensor [C, H, W]
        """
        # 转为float32
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # 灰度图转RGB
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # 归一化到[0, 1]
        if self.max_val > self.min_val:
            image = (image - self.min_val) / (self.max_val - self.min_val)
        
        # 标准化 (ImageNet)
        image = (image - self.mean) / self.std
        
        # 转为tensor [C, H, W]
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return tensor


# =============================================================================
# 医学影像专用增强
# =============================================================================

class CTWindowAdjustment:
    """
    CT窗宽窗位调整
    
    CT图像需要根据诊断目的选择不同的窗宽窗位:
    - 骨窗 (Bone): WW=2000, WL=500
    - 肺窗 (Lung): WW=1500, WL=-600
    - 软组织窗 (Soft): WW=400, WL=50
    - 脑窗 (Brain): WW=80, WL=40
    - 肝脏窗 (Liver): WW=150, WL=30
    
    窗宽 (Window Width): 图像对比度范围
    窗位 (Window Level): 图像亮度中心
    """
    
    def __init__(
        self,
        presets: Optional[Dict[str, Tuple[int, int]]] = None,
        prob: float = 0.5,
    ):
        """
        Args:
            presets: 窗宽窗位预设字典 {name: (width, level)}
            prob: 应用概率
        """
        self.presets = presets or {
            'bone': (2000, 500),
            'lung': (1500, -600),
            'soft': (400, 50),
            'brain': (80, 40),
            'liver': (150, 30),
        }
        self.prob = prob
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        执行窗宽窗位调整
        
        Args:
            image: 输入CT图像 [H, W] 或 [H, W, C] (假设为原始HU值或归一化值)
        
        Returns:
            image: 调整后的图像
        """
        if random.random() > self.prob:
            return image
        
        # 随机选择预设
        preset_name = random.choice(list(self.presets.keys()))
        window_width, window_level = self.presets[preset_name]
        
        # 执行窗宽窗位调整
        adjusted = self._apply_window(image, window_width, window_level)
        
        return adjusted
    
    def _apply_window(
        self,
        image: np.ndarray,
        window_width: int,
        window_level: int,
    ) -> np.ndarray:
        """
        应用窗宽窗位
        
        Args:
            image: 输入图像
            window_width: 窗宽
            window_level: 窗位
        
        Returns:
            adjusted: 调整后的图像 uint8
        """
        # 计算窗口边界
        window_min = window_level - window_width / 2
        window_max = window_level + window_width / 2
        
        # 裁剪到窗口范围
        image = np.clip(image, window_min, window_max)
        
        # 归一化到[0, 255]
        image = ((image - window_min) / window_width * 255).astype(np.uint8)
        
        return image


class MRIntensityNormalization:
    """
    MRI强度归一化
    
    MRI图像需要特殊处理:
    - bias field correction (偏置场校正)
    - intensity normalization (强度归一化)
    
    这里实现简化的强度归一化
    """
    
    def __init__(self, clip_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Args:
            clip_range: 裁剪范围
        """
        self.clip_range = clip_range
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        执行MRI归一化
        
        Args:
            image: 输入MRI图像
        
        Returns:
            normalized: 归一化后的图像
        """
        # 转换为float
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # 裁剪
        image = np.clip(image, self.clip_range[0], self.clip_range[1])
        
        # 减均值除标准差
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        
        return image


# =============================================================================
# Mixup / Cutmix增强
# =============================================================================

class Mixup:
    """
    Mixup数据增强
    
    将两个样本及其标签按比例混合
    公式: x = λ*x1 + (1-λ)*x2, y = λ*y1 + (1-λ)*y2
    
    适用于分类任务, 可提高模型泛化能力
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Args:
            alpha: Beta分布参数
            prob: 应用概率
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        batch_images: torch.Tensor,
        batch_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        执行Mixup
        
        Args:
            batch_images: 批次图像 [B, C, H, W]
            batch_labels: 批次标签 [B] 或 [B, num_classes]
        
        Returns:
            mixed_images: 混合后的图像
            labels_a: 原始标签
            labels_b: 混合标签
            lam: 混合比例
        """
        if random.random() > self.prob:
            return batch_images, batch_labels, batch_labels, 1.0
        
        batch_size = batch_images.size(0)
        
        if batch_size < 2:
            return batch_images, batch_labels, batch_labels, 1.0
        
        # 生成混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # 确保lam >= 0.5
        
        # 随机选择配对
        indices = torch.randperm(batch_size)
        
        # 混合图像
        mixed_images = lam * batch_images + (1 - lam) * batch_images[indices]
        
        # 混合标签 (one-hot或类别)
        if batch_labels.dim() == 1:
            # 类别标签
            labels_a = batch_labels
            labels_b = batch_labels[indices]
        else:
            # one-hot标签
            labels_a = batch_labels
            labels_b = batch_labels[indices]
        
        return mixed_images, labels_a, labels_b, lam


class Cutmix:
    """
    Cutmix数据增强
    
    将两个样本的区域混合
    公式: x = λ*x1 + (1-λ)*x2, 从区域中裁剪拼贴
    
    适用于分类任务, 效果通常优于Mixup
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Args:
            alpha: Beta分布参数
            prob: 应用概率
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        batch_images: torch.Tensor,
        batch_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        执行Cutmix
        
        Args:
            batch_images: 批次图像 [B, C, H, W]
            batch_labels: 批次标签
        
        Returns:
            mixed_images: 混合后的图像
            labels_a: 原始标签
            labels_b: 混合标签
            lam: 混合比例 (实际为裁剪区域比例)
        """
        if random.random() > self.prob:
            return batch_images, batch_labels, batch_labels, 1.0
        
        batch_size = batch_images.size(0)
        
        if batch_size < 2:
            return batch_images, batch_labels, batch_labels, 1.0
        
        # 计算裁剪区域比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 随机选择配对
        indices = torch.randperm(batch_size)
        
        # 计算裁剪区域
        _, _, h, w = batch_images.shape
        
        # 随机裁剪中心
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # 执行裁剪拼贴
        mixed_images = batch_images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = batch_images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # 调整lam为实际裁剪比例
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        # 标签
        if batch_labels.dim() == 1:
            labels_a = batch_labels
            labels_b = batch_labels[indices]
        else:
            labels_a = batch_labels
            labels_b = batch_labels[indices]
        
        return mixed_images, labels_a, labels_b, lam


# =============================================================================
# 分割任务专用增强
# =============================================================================

class ElasticDeformation:
    """
    弹性形变增强 (Elastic Deformation)
    
    医学影像分割中常用的数据增强
    对图像和mask同时应用弹性变换
    """
    
    def __init__(
        self,
        alpha: int = 1000,
        sigma: int = 30,
        prob: float = 0.5,
    ):
        """
        Args:
            alpha: 形变幅度
            sigma: 平滑因子
            prob: 应用概率
        """
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob
    
    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        执行弹性形变
        
        Args:
            image: 输入图像
            mask: 对应的mask (可选)
        
        Returns:
            image: 变形后的图像
            mask: 变形后的mask (如果提供)
        """
        if random.random() > self.prob:
            if mask is not None:
                return image, mask
            return image
        
        from scipy.ndimage import gaussian_filter, map_coordinates
        
        shape = image.shape[:2]
        
        # 生成随机位移场
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), self.sigma
        ) * self.alpha
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), self.sigma
        ) * self.alpha
        
        # 创建坐标网格
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
        
        # 对图像应用变形
        if len(image.shape) == 3:
            channels = []
            for c in range(image.shape[2]):
                channels.append(
                    map_coordinates(image[:, :, c], indices, order=1).reshape(shape)
                )
            image = np.stack(channels, axis=-1)
        else:
            image = map_coordinates(image, indices, order=1).reshape(shape)
        
        # 对mask应用变形
        if mask is not None:
            if len(mask.shape) == 3:
                mask_channels = []
                for c in range(mask.shape[2]):
                    mask_channels.append(
                        map_coordinates(mask[:, :, c], indices, order=0).reshape(shape)
                    )
                mask = np.stack(mask_channels, axis=-1)
            else:
                mask = map_coordinates(mask, indices, order=0).reshape(shape)
            
            return image, mask
        
        return image


# =============================================================================
# 增强组合
# =============================================================================

class Compose:
    """
    组合多个数据增强
    """
    
    def __init__(self, transforms: List[Callable]):
        """
        Args:
            transforms: 变换列表
        """
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        依次应用所有变换
        
        Args:
            image: 输入图像
        
        Returns:
            image: 增强后的图像
        """
        for t in self.transforms:
            image = t(image)
        
        return image


class MedicalAugmentation:
    """
    医学影像增强流水线
    
    预定义的增强组合
    """
    
    # 训练增强
    TRAIN = [
        ('flip', RandomFlip(prob=0.5)),
        ('crop', RandomCrop(crop_size=224, prob=0.3)),
        ('color_jitter', ColorJitter(brightness=0.2, contrast=0.2)),
        ('normalize', Normalize()),
    ]
    
    # CT训练增强
    TRAIN_CT = [
        ('flip', RandomFlip(prob=0.5)),
        ('ct_window', CTWindowAdjustment(prob=0.5)),
        ('normalize', Normalize()),
    ]
    
    # 验证增强
    VAL = [
        ('normalize', Normalize()),
    ]


def get_augmentation_pipeline(
    mode: str = "train",
    use_ct: bool = False,
    custom_transforms: Optional[List[Callable]] = None,
) -> Callable:
    """
    获取增强流水线
    
    Args:
        mode: 模式 (train/val/test)
        use_ct: 是否使用CT窗宽窗位调整
        custom_transforms: 自定义变换列表
    
    Returns:
        pipeline: 增强流水线
    """
    if custom_transforms:
        return Compose(custom_transforms)
    
    if mode == "train":
        if use_ct:
            transforms = [t for _, t in MedicalAugmentation.TRAIN_CT]
        else:
            transforms = [t for _, t in MedicalAugmentation.TRAIN]
    else:
        transforms = [t for _, t in MedicalAugmentation.VAL]
    
    return Compose(transforms)


# Public API alias.
# get_train_transforms is the conventional name expected by tests / external
# scripts; it maps to the train-mode pipeline of get_augmentation_pipeline.
def get_train_transforms(
    use_ct: bool = False,
    custom_transforms: Optional[List[Callable]] = None,
) -> Callable:
    """Get the training augmentation pipeline."""
    return get_augmentation_pipeline(
        mode="train",
        use_ct=use_ct,
        custom_transforms=custom_transforms,
    )