+++
date = '2026-01-17T18:28:52+08:00'
draft = true
title = '复现 Faster R-CNN：从理论到实践'
+++

最近在准备论文时，我意识到自己对深度学习项目的理解还不够深入。为了加强实际项目经验，我决定复现几个经典的深度学习算法作为练习。通过实际操作，可以更好地理解算法的实现细节和优化策略，从而提高编程能力和项目经验。

经过考虑，我选择从目标检测领域的经典算法 Faster R-CNN 开始。相比基础的卷积神经网络，Faster R-CNN 更接近实际应用场景，同时又不像 YOLO 系列那样复杂（YOLO 系列计划后续复现）。至于 NLP 领域的算法，目前看来对算力和数据处理要求较高，暂时留待以后探索。

Faster R-CNN 是一种基于区域提议网络（Region Proposal Network，RPN）的目标检测算法。它通过 RPN 生成候选区域，然后使用卷积神经网络（CNN）对这些区域进行分类和边界框回归，从而实现高效准确的目标检测。

## 数据集

使用 PASCAL VOC 2007 数据集，包含 20 个物体类别，实际训练时添加背景类作为第 21 类。

数据预处理策略：
- 图像缩放：短边不超过 600 像素，长边等比例缩放
- 边界框调整：图像缩放时，对应的边界框坐标同步进行相同比例的变换
- 标准化：将图像张量化并进行标准化处理
- 数据增强：仅使用上述基础处理，未添加其他复杂的增强方法

```python
def get_transforms(
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Callable[[Image.Image], torch.Tensor]:
    """Get transforms for image preprocessing.

    Args:
        mean (Tuple[float, float, float]): Mean values for normalization.
        std (Tuple[float, float, float]): Standard deviation values for normalization.

    Returns:
        Callable[[Image.Image], torch.Tensor]: A callable that applies the transforms to an image.
    """

    def resize_shorter_side(image: Image.Image) -> Image.Image:
        """Resize image so that the shorter side is 600 pixels long."""
        width, height = image.size

        if width < height:
            new_width = 600
            new_height = int(height * 600 / width)
        else:
            new_height = 600
            new_width = int(width * 600 / height)

        new_width = int(new_width)
        new_height = int(new_height)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return transforms.Compose(
        [
            resize_shorter_side,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
```

```python
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from faster_rcnn.data.transforms import get_transforms


class PascalVOC(Dataset):
    def __init__(
        self, root_dir: str | Path, train: bool, transform: Optional[Callable] = None
    ):
        """
        Initialize the PascalVOC dataset.

        Args:
            root_dir: Root directory of the dataset.
            train: Whether to load the training or validation set.
            transform: Optional transform to be applied on the image.
        """
        self.root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir
        self.transform = transform
        self.train = train

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        # Load class names from JSON file
        class_names_path = self.root_dir / "class_names.json"
        if not class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found: {class_names_path}")

        with open(class_names_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

        # 创建从类别名到索引的映射
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load image paths from text file
        text_file = (
            self.root_dir / "ImageSets/Main/train.txt"
            if train
            else self.root_dir / "ImageSets/Main/val.txt"
        )

        if not text_file.exists():
            raise FileNotFoundError(f"Image list file not found: {text_file}")

        with open(text_file, "r", encoding="utf-8") as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]

        self._validate_files()

    def _validate_files(self):
        """验证所有图像和标注文件是否存在"""
        missing_files = []

        for img_name in self.image_paths:
            # 检查图像文件
            img_path = self.root_dir / "JPEGImages" / (img_name + ".jpg")
            if not img_path.exists():
                missing_files.append(str(img_path))

            # 检查标注文件
            xml_path = self.root_dir / "Annotations" / (img_name + ".xml")
            if not xml_path.exists():
                missing_files.append(str(xml_path))

        if missing_files:
            print(f"警告: 找到 {len(missing_files)} 个缺失文件")
            if len(missing_files) <= 10:
                for f in missing_files[:10]:
                    print(f"  - {f}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.types.Tensor, Dict[str, Any]]:
        """Get item from dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Tuple containing the image tensor and target dictionary.
        """
        img_name = self.image_paths[idx]

        image_path = self.root_dir / "JPEGImages" / (img_name + ".jpg")
        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        xml_path = self.root_dir / "Annotations" / (img_name + ".xml")
        boxes, labels = self._parse_xml(xml_path)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
            boxes_tensor[:, 3] - boxes_tensor[:, 1]
        )
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor(
                [original_height, original_width], dtype=torch.int64
            ),
        }

        if self.transform:
            image, target = self._apply_transform_with_boxes(image, target)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image, target

    def _apply_transform_with_boxes(self, image: Image.Image, target: Dict[str, Any]):
        """
        应用transform并相应地调整边界框。
        """
        orig_width, orig_height = image.size

        if self.transform:
            image = self.transform(image)

        if isinstance(image, torch.Tensor):
            new_height, new_width = image.shape[1], image.shape[2]
        else:
            new_width, new_height = image.size

        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        boxes = target["boxes"].clone()
        boxes[:, 0] *= scale_x  # xmin
        boxes[:, 1] *= scale_y  # ymin
        boxes[:, 2] *= scale_x  # xmax
        boxes[:, 3] *= scale_y  # ymax

        target["boxes"] = boxes
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target["size"] = torch.tensor([new_height, new_width], dtype=torch.int64)

        return image, target

    def _parse_xml(self, xml_path: Path) -> Tuple[List[List[float]], List[int]]:
        """
        Parse XML file and extract bounding boxes and labels.

        Args:
            xml_path: Path to the XML file.

        Returns:
            Tuple containing bounding boxes and labels.
        """
        if not xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            boxes = []
            labels = []

            for obj in root.findall("object"):
                difficult = obj.find("difficult")
                if difficult is not None and difficult.text == "1":
                    continue

                bbox = obj.find("bndbox")
                if bbox is None:
                    continue

                try:
                    xmin = float(bbox.find("xmin").text)  # type: ignore
                    ymin = float(bbox.find("ymin").text)  # type: ignore
                    xmax = float(bbox.find("xmax").text)  # type: ignore
                    ymax = float(bbox.find("ymax").text)  # type: ignore

                    if xmin >= xmax or ymin >= ymax:
                        print(
                            f"警告: 无效边界框 {xmin},{ymin},{xmax},{ymax} 在 {xml_path}"
                        )
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])

                    name_elem = obj.find("name")
                    if name_elem is None:
                        print(f"警告: 对象没有名称在 {xml_path}")
                        continue

                    label_name = name_elem.text
                    if label_name not in self.class_to_idx:
                        print(f"警告: 未知类别 '{label_name}' 在 {xml_path}")
                        continue

                    labels.append(self.class_to_idx[label_name])

                except (AttributeError, ValueError) as e:
                    print(f"警告: 解析边界框时出错在 {xml_path}: {e}")
                    continue

            return boxes, labels

        except ET.ParseError as e:
            raise RuntimeError(f"解析XML文件失败 {xml_path}: {e}")

    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.class_names

    def get_class_to_idx(self) -> Dict[str, int]:
        """获取类别到索引的映射"""
        return self.class_to_idx.copy()


```

## 网络架构

### 特征提取网络

使用 VGG16 作为骨干网络：
- 移除原始的分类头（全连接层）
- 冻结前 3 个卷积层的参数（迁移学习策略）
- 输出特征图尺寸为输入图像的 1/16

* 前向传播维度：
  - 输入：`(batch_size, 3, H, W)`，其中 `min(H, W) ≤ 600`
  - 输出：`(batch_size, 512, H/16, W/16)`
  
```python
"""Shared Convolutional Backbone: VGG-16: 13 convolutional layers (more commonly used)"""

import torch.nn as nn
from torchvision import models


def vgg16_backbone():
    """VGG-16 backbone"""
    backbone = models.vgg16(pretrained=True).features
    shared_backbone = nn.Sequential(*list(backbone.children())[:-1])

    # Freeze first 10 conv layers (conv1_1 through conv4_3)
    # These are the convolutional layers at indices: 0, 2, 5, 7, 10, 12, 14, 17, 19, 21
    conv_indices_to_freeze = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]

    for idx in conv_indices_to_freeze:
        layer = shared_backbone[idx]
        if isinstance(layer, nn.Conv2d):
            for param in layer.parameters():
                param.requires_grad = False

    return shared_backbone

```

### 区域提议网络（RPN）

RPN 是 Faster R-CNN 的核心组件，负责生成高质量的候选区域：

1. **滑动窗口**：使用 3×3 卷积核，填充为 1
2. **双分支结构**：
   - 前景/背景分类分支：1×1 卷积，无填充
   - 边界框回归分支：1×1 卷积，无填充
3. **锚框生成**：为特征图的每个位置生成 9 个锚框（3 种尺度 × 3 种宽高比）
4. **区域提议生成**：基于分类得分和回归偏移生成候选区域

* 前向传播流程：
  - 输入特征：`(batch_size, 512, H/16, W/16)`
  - 3×3 卷积：`(batch_size, mid_channels, H/16, W/16)`
  - 分类层：`(batch_size, num_anchors×2, H/16, W/16)`
  - 回归层：`(batch_size, num_anchors×4, H/16, W/16)`
  - 重塑维度：
    - 分类得分：`(batch_size, H/16×W/16×num_anchors, 2)`
    - 回归偏移：`(batch_size, H/16×W/16×num_anchors, 4)`

```python
class RPN(nn.Module):
    """Region Proposal Network (RPN) for object detection."""

    def __init__(
        self,
        input_channels: int,
        mid_channels: int,
        num_anchors: int,
        im_size: Tuple[int, int],
        min_size: int = 16,
        nms_thresh: float = 0.7,
        base_size: int = 16,
        stride: int = 16,
        scale: float = 1.0,
        scales=[8, 16, 32],
        ratios=[0.5, 1, 1.5],
        num_sample_before_nms: int = 2000,
        num_sample_after_nms: int = 2000,
        device="cpu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, mid_channels, kernel_size=3, padding=1)

        self.cls_layer = nn.Conv2d(mid_channels, num_anchors * 2, kernel_size=1)
        self.reg_layer = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1)

        self.anchor_generator = AnchorGenerator(
            base_size, scales, ratios, stride, device
        )

        self.proposal_generator = ProposalGenerator(
            im_size,
            min_size,
            nms_thresh,
            scale,
            num_sample_before_nms,
            num_sample_after_nms,
        )

        self.relu = nn.ReLU()

    def forward(self, feature):
        batch_size = feature.size(0)

        # output: [batch_size, mid_channels, height, width]
        x = self.relu(self.conv(feature))

        # output: [batch_size, num_anchors * 2, height, width]
        cls_logits = self.cls_layer(x)
        # output: [batch_size, height, width, num_anchors * 2]
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        # output: [batch_size, num_anchors * height * width, 2]
        cls_logits = cls_logits.view(batch_size, -1, 2)

        # output: [batch_size, num_anchors * 4, height, width]
        reg_logits = self.reg_layer(x)
        # output: [batch_size, height, width, num_anchors * 4]
        reg_logits = reg_logits.permute(0, 2, 3, 1).contiguous()
        # output: [batch_size, num_anchors * height * width, 4]
        reg_logits = reg_logits.view(batch_size, -1, 4)

        anchors = self.anchor_generator(feature)
        proposals = self.proposal_generator(anchors, cls_logits, reg_logits)

        return cls_logits, proposals

```
#### 锚框生成器（AnchorGenerator）

##### 基础锚框计算
- 基础尺寸：`base_size = 16`（对应特征图下采样率）
- 锚框高度：`anchor_h = base_size × scale × √ratio`
- 锚框宽度：`anchor_w = base_size × scale ÷ √ratio`

##### 所有锚框生成
通过特征图上的每个位置生成基础锚框：
1. **中心坐标计算**：特征图位置 `(i, j)` 对应原始图像的中心坐标为：
   - `中心x = (j + 0.5) × base_size`
   - `中心y = (i + 0.5) × base_size`
   （其中 `base_size = 16`，`+0.5` 确保锚框中心位于特征图对应区域的中心）

2. **空间偏移**：为每个中心位置生成 9 个不同尺度和宽高比的锚框
3. **总锚框数**：`H/16 × W/16 × 9`

* 前向传播：
  - 输入特征图尺寸：`(batch_size, H/16, W/16)`
  - 输出锚框：`(batch_size, H/16×W/16×9, 4)`，其中 4 维表示 `[中心x, 中心y, 高度, 宽度]`
  （中心坐标基于特征图位置计算得出，高度和宽度由尺度和宽高比决定）
  
```python
class AnchorGenerator(nn.Module):
    """Generate anchors for a given feature map size."""

    def __init__(self, base_size, scales, ratios, stride, device: torch.types.Device):
        super().__init__()
        self.base_size = base_size
        self.scales = scales
        self.ratios = ratios
        self.stride = stride
        self.device = device

        self.base_anchor_boxes = self._generate_base_anchor_boxes()

    def forward(self, feature):
        batch_size = feature.size(0)
        height, width = feature.size(2), feature.size(3)

        anchor_boxes = self._generate_all_anchors(self.base_anchor_boxes, width, height)
        anchor_boxes = anchor_boxes.unsqueeze(0).expand(batch_size, -1, -1)

        return anchor_boxes

    def _generate_base_anchor_boxes(self) -> torch.Tensor:
        """Generate the base anchor boxes for feature map

        Return:
            base_anchor_boxes(torch.Tensor): shape[num_scales * num_ratios, 4], [:, x1, y1, x2, y2]
        """
        num_anchors = len(self.scales) * len(self.ratios)
        base_anchor_boxes = torch.zeros(
            (num_anchors, 4), dtype=torch.float32, device=self.device
        )

        cx, cy = (self.base_size - 1) / 2.0

        anchor_idx = 0
        for scale in self.scales:
            for ratio in self.ratios:
                w = self.base_size * scale * math.sqrt(ratio)
                h = self.base_size * scale / math.sqrt(ratio)
                base_anchor_boxes[anchor_idx, 0] = cx - w / 2.0
                base_anchor_boxes[anchor_idx, 1] = cy - h / 2.0
                base_anchor_boxes[anchor_idx, 2] = cx + w / 2.0
                base_anchor_boxes[anchor_idx, 3] = cy + h / 2.0
                anchor_idx += 1

        return base_anchor_boxes

    def _generate_all_anchors(self, base_anchor_boxes, width, height) -> torch.Tensor:
        """Generate all anchors for any position in feature map
        Args:
            base_anchor_boxes(torch.tensor): [num_anchors, 4]
            width(int): The width of feature map
            height(int): The height of feature map
        Return:
            anchors(torch.tensor): [num_positions * num_anchors, 4]
        """
        shift_x = (
            torch.arange(0, width, dtype=torch.float32, device=self.device)
            * self.stride
        )
        shift_y = (
            torch.arange(0, height, dtype=torch.float32, device=self.device)
            * self.stride
        )

        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing="xy")

        shift_x_flat = shift_x.reshape(-1)
        shift_y_flat = shift_y.reshape(-1)

        shifts = torch.stack(
            (shift_x_flat, shift_y_flat, shift_x_flat, shift_y_flat), dim=1
        )

        # [1, num_anchors, 4] + [num_positions, 1, 4] = [num_positions, num_anchors, 4]
        anchors = base_anchor_boxes.unsqueeze(0) + shifts.unsqueeze(1)

        # [num_positions * num_anchors, 4]
        anchors = anchors.reshape(-1, 4)

        return anchors

```

#### 提议生成器（ProposalGenerator）

* 前向传播：
  - 输入锚框：`(batch_size, H/16×W/16×9, 4)`
  - 输入分类得分和回归偏移
  - 应用回归偏移修正锚框：`fix_anchors(reg_logits, input_anchor)`
    → `(batch_size, H/16×W/16×9, 4)`，修正后的边界框

* 后处理步骤：
  1. 根据前景置信度排序，选择前 N 个候选（N ≤ 2000）
  2. 应用非极大值抑制（NMS）去除重叠的候选框
  3. 最终输出约 2000 个高质量的区域提议

* 最终输出：`(batch_size, N, 4)`，其中 N ≤ 2000，4 维表示 `[中心x, 中心y, 高度, 宽度]`
```python
class ProposalGenerator(nn.Module):
    def __init__(
        self,
        im_size: Tuple[int, int],
        min_size: int = 16,
        nms_thresh: float = 0.7,
        scale: float = 1.0,
        num_sample_before_nms: int = 2000,
        num_sample_after_nms: int = 2000,
    ):
        super().__init__()
        self.im_size = im_size
        self.scale = scale
        self.min_size = min_size
        self.nms_thresh = nms_thresh
        self.num_sample_before_nms = num_sample_before_nms
        self.num_sample_after_nms = num_sample_after_nms

    def forward(
        self,
        anchors: torch.Tensor,
        cls_logits: torch.Tensor,
        reg_logits: torch.Tensor,
    ):
        """Get proposal
        Args:
            anchors(torch.Tensor): [batch_size, num_position * num_anchors, 4]
            cls_logits(torch.Tensor): [batch_size, num_anchors * height * width, 2]
            reg_logits(torch.Tensor): [batch_size, num_anchors * height * width, 4]
        Return:
            proposals(torch.Tensor): [batch_size, num_sample_after_nms, 4]
        """

        # 添加softmax得到概率分数
        cls_score = torch.softmax(cls_logits, dim=-1)

        batch_size = anchors.size(0)
        proposals_list = []

        for i in range(batch_size):
            batch_anchors = anchors[i]  # [num_anchors * num_positions, 4]
            batch_reg_logits = reg_logits[i]  # [num_anchors * num_positions, 4]
            batch_cls_score = cls_score[i]  # [num_anchors * num_positions, 2]

            # 1. 解码边界框
            proposal = decode_boxes(
                batch_reg_logits, batch_anchors
            )  # [num_anchors * num_positions, 4]

            # 2. 限制边界框在图像范围内
            im_width, im_height = self.im_size
            proposal[:, [0, 2]] = torch.clamp(
                proposal[:, [0, 2]], min=0, max=im_width - 1
            )
            proposal[:, [1, 3]] = torch.clamp(
                proposal[:, [1, 3]], min=0, max=im_height - 1
            )

            # 3. 移除尺寸太小的边界框
            min_size = self.min_size * self.scale
            proposal_w = proposal[:, 2] - proposal[:, 0] + 1  # 需要+1
            proposal_h = proposal[:, 3] - proposal[:, 1] + 1  # 需要+1

            is_valid = (proposal_w >= min_size) & (proposal_h >= min_size)

            # 只保留有效的proposals和对应的分数
            valid_proposals = proposal[is_valid]
            valid_cls_score = batch_cls_score[is_valid, :]  # 修复维度索引

            # 4. 获取前景分数（第1个通道是前景，第0个是背景）
            if valid_cls_score.size(0) > 0:
                # 获取前景分数
                fg_scores = valid_cls_score[:, 1]
            else:
                fg_scores = torch.tensor(
                    [], dtype=torch.float32, device=valid_proposals.device
                )

            # 5. 按前景分数降序排序
            if len(fg_scores) > 0:
                sorted_indices = torch.argsort(fg_scores, descending=True)

                # 6. 保留前num_sample_before_nms个
                if (
                    self.num_sample_before_nms > 0
                    and len(sorted_indices) > self.num_sample_before_nms
                ):
                    sorted_indices = sorted_indices[: self.num_sample_before_nms]

                sorted_proposals = valid_proposals[sorted_indices]
                sorted_scores = fg_scores[sorted_indices]

                # 7. 应用NMS
                if len(sorted_proposals) > 0:
                    keep = nms(sorted_proposals, sorted_scores, self.nms_thresh)

                    # 8. 如果数量不足则随机抽取填补
                    if len(keep) < self.num_sample_after_nms:
                        num_needed = self.num_sample_after_nms - len(keep)
                        if len(keep) > 0:
                            # 从已保留的框中随机选择
                            random_indices = torch.randint(
                                0, len(keep), (num_needed,), device=keep.device
                            )
                            keep = torch.cat([keep, keep[random_indices]])
                        else:
                            # 如果没有保留任何框，创建空tensor
                            device = sorted_proposals.device
                            keep = torch.tensor([], dtype=torch.long, device=device)

                    # 9. 截取固定数量的边界框
                    if len(keep) > self.num_sample_after_nms:
                        keep = keep[: self.num_sample_after_nms]

                    if len(keep) > 0:
                        batch_proposals = sorted_proposals[keep]
                    else:
                        # 如果没有proposals，创建空的tensor
                        device = sorted_proposals.device
                        batch_proposals = torch.zeros(
                            (0, 4), dtype=torch.float32, device=device
                        )
                else:
                    device = valid_proposals.device
                    batch_proposals = torch.zeros(
                        (0, 4), dtype=torch.float32, device=device
                    )
            else:
                device = valid_proposals.device
                batch_proposals = torch.zeros(
                    (0, 4), dtype=torch.float32, device=device
                )

            # 10. 确保每个batch有固定数量的proposals
            if batch_proposals.size(0) < self.num_sample_after_nms:
                # 填充到固定数量
                num_to_pad = self.num_sample_after_nms - batch_proposals.size(0)
                if batch_proposals.size(0) > 0:
                    # 随机复制现有的proposals
                    pad_indices = torch.randint(
                        0,
                        batch_proposals.size(0),
                        (num_to_pad,),
                        device=batch_proposals.device,
                    )
                    padding = batch_proposals[pad_indices]
                else:
                    # 如果没有proposals，创建零填充
                    padding = torch.zeros(
                        (num_to_pad, 4),
                        dtype=torch.float32,
                        device=batch_proposals.device,
                    )
                batch_proposals = torch.cat([batch_proposals, padding], dim=0)

            proposals_list.append(batch_proposals.unsqueeze(0))

        # 11. 合并所有batch的结果
        proposals = torch.cat(
            proposals_list, dim=0
        )  # [batch_size, num_sample_after_nms, 4]

        return proposals

```
#### RPN 训练

#### 损失函数
RPN 使用多任务损失函数：
```
loss = classification_loss + regression_loss
```
- 分类损失：交叉熵损失（Cross-Entropy Loss），区分前景和背景
- 回归损失：平滑 L1 损失（Smooth L1 Loss），优化边界框位置

> 平滑 L1 损失公式：
> ```
> smooth_L1(x) = {
>     0.5 × x²          if |x| < 1
>     |x| - 0.5        otherwise
> }
> ```

```python
def rpn_loss(cls_logit, proposal, target, lambda_: float = 0.5):
    """Compute RPN loss
    Args:
        cls_logit: [batch_size, H * W * num_anchors, 2]
        proposal: [batch_size, N, 4]
        target: [batch_size, N, 4]

    """
    cls_loss_func = nn.CrossEntropyLoss()
    reg_loss_func = nn.SmoothL1Loss()

    return cls_loss_func(cls_logit[:, :, 0]) + lambda_ * reg_loss_func(proposal, target)
```
