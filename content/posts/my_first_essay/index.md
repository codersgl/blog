+++
date = '2026-01-06T18:55:14+08:00'
draft = true
title = 'My_first_essay'
math = true
image = 'cover.jpeg'
+++

## 前情提要

导师让我发一篇期刊，参考论文和代码都有，让我改。时间所剩不多大概两个礼拜。不管怎样，先拆解一下任务。

- [ ] 改代码
- [ ] 跑代码
- [ ] 写论文


## 改代码

主要的修改思路是修改特征提取部分，让模型提取更多的语义信息以期提高模型的整体性能。具体来说是在原特征提取分支的基础上添加两个特征提取分支SAW2和Dinov2。听起来挺简单的，但具体操作还是不容易。先分析原始特征提取分支的输入输出吧。

### input

网络的输入可以从数据加载 (loading_data.py, SHHA.py) 和处理流程 (engine.py, collate_fn) 中推断出来。

简而言之，DSGCNet 网络的输入是一个标准的 4D PyTorch 张量，包含了一批经过预处理和归一化的 RGB 图像。

具体输入格式
1. 训练阶段 (Training)
在训练时，网络接收的输入张量形状通常为：

$$(B \times 4, 3, 128, 128)$$

* B (Batch Size): 由 train.py 中的 --batch_size 参数控制（默认 8）。
* 4 (Num Patches): 数据加载通过 random_crop 将每张原始图像裁剪为 4 个 patches。collate_fn_crowd_train 会将这 4 个 patch 展平并入 Batch 维度。
* 3 (Channels): RGB 通道。
* 128×128 (Resolution): 训练时使用固定大小的随机裁剪（在 SHHA.py 中硬编码为 128）。
> 随机裁剪的目的：从一张大图中随机裁剪 4 个不同的区域，相当于增加了训练数据的多样性。这有助于模型学习局部特征，防止过拟合，并提高模型对不同密度分布区域的鲁棒性。

数据类型: Float32。归一化: 使用 ImageNet 标准均值和方差：
Mean: [0.485, 0.456, 0.406]
Std: [0.229, 0.224, 0.225]

2. 测试/评估阶段 (Evaluation)
在评估时，通常 batch_size 设为 1（因为图像分辨率不一），输入形状为：

$$(1, 3, H, W)$$

* H,W: 原始图像的高度和宽度（为了处理不同尺寸，通常使用 Padding 或 Batch Size=1）。
> 注意: 这里的 H 和 W 会被处理成 128 的倍数（通过 padding），这在 misc.py 的 nested_tensor_from_tensor_list 中处理。

辅助输入 (用于 Loss 计算)
虽然网络前向传播 (model(samples)) 只接收图像张量，但在训练循环 (engine.py) 中，还需要以下数据来计算损失：
* Ground Truth Density Map (gt_dmap): 形状对应于输入图像，用于监督 Density Approximation 分支。
* Point Annotations (targets): 包含人员坐标点
(point)，用于监督 Representation Approximation 分支和分类/回归损失。

### Net info

#### DSGC-Net

![DSGCNet_architecture](DSGCNet_architecture.png "DSGCNet_architecture") 

这是DSGC-Net的网络定义
```python
class DSGCnet(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        num_anchor_points = row * line

        self.fusion_total = nn.Sequential(
            nn.Conv2d(3 * 256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.regression = RegressionModel(
            num_features_in=256, num_anchor_points=num_anchor_points
        )
        self.classification = ClassificationModel(
            num_features_in=256,
            num_classes=self.num_classes,
            num_anchor_points=num_anchor_points,
        )

        self.anchor_points = AnchorPoints(
            pyramid_levels=[
                3,
            ],
            row=row,
            line=line,
        )

        self.pa = Decoder_SPD_PAFPN(256, 512, 512)
        self.density_pred = Density_pred()
        self.density_gcn = DensityGCNProcessor(k=4)
        self.feature_gcn = FeatureGCNProcessor(k=4)
        self.alpha = nn.Parameter(
            torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)
        )

    def forward(self, samples: NestedTensor):
        features = self.backbone(samples)

        features_pa = self.pa([features[1], features[2], features[3]])

        batch_size = features[0].shape[0]
        density = self.density_pred(features_pa)
        density_gcn_feature = self.density_gcn(density, features_pa)
        feature_gcn_feature = self.feature_gcn(features_pa)
        feature_fl = (
            features_pa
            + self.alpha[0] * density_gcn_feature
            + self.alpha[1] * feature_gcn_feature
        )
        regression = self.regression(feature_fl) * 100
        classification = self.classification(feature_fl)
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        output_coord = regression + anchor_points
        output_class = classification
        out = {
            "pred_logits": output_class,
            "pred_points": output_coord,
            "density_out": density,
        }

        return out
```

运行下面的脚本，查看输入数据在网络中的流动情况

```python
import torch
from torchinfo import summary
from models.DSGCnet import DSGCnet
from models.backbone import Backbone_VGG

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn((1, 3, 128, 128)).to(device)
    backbone = Backbone_VGG("vgg16_bn", True).to(device)
    dsgc_net = DSGCnet(backbone).to(device)
    summary(dsgc_net, input_data=input)
```

输出如下:
```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
DSGCnet                                  [1, 1, 16, 16]            197,378
├─Backbone_VGG: 1-1                      [1, 128, 64, 64]          --
│    └─Sequential: 2-1                   [1, 128, 64, 64]          --
│    │    └─Conv2d: 3-1                  [1, 64, 128, 128]         1,792
│    │    └─BatchNorm2d: 3-2             [1, 64, 128, 128]         128
│    │    └─ReLU: 3-3                    [1, 64, 128, 128]         --
│    │    └─Conv2d: 3-4                  [1, 64, 128, 128]         36,928
│    │    └─BatchNorm2d: 3-5             [1, 64, 128, 128]         128
│    │    └─ReLU: 3-6                    [1, 64, 128, 128]         --
│    │    └─MaxPool2d: 3-7               [1, 64, 64, 64]           --
│    │    └─Conv2d: 3-8                  [1, 128, 64, 64]          73,856
│    │    └─BatchNorm2d: 3-9             [1, 128, 64, 64]          256
│    │    └─ReLU: 3-10                   [1, 128, 64, 64]          --
│    │    └─Conv2d: 3-11                 [1, 128, 64, 64]          147,584
│    │    └─BatchNorm2d: 3-12            [1, 128, 64, 64]          256
│    │    └─ReLU: 3-13                   [1, 128, 64, 64]          --
│    └─Sequential: 2-2                   [1, 256, 32, 32]          --
│    │    └─MaxPool2d: 3-14              [1, 128, 32, 32]          --
│    │    └─Conv2d: 3-15                 [1, 256, 32, 32]          295,168
│    │    └─BatchNorm2d: 3-16            [1, 256, 32, 32]          512
│    │    └─ReLU: 3-17                   [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-18                 [1, 256, 32, 32]          590,080
│    │    └─BatchNorm2d: 3-19            [1, 256, 32, 32]          512
│    │    └─ReLU: 3-20                   [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-21                 [1, 256, 32, 32]          590,080
│    │    └─BatchNorm2d: 3-22            [1, 256, 32, 32]          512
│    │    └─ReLU: 3-23                   [1, 256, 32, 32]          --
│    └─Sequential: 2-3                   [1, 512, 16, 16]          --
│    │    └─MaxPool2d: 3-24              [1, 256, 16, 16]          --
│    │    └─Conv2d: 3-25                 [1, 512, 16, 16]          1,180,160
│    │    └─BatchNorm2d: 3-26            [1, 512, 16, 16]          1,024
│    │    └─ReLU: 3-27                   [1, 512, 16, 16]          --
│    │    └─Conv2d: 3-28                 [1, 512, 16, 16]          2,359,808
│    │    └─BatchNorm2d: 3-29            [1, 512, 16, 16]          1,024
│    │    └─ReLU: 3-30                   [1, 512, 16, 16]          --
│    │    └─Conv2d: 3-31                 [1, 512, 16, 16]          2,359,808
│    │    └─BatchNorm2d: 3-32            [1, 512, 16, 16]          1,024
│    │    └─ReLU: 3-33                   [1, 512, 16, 16]          --
│    └─Sequential: 2-4                   [1, 512, 8, 8]            --
│    │    └─MaxPool2d: 3-34              [1, 512, 8, 8]            --
│    │    └─Conv2d: 3-35                 [1, 512, 8, 8]            2,359,808
│    │    └─BatchNorm2d: 3-36            [1, 512, 8, 8]            1,024
│    │    └─ReLU: 3-37                   [1, 512, 8, 8]            --
│    │    └─Conv2d: 3-38                 [1, 512, 8, 8]            2,359,808
│    │    └─BatchNorm2d: 3-39            [1, 512, 8, 8]            1,024
│    │    └─ReLU: 3-40                   [1, 512, 8, 8]            --
│    │    └─Conv2d: 3-41                 [1, 512, 8, 8]            2,359,808
│    │    └─BatchNorm2d: 3-42            [1, 512, 8, 8]            1,024
│    │    └─ReLU: 3-43                   [1, 512, 8, 8]            --
├─Decoder_SPD_PAFPN: 1-2                 [1, 256, 16, 16]          --
│    └─Sequential: 2-5                   [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-44                 [1, 256, 8, 8]            131,328
│    │    └─BatchNorm2d: 3-45            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-46                   [1, 256, 8, 8]            --
│    └─Upsample: 2-6                     [1, 256, 16, 16]          --
│    └─Sequential: 2-7                   [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-47                 [1, 256, 8, 8]            590,080
│    │    └─BatchNorm2d: 3-48            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-49                   [1, 256, 8, 8]            --
│    └─Sequential: 2-8                   [1, 256, 16, 16]          --
│    │    └─Conv2d: 3-50                 [1, 256, 16, 16]          131,328
│    │    └─BatchNorm2d: 3-51            [1, 256, 16, 16]          512
│    │    └─ReLU: 3-52                   [1, 256, 16, 16]          --
│    └─Upsample: 2-9                     [1, 256, 32, 32]          --
│    └─Sequential: 2-10                  [1, 256, 16, 16]          --
│    │    └─Conv2d: 3-53                 [1, 256, 16, 16]          590,080
│    │    └─BatchNorm2d: 3-54            [1, 256, 16, 16]          512
│    │    └─ReLU: 3-55                   [1, 256, 16, 16]          --
│    └─Sequential: 2-11                  [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-56                 [1, 256, 32, 32]          65,792
│    │    └─BatchNorm2d: 3-57            [1, 256, 32, 32]          512
│    │    └─ReLU: 3-58                   [1, 256, 32, 32]          --
│    └─Sequential: 2-12                  [1, 256, 32, 32]          --
│    │    └─Conv2d: 3-59                 [1, 256, 32, 32]          590,080
│    │    └─BatchNorm2d: 3-60            [1, 256, 32, 32]          512
│    │    └─ReLU: 3-61                   [1, 256, 32, 32]          --
│    └─Sequential: 2-13                  [1, 256, 16, 16]          --
│    │    └─SPD: 3-62                    [1, 1024, 16, 16]         --
│    │    └─Conv2d: 3-63                 [1, 256, 16, 16]          262,400
│    │    └─BatchNorm2d: 3-64            [1, 256, 16, 16]          512
│    │    └─ReLU: 3-65                   [1, 256, 16, 16]          --
│    └─Sequential: 2-14                  [1, 256, 16, 16]          (recursive)
│    │    └─Conv2d: 3-66                 [1, 256, 16, 16]          (recursive)
│    │    └─BatchNorm2d: 3-67            [1, 256, 16, 16]          (recursive)
│    │    └─ReLU: 3-68                   [1, 256, 16, 16]          --
│    └─Sequential: 2-15                  [1, 256, 8, 8]            --
│    │    └─SPD: 3-69                    [1, 1024, 8, 8]           --
│    │    └─Conv2d: 3-70                 [1, 256, 8, 8]            262,400
│    │    └─BatchNorm2d: 3-71            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-72                   [1, 256, 8, 8]            --
│    └─Sequential: 2-16                  [1, 256, 8, 8]            (recursive)
│    │    └─Conv2d: 3-73                 [1, 256, 8, 8]            (recursive)
│    │    └─BatchNorm2d: 3-74            [1, 256, 8, 8]            (recursive)
│    │    └─ReLU: 3-75                   [1, 256, 8, 8]            --
│    └─Upsample: 2-17                    [1, 256, 16, 16]          --
│    └─Sequential: 2-18                  [1, 256, 16, 16]          --
│    │    └─Conv2d: 3-76                 [1, 256, 16, 16]          196,864
│    │    └─BatchNorm2d: 3-77            [1, 256, 16, 16]          512
│    │    └─ReLU: 3-78                   [1, 256, 16, 16]          --
├─Density_pred: 1-3                      [1, 1, 16, 16]            --
│    └─Sequential: 2-19                  [1, 256, 16, 16]          --
│    │    └─Conv2d: 3-79                 [1, 256, 16, 16]          590,080
│    │    └─BatchNorm2d: 3-80            [1, 256, 16, 16]          512
│    │    └─ReLU: 3-81                   [1, 256, 16, 16]          --
│    └─Sequential: 2-20                  [1, 256, 16, 16]          --
│    │    └─Conv2d: 3-82                 [1, 256, 16, 16]          590,080
│    │    └─BatchNorm2d: 3-83            [1, 256, 16, 16]          512
│    │    └─ReLU: 3-84                   [1, 256, 16, 16]          --
│    └─Sequential: 2-21                  [1, 256, 16, 16]          --
│    │    └─Conv2d: 3-85                 [1, 256, 16, 16]          590,080
│    │    └─BatchNorm2d: 3-86            [1, 256, 16, 16]          512
│    │    └─ReLU: 3-87                   [1, 256, 16, 16]          --
│    └─Sequential: 2-22                  [1, 1, 16, 16]            --
│    │    └─Conv2d: 3-88                 [1, 128, 16, 16]          295,040
│    │    └─BatchNorm2d: 3-89            [1, 128, 16, 16]          256
│    │    └─ReLU: 3-90                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-91                 [1, 64, 16, 16]           73,792
│    │    └─BatchNorm2d: 3-92            [1, 64, 16, 16]           128
│    │    └─ReLU: 3-93                   [1, 64, 16, 16]           --
│    │    └─Conv2d: 3-94                 [1, 1, 16, 16]            65
│    │    └─ReLU: 3-95                   [1, 1, 16, 16]            --
├─DensityGCNProcessor: 1-4               [1, 256, 16, 16]          --
│    └─GCNModel: 2-23                    [256, 256]                --
│    │    └─GCNConv: 3-96                [256, 512]                131,584
│    │    └─GCNConv: 3-97                [256, 256]                131,328
├─FeatureGCNProcessor: 1-5               [1, 256, 16, 16]          --
│    └─GCNModel: 2-24                    [256, 256]                --
│    │    └─GCNConv: 3-98                [256, 512]                131,584
│    │    └─GCNConv: 3-99                [256, 256]                131,328
├─RegressionModel: 1-6                   [1, 1024, 2]              1,180,160
│    └─Conv2d: 2-25                      [1, 256, 16, 16]          590,080
│    └─ReLU: 2-26                        [1, 256, 16, 16]          --
│    └─Conv2d: 2-27                      [1, 256, 16, 16]          590,080
│    └─ReLU: 2-28                        [1, 256, 16, 16]          --
│    └─Conv2d: 2-29                      [1, 8, 16, 16]            18,440
├─ClassificationModel: 1-7               [1, 1024, 2]              1,180,160
│    └─Conv2d: 2-30                      [1, 256, 16, 16]          590,080
│    └─ReLU: 2-31                        [1, 256, 16, 16]          --
│    └─Conv2d: 2-32                      [1, 256, 16, 16]          590,080
│    └─ReLU: 2-33                        [1, 256, 16, 16]          --
│    └─Conv2d: 2-34                      [1, 8, 16, 16]            18,440
├─AnchorPoints: 1-8                      [1, 1024, 2]              --
==========================================================================================
Total params: 25,169,875
Trainable params: 25,169,875
Non-trainable params: 0
Total mult-adds (G): 7.54
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 94.67
Params size (MB): 90.44
Estimated Total Size (MB): 185.31
==========================================================================================
```

我们重点关注特征提取部分，可以看到网络在经过特征提取和融合后的输出形状为`[1, 256, 16, 16]`，这是我们修改代码的关键，另外两条新增分支的最终形状也要如此。

#### DSU-Net

接下来观察一下`DSU-Net`的网络结构和输入输出：

![DSU-Net Architecture](DSU-Net.png "DSU-Net Architecture")

这是DGSU-Net的网络定义：
```python
class DGSUNet(nn.Module):
    def __init__(
        self,
        dino_model_name=None,
        dino_hub_dir=None,
        sam_config_file=None,
        sam_ckpt_path=None,
    ):
        super(DGSUNet, self).__init__()
        if dino_model_name is None:
            print("No model_name specified, using default")
            dino_model_name = "dinov2_vitl14"
        if dino_hub_dir is None:
            print("No dino_hub_dir specified, using default")
            dino_hub_dir = "facebookresearch/dinov2"
        if sam_config_file is None:
            print("No sam_config_file specified, using default")
            # Replace with your own SAM configuration file path
            sam_config_file = r"../sam2_configs/sam2.1_hiera_l.yaml"
        if sam_ckpt_path is None:
            print("No sam_ckpt_path specified, using default")
            # Replace with your own SAM pt file path
            sam_ckpt_path = r"checkpoints/sam2.1_hiera_large.pt"
        # Backbone Feature Extractor
        self.backbone_dino = dinov2_extract.DinoV2FeatureExtractor(
            dino_model_name, dino_hub_dir
        )
        self.backbone_sam = sam2hiera.sam2hiera(sam_config_file, sam_ckpt_path)
        # Feature Fusion
        self.fusion4 = CGAFusion.CGAFusion(1152)
        # (1024,37,37)->(1024,11,11)
        self.dino2sam_down4 = updown.interpolate_upsample(11)
        # (1024,11,11)->(1152,11,11)
        self.dino2sam_down14 = wtconv.DepthwiseSeparableConvWithWTConv2d(
            in_channels=1024, out_channels=1152
        )
        self.rfb1 = RFB.RFB_modified(144, 64)
        self.rfb2 = RFB.RFB_modified(288, 64)
        self.rfb3 = RFB.RFB_modified(576, 64)
        self.rfb4 = RFB.RFB_modified(1152, 64)
        self.decoder1 = sff.SFF(64)
        self.decoder2 = sff.SFF(64)
        self.decoder3 = sff.SFF(64)
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x_dino, x_sam):
        # Backbone Feature Extractor
        x1, x2, x3, x4 = self.backbone_sam(x_sam)
        x_dino = self.backbone_dino(x_dino)
        # change dino feature map size and dimension
        x_dino4 = self.dino2sam_down4(x_dino)
        x_dino4 = self.dino2sam_down14(x_dino4)
        # Feature Fusion(sam & dino)
        x4 = self.fusion4(x4, x_dino4)
        # change fusion feature map dimension->(64,11/22/44/88,11/22/44/88)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.decoder1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode="bilinear")
        x = self.decoder2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode="bilinear")
        x = self.decoder3(x, x1)
        out3 = F.interpolate(self.head(x), scale_factor=4, mode="bilinear")
        return out1, out2, out3
```

根据 DGSUNet.py 和测试脚本，输入形状规定如下：

1.  **`x_dino` (DiNOv2 输入)**: `(B, 3, 518, 518)`
    *   主要用于语义特征提取。
    *   分辨率固定为 518x518。

2.  **`x_sam` (SAM2 输入)**: `(B, 3, 352, 352)`
    *   主要用于细粒度细节提取。
    *   分辨率固定为 352x352。

**代码示例:**
```python
# B=Batch size (e.g., 1)
x_dino = torch.randn(1, 3, 518, 518).cuda()
x_sam  = torch.randn(1, 3, 352, 352).cuda()

# 注意参数顺序：先 dino (518)，后 sam (352)
out1, out # B=Batch size (e.g., 1)
x_dino = torch.randn(1, 3, 518, 518).cuda()
x_sam  = torch.randn(1, 3, 352, 352).cuda()

# 注意参数顺序：先 dino (518)，后 sam (352)
out1, out2, out3 = model(x_dino, x_sam)
```

**注意**: 千万不要搞反这两个输入的顺序或分辨率，否则会导致 `RuntimeError`，因为内部的特征融合层（如 `dino2sam_down`）是针对这种特定的尺寸比例硬编码的。**注意**: 千万不要搞反这两个输入的顺序或分辨率，否则会导致 `RuntimeError`，因为内部的特征融合层（如 `dino2sam_down`）是针对这种特定的尺寸比例硬编码的。

运行下面的脚本，查看输入数据在网络中的流动情况

```python
import torch
from sam2dino_seg.DGSUNet import DGSUNet
from torchinfo import summary


if __name__ == "__main__":
    with torch.no_grad():
        dgsUnet = DGSUNet().cuda()
        x_dino = torch.randn(1, 3, 518, 518).cuda()
        x_sam = torch.randn(1, 3, 352, 352).cuda()
        # print(model)
        summary(dgsUnet, input_data=(x_dino, x_sam))
        out, out1, out2 = dgsUnet(x_dino, x_sam)
        print(out.shape, out1.shape, out2.shape)
```

输出如下：

```text
==============================================================================================================
Layer (type:depth-idx)                                       Output Shape              Param #
==============================================================================================================
DGSUNet                                                      [1, 1, 352, 352]          --
├─sam2hiera: 1-1                                             [1, 144, 88, 88]          --
│    └─Hiera: 2-1                                            [1, 144, 88, 88]          16,272
│    │    └─PatchEmbed: 3-1                                  [1, 88, 88, 144]          (21,312)
│    │    └─Sequential: 3-2                                  --                        213,826,128
├─DinoV2FeatureExtractor: 1-2                                [1, 1024, 37, 37]         --
│    └─DinoVisionTransformer: 2-2                            --                        1,404,928
│    │    └─PatchEmbed: 3-3                                  [1, 1369, 1024]           (603,136)
│    │    └─ModuleList: 3-4                                  --                        (302,358,528)
│    │    └─LayerNorm: 3-5                                   [1, 1370, 1024]           (2,048)
├─interpolate_upsample: 1-3                                  [1, 1024, 11, 11]         --
├─DepthwiseSeparableConvWithWTConv2d: 1-4                    [1, 1152, 11, 11]         --
│    └─WTConv2d: 2-3                                         [1, 1024, 11, 11]         32,768
│    │    └─ModuleList: 3-6                                  --                        36,864
│    │    └─ModuleList: 3-7                                  --                        4,096
│    │    └─Conv2d: 3-8                                      [1, 1024, 11, 11]         10,240
│    │    └─_ScaleModule: 3-9                                [1, 1024, 11, 11]         1,024
│    └─Conv2d: 2-4                                           [1, 1152, 11, 11]         1,179,648
│    └─BatchNorm2d: 2-5                                      [1, 1152, 11, 11]         2,304
│    └─ReLU: 2-6                                             [1, 1152, 11, 11]         --
├─CGAFusion: 1-5                                             [1, 1152, 11, 11]         --
│    └─ChannelAttention: 2-7                                 [1, 1152, 1, 1]           --
│    │    └─AdaptiveAvgPool2d: 3-10                          [1, 1152, 1, 1]           --
│    │    └─Sequential: 3-11                                 [1, 1152, 1, 1]           333,072
│    └─SpatialAttention: 2-8                                 [1, 1, 11, 11]            --
│    │    └─Conv2d: 3-12                                     [1, 1, 11, 11]            99
│    └─PixelAttention: 2-9                                   [1, 1152, 11, 11]         --
│    │    └─Conv2d: 3-13                                     [1, 1152, 11, 11]         114,048
│    │    └─Sigmoid: 3-14                                    [1, 1152, 11, 11]         --
│    └─Sigmoid: 2-10                                         [1, 1152, 11, 11]         --
│    └─Conv2d: 2-11                                          [1, 1152, 11, 11]         1,328,256
├─RFB_modified: 1-6                                          [1, 64, 88, 88]           --
│    └─Sequential: 2-12                                      [1, 64, 88, 88]           --
│    │    └─BasicConv2d: 3-15                                [1, 64, 88, 88]           9,344
│    └─Sequential: 2-13                                      [1, 64, 88, 88]           --
│    │    └─BasicConv2d: 3-16                                [1, 64, 88, 88]           9,344
│    │    └─BasicConv2d: 3-17                                [1, 64, 88, 88]           12,416
│    │    └─BasicConv2d: 3-18                                [1, 64, 88, 88]           12,416
│    │    └─BasicConv2d: 3-19                                [1, 64, 88, 88]           36,992
│    └─Sequential: 2-14                                      [1, 64, 88, 88]           --
│    │    └─BasicConv2d: 3-20                                [1, 64, 88, 88]           9,344
│    │    └─BasicConv2d: 3-21                                [1, 64, 88, 88]           20,608
│    │    └─BasicConv2d: 3-22                                [1, 64, 88, 88]           20,608
│    │    └─BasicConv2d: 3-23                                [1, 64, 88, 88]           36,992
│    └─Sequential: 2-15                                      [1, 64, 88, 88]           --
│    │    └─BasicConv2d: 3-24                                [1, 64, 88, 88]           9,344
│    │    └─BasicConv2d: 3-25                                [1, 64, 88, 88]           28,800
│    │    └─BasicConv2d: 3-26                                [1, 64, 88, 88]           28,800
│    │    └─BasicConv2d: 3-27                                [1, 64, 88, 88]           36,992
│    └─BasicConv2d: 2-16                                     [1, 64, 88, 88]           --
│    │    └─Conv2d: 3-28                                     [1, 64, 88, 88]           147,456
│    │    └─BatchNorm2d: 3-29                                [1, 64, 88, 88]           128
│    └─BasicConv2d: 2-17                                     [1, 64, 88, 88]           --
│    │    └─Conv2d: 3-30                                     [1, 64, 88, 88]           9,216
│    │    └─BatchNorm2d: 3-31                                [1, 64, 88, 88]           128
│    └─ReLU: 2-18                                            [1, 64, 88, 88]           --
├─RFB_modified: 1-7                                          [1, 64, 44, 44]           --
│    └─Sequential: 2-19                                      [1, 64, 44, 44]           --
│    │    └─BasicConv2d: 3-32                                [1, 64, 44, 44]           18,560
│    └─Sequential: 2-20                                      [1, 64, 44, 44]           --
│    │    └─BasicConv2d: 3-33                                [1, 64, 44, 44]           18,560
│    │    └─BasicConv2d: 3-34                                [1, 64, 44, 44]           12,416
│    │    └─BasicConv2d: 3-35                                [1, 64, 44, 44]           12,416
│    │    └─BasicConv2d: 3-36                                [1, 64, 44, 44]           36,992
│    └─Sequential: 2-21                                      [1, 64, 44, 44]           --
│    │    └─BasicConv2d: 3-37                                [1, 64, 44, 44]           18,560
│    │    └─BasicConv2d: 3-38                                [1, 64, 44, 44]           20,608
│    │    └─BasicConv2d: 3-39                                [1, 64, 44, 44]           20,608
│    │    └─BasicConv2d: 3-40                                [1, 64, 44, 44]           36,992
│    └─Sequential: 2-22                                      [1, 64, 44, 44]           --
│    │    └─BasicConv2d: 3-41                                [1, 64, 44, 44]           18,560
│    │    └─BasicConv2d: 3-42                                [1, 64, 44, 44]           28,800
│    │    └─BasicConv2d: 3-43                                [1, 64, 44, 44]           28,800
│    │    └─BasicConv2d: 3-44                                [1, 64, 44, 44]           36,992
│    └─BasicConv2d: 2-23                                     [1, 64, 44, 44]           --
│    │    └─Conv2d: 3-45                                     [1, 64, 44, 44]           147,456
│    │    └─BatchNorm2d: 3-46                                [1, 64, 44, 44]           128
│    └─BasicConv2d: 2-24                                     [1, 64, 44, 44]           --
│    │    └─Conv2d: 3-47                                     [1, 64, 44, 44]           18,432
│    │    └─BatchNorm2d: 3-48                                [1, 64, 44, 44]           128
│    └─ReLU: 2-25                                            [1, 64, 44, 44]           --
├─RFB_modified: 1-8                                          [1, 64, 22, 22]           --
│    └─Sequential: 2-26                                      [1, 64, 22, 22]           --
│    │    └─BasicConv2d: 3-49                                [1, 64, 22, 22]           36,992
│    └─Sequential: 2-27                                      [1, 64, 22, 22]           --
│    │    └─BasicConv2d: 3-50                                [1, 64, 22, 22]           36,992
│    │    └─BasicConv2d: 3-51                                [1, 64, 22, 22]           12,416
│    │    └─BasicConv2d: 3-52                                [1, 64, 22, 22]           12,416
│    │    └─BasicConv2d: 3-53                                [1, 64, 22, 22]           36,992
│    └─Sequential: 2-28                                      [1, 64, 22, 22]           --
│    │    └─BasicConv2d: 3-54                                [1, 64, 22, 22]           36,992
│    │    └─BasicConv2d: 3-55                                [1, 64, 22, 22]           20,608
│    │    └─BasicConv2d: 3-56                                [1, 64, 22, 22]           20,608
│    │    └─BasicConv2d: 3-57                                [1, 64, 22, 22]           36,992
│    └─Sequential: 2-29                                      [1, 64, 22, 22]           --
│    │    └─BasicConv2d: 3-58                                [1, 64, 22, 22]           36,992
│    │    └─BasicConv2d: 3-59                                [1, 64, 22, 22]           28,800
│    │    └─BasicConv2d: 3-60                                [1, 64, 22, 22]           28,800
│    │    └─BasicConv2d: 3-61                                [1, 64, 22, 22]           36,992
│    └─BasicConv2d: 2-30                                     [1, 64, 22, 22]           --
│    │    └─Conv2d: 3-62                                     [1, 64, 22, 22]           147,456
│    │    └─BatchNorm2d: 3-63                                [1, 64, 22, 22]           128
│    └─BasicConv2d: 2-31                                     [1, 64, 22, 22]           --
│    │    └─Conv2d: 3-64                                     [1, 64, 22, 22]           36,864
│    │    └─BatchNorm2d: 3-65                                [1, 64, 22, 22]           128
│    └─ReLU: 2-32                                            [1, 64, 22, 22]           --
├─RFB_modified: 1-9                                          [1, 64, 11, 11]           --
│    └─Sequential: 2-33                                      [1, 64, 11, 11]           --
│    │    └─BasicConv2d: 3-66                                [1, 64, 11, 11]           73,856
│    └─Sequential: 2-34                                      [1, 64, 11, 11]           --
│    │    └─BasicConv2d: 3-67                                [1, 64, 11, 11]           73,856
│    │    └─BasicConv2d: 3-68                                [1, 64, 11, 11]           12,416
│    │    └─BasicConv2d: 3-69                                [1, 64, 11, 11]           12,416
│    │    └─BasicConv2d: 3-70                                [1, 64, 11, 11]           36,992
│    └─Sequential: 2-35                                      [1, 64, 11, 11]           --
│    │    └─BasicConv2d: 3-71                                [1, 64, 11, 11]           73,856
│    │    └─BasicConv2d: 3-72                                [1, 64, 11, 11]           20,608
│    │    └─BasicConv2d: 3-73                                [1, 64, 11, 11]           20,608
│    │    └─BasicConv2d: 3-74                                [1, 64, 11, 11]           36,992
│    └─Sequential: 2-36                                      [1, 64, 11, 11]           --
│    │    └─BasicConv2d: 3-75                                [1, 64, 11, 11]           73,856
│    │    └─BasicConv2d: 3-76                                [1, 64, 11, 11]           28,800
│    │    └─BasicConv2d: 3-77                                [1, 64, 11, 11]           28,800
│    │    └─BasicConv2d: 3-78                                [1, 64, 11, 11]           36,992
│    └─BasicConv2d: 2-37                                     [1, 64, 11, 11]           --
│    │    └─Conv2d: 3-79                                     [1, 64, 11, 11]           147,456
│    │    └─BatchNorm2d: 3-80                                [1, 64, 11, 11]           128
│    └─BasicConv2d: 2-38                                     [1, 64, 11, 11]           --
│    │    └─Conv2d: 3-81                                     [1, 64, 11, 11]           73,728
│    │    └─BatchNorm2d: 3-82                                [1, 64, 11, 11]           128
│    └─ReLU: 2-39                                            [1, 64, 11, 11]           --
├─SFF: 1-10                                                  [1, 64, 22, 22]           --
│    └─Sequential: 2-40                                      [1, 64, 22, 22]           --
│    │    └─Conv2d: 3-83                                     [1, 64, 22, 22]           4,096
│    │    └─BatchNorm2d: 3-84                                [1, 64, 22, 22]           128
│    │    └─ReLU: 3-85                                       [1, 64, 22, 22]           --
│    └─Sequential: 2-41                                      [1, 64, 22, 22]           --
│    │    └─Conv2d: 3-86                                     [1, 64, 22, 22]           4,096
│    │    └─BatchNorm2d: 3-87                                [1, 64, 22, 22]           128
│    │    └─ReLU: 3-88                                       [1, 64, 22, 22]           --
│    └─SelfAttentionBlock: 2-42                              [1, 64, 22, 22]           --
│    │    └─Sequential: 3-89                                 [1, 32, 22, 22]           3,200
│    │    └─Sequential: 3-90                                 [1, 32, 22, 22]           3,200
│    │    └─Sequential: 3-91                                 [1, 32, 22, 22]           2,112
│    │    └─Sequential: 3-92                                 [1, 64, 22, 22]           2,176
│    └─Sequential: 2-43                                      [1, 64, 22, 22]           --
│    │    └─Conv2d: 3-93                                     [1, 64, 22, 22]           73,728
│    │    └─BatchNorm2d: 3-94                                [1, 64, 22, 22]           128
│    │    └─ReLU: 3-95                                       [1, 64, 22, 22]           --
├─Conv2d: 1-11                                               [1, 1, 22, 22]            65
├─SFF: 1-12                                                  [1, 64, 44, 44]           --
│    └─Sequential: 2-44                                      [1, 64, 44, 44]           --
│    │    └─Conv2d: 3-96                                     [1, 64, 44, 44]           4,096
│    │    └─BatchNorm2d: 3-97                                [1, 64, 44, 44]           128
│    │    └─ReLU: 3-98                                       [1, 64, 44, 44]           --
│    └─Sequential: 2-45                                      [1, 64, 44, 44]           --
│    │    └─Conv2d: 3-99                                     [1, 64, 44, 44]           4,096
│    │    └─BatchNorm2d: 3-100                               [1, 64, 44, 44]           128
│    │    └─ReLU: 3-101                                      [1, 64, 44, 44]           --
│    └─SelfAttentionBlock: 2-46                              [1, 64, 44, 44]           --
│    │    └─Sequential: 3-102                                [1, 32, 44, 44]           3,200
│    │    └─Sequential: 3-103                                [1, 32, 44, 44]           3,200
│    │    └─Sequential: 3-104                                [1, 32, 44, 44]           2,112
│    │    └─Sequential: 3-105                                [1, 64, 44, 44]           2,176
│    └─Sequential: 2-47                                      [1, 64, 44, 44]           --
│    │    └─Conv2d: 3-106                                    [1, 64, 44, 44]           73,728
│    │    └─BatchNorm2d: 3-107                               [1, 64, 44, 44]           128
│    │    └─ReLU: 3-108                                      [1, 64, 44, 44]           --
├─Conv2d: 1-13                                               [1, 1, 44, 44]            65
├─SFF: 1-14                                                  [1, 64, 88, 88]           --
│    └─Sequential: 2-48                                      [1, 64, 88, 88]           --
│    │    └─Conv2d: 3-109                                    [1, 64, 88, 88]           4,096
│    │    └─BatchNorm2d: 3-110                               [1, 64, 88, 88]           128
│    │    └─ReLU: 3-111                                      [1, 64, 88, 88]           --
│    └─Sequential: 2-49                                      [1, 64, 88, 88]           --
│    │    └─Conv2d: 3-112                                    [1, 64, 88, 88]           4,096
│    │    └─BatchNorm2d: 3-113                               [1, 64, 88, 88]           128
│    │    └─ReLU: 3-114                                      [1, 64, 88, 88]           --
│    └─SelfAttentionBlock: 2-50                              [1, 64, 88, 88]           --
│    │    └─Sequential: 3-115                                [1, 32, 88, 88]           3,200
│    │    └─Sequential: 3-116                                [1, 32, 88, 88]           3,200
│    │    └─Sequential: 3-117                                [1, 32, 88, 88]           2,112
│    │    └─Sequential: 3-118                                [1, 64, 88, 88]           2,176
│    └─Sequential: 2-51                                      [1, 64, 88, 88]           --
│    │    └─Conv2d: 3-119                                    [1, 64, 88, 88]           73,728
│    │    └─BatchNorm2d: 3-120                               [1, 64, 88, 88]           128
│    │    └─ReLU: 3-121                                      [1, 64, 88, 88]           --
├─Conv2d: 1-15                                               [1, 1, 88, 88]            65
==============================================================================================================
Total params: 523,776,534
Trainable params: 7,225,830
Non-trainable params: 516,550,704
Total mult-adds (Units.GIGABYTES): 7.88
==============================================================================================================
Input size (MB): 4.71
Forward/backward pass size (MB): 5799.91
Params size (MB): 2089.29
Estimated Total Size (MB): 7893.91
==============================================================================================================
torch.Size([1, 1, 352, 352]) torch.Size([1, 1, 352, 352]) torch.Size([1, 1, 352, 352])
```

可以看到，在特征提取部分经过一个`CGA`模块融合两个分支的特征后，最终的形状为`[1, 1152, 11, 11]`，而我们的目标是`[1, 256, 16, 16]`，因此需要进行维度处理。

> 为什么这里使用CGA融合模块？
> 
> 在 `DGSUNet` 中使用 **CGAFusion** (Cross-Guided Attention Fusion) 模块，是为了**自适应地融合**来自 SAM2（侧重细节和边界）与 DiNOv2（侧重高层语义）的特征。
> 
> 简单来说，简单的相加或拼接无法区分两个骨干网络的优劣，而 `CGAFusion` 通过注意力机制让网络自己“决定”在每个像素点上更应该相信 SAM2 还是 DiNOv2。
> 
> 具体原因和机制如下：
> 1.  **特征优势互补**：
>     *   **SAM2 (x)**：在分割任务中，擅长捕捉细粒度的边界和形状信息。
>     *   **DiNOv2 (y)**：在大规模数据上预训练，拥有极强的语义理解能力。
>     *   **CGAFusion** 的作用就是将这两者的长处结合起来。
> 
> 2.  **三维注意力机制（Channel, Spatial, Pixel）**：
>     代码 (CGAFusion.py) 显示该模块实际上串联了三种注意力：
>     *   **Channel Attention (`ca`)**：识别哪些特征通道更重要（主要关注“是什么”）。
>     *   **Spatial Attention (`sa`)**：识别图像中哪些区域更重要（主要关注“在哪里”）。
>     *   **Pixel Attention (`pa`)**：结合上述信息生成像素级的权重图。
> 
> 3.  **门控融合机制 (Gated Fusion)**：
>     模块最核心的公式如下：
>     ```python
>     # pattn2 是经过 Sigmoid 生成的 0~1 之间的权重图
>     result = initial + pattn2 * x + (1 - pattn2) * y
>     ```
>     *   这意味着网络为每个位置生成了一个权重 `pattn2`。
>     *   如果 `pattn2` 接近 1，主要保留 **SAM2 (x)** 的特征。
>     *   如果 `pattn2` 接近 0，主要保留 **DiNOv2 (y)** 的特征。
>     *   这种软选择（Soft Selection）机制比硬性的 `Add` 或 `Concat` 更灵活，能有效减少特征冲突。
> 
> **总结：**
> 使用 CGAFusion 是为了**智能地**解决双骨干网络（Dual Backbone）特征融合的问题，确保模型既有 SAM2 的精准边缘，又有 DiNOv2 的丰富语义。

### 开始修改

好像CGA只能处理双骨干网络，否则就要修改原始`CGAFusion`的代码了。要不换个思路，抛弃三分支，直接使用DSU-Net的骨干网络。

好吧，参考了一下ai的意见，它建议我将`DSGC-Net`骨干网络替换为`DSU-Net`的骨干网络。如果这样的话就要修改数据预处理的代码，

欸，`DSU-Net`中有个`RFB`模块很有意思，似乎能调整通道数。

> **RFB (Receptive Field Block)** 模块的主要作用是**增强特征提取能力**，通过模拟人类视觉的感受野机制，使网络能够同时捕获多尺度的上下文信息。
> 
> 在这个 `DGSUNet` 网络中，RFB 模块具体扮演了以下两个关键角色：
> 
> 1. 扩大感受野 (Expand Receptive Field)
> 代码显示 `RFB_modified` 包含四个并行的分支 (`branch0` ~ `branch3`)，它们使用了不同的膨胀卷积（Dilated Convolution）率：
> *   **Branch 0**: 1x1 卷积（基础特征）。
> *   **Branch 1**: 膨胀率 `dilation=3`。
> *   **Branch 2**: 膨胀率 `dilation=5`。
> *   **Branch 3**: 膨胀率 `dilation=7`。
> 
> 通过将这些不同“视野”范围提取到的特征拼接在一起，网络既能看到局部细节（小感受野），也能感知更大的上下文环境（大感受野），这对于分割不同大小的目标至关重要。
> 
> 2. 统一通道数 (Channel Adaptation)
> 在 DGSUNet.py 中，你可以看到 RFB 连接在骨干网络（Backbone）和解码器（Decoder）之间：
> ```python
> # 骨干网络输出的不同尺度的通道数
> # x1: 144, x2: 288, x3: 576, x4: 1152
> self.rfb1 = RFB.RFB_modified(144, 64)   # 输出 64
> self.rfb2 = RFB.RFB_modified(288, 64)   # 输出 64
> # ...
> ```
> 它将骨干网络输出的**不一致的通道数**（如 144, 288, 576, 1152）全部降维并统映射到 **64** 个通道。这样做是为了方便后续解码器 (`SFF` 模块) 进行逐级上采样和融合。
> 
> 总结
> 简单来说，RFB 在这里起到了 **“承上启下”** 的作用：
> *   **承上**：接收骨干网络的特征，利用多分支空洞卷积使其包含更丰富的尺度信息。
> *   **启下**：把特征压缩成统一的厚度（Channel=64），喂给解码器生成最终的分割图。

通过这个模块不就可以让`channels=1152`变成`channels=256`吗？

所以现在唯一的难点就是怎样将**output_size = [11, 11] -> output_size = [16, 16]**

这个解决起来也简单，经过一个`nn.Upsample(size=(16, 16))`就能解决关键是，修改数据加载模块。

```python
import os
import random
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
from PIL import Image
import scipy.io as io

# 【建议1】定义全局 CROP_SIZE，方便统一修改
# CROP_SIZE 必须能被 32 整除 (Patch Size 14 * 16 = 224 是最理想的)
CROP_SIZE = 224 

class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "train.txt"
        self.eval_list = "test.txt"
        self.gt_density = "gt_density_maps" 
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
            self.gt_dmap_root = os.path.join(self.root_path, self.gt_density,'train')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        imgname = os.path.basename(img_path)
        if self.train:
            gt_dmap = np.load(os.path.join(self.gt_dmap_root,imgname.replace('.jpg','.npy')))
            gt_dmap = torch.from_numpy(gt_dmap)
            gt_dmap1 = gt_dmap.unsqueeze(0)
        img, point = load_data((img_path, gt_path), self.train)
        if self.train:
            augmentation = transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
                ], p=0.5),
                transforms.RandomGrayscale(p=0.5)
            ])
            img = augmentation(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)

            # 【修改2】调整缩放逻辑中的最小尺寸判断
            # 必须保证缩放后的图片至少比 CROP_SIZE 大，否则 random_crop 会报错
            if scale * min_size > CROP_SIZE:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                gt_dmap1 = torch.nn.functional.upsample_bilinear(gt_dmap1.unsqueeze(0), scale_factor=scale).squeeze(0)
                gt_dmap1 = gt_dmap1 / torch.sum(gt_dmap1) * torch.sum(gt_dmap)
                point *= scale
        if self.train:
            img_with_density = torch.cat((img, gt_dmap1), dim=0)
        if self.train and self.patch:
            img_with_density, point = random_crop(img_with_density, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])

        # 【修改3】Flip 和 Crop 后的点坐标归一化修正
        # 原代码硬编码了 128 (`point[i][:, 0] = 128 - ...`)
        # 现在必须改为 CROP_SIZE (224)
        if random.random() > 0.5 and self.train and self.flip:
            img_with_density = torch.Tensor(img_with_density[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = CROP_SIZE - point[i][:, 0]
        if self.train:
            img = img_with_density[:, :-1, :, :]
            density = img_with_density[:, -1:, :, :]
            density = torch.Tensor(density)
        if not self.train:
            point = [point]
        img = torch.Tensor(img)
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()
        if self.train:
            # 【修改4】 生成 Density Map 标签的下采样倍率修正
            # 原本: 128 -> 16 (倍率8)
            # 现在: 224 -> 16 (倍率14) -> 这会导致 label 和 output 不匹配！
            # 
            # 关键决策：你的 DSGCNet 输出是强制对齐到 16x16 的。
            # 所以这里 GT 的生成也必须是 16x16。
            # 原代码逻辑：kernel size = 128 / 16 = 8。
            # 新代码逻辑：kernel size = 224 / 16 = 14。
            
            density_images = torch.zeros((density.shape[0], 1, 16, 16), dtype=density.dtype)

            # 计算下采样快的大小（Stride）：224 // 16 = 14
            stride = CROP_SIZE // 16
            for i in range(density.shape[0]):
                density_img = density[i, 0, :, :]
                # Sum pooling: 将 224x224 块加和成 16x16
                # 注意 reshape 参数：(H_out, Stride, W_out, Stride)
                resized_img = density_img.reshape([16, stride, 16, stride]).sum(axis=(1, 3))
                density_images[i, 0, :, :] = resized_img
        if self.train:
            return img, target, density_images
        else:
            return img, target

def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = []
    
    # 支持.mat和.txt两种格式
    if gt_path.endswith('.mat'):
        # 处理.mat格式（ShanghaiTech原始格式）
        try:
            mat = io.loadmat(gt_path)
            # ShanghaiTech Part A格式: image_info[0,0][0,0][0]
            if 'image_info' in mat:
                points_data = mat["image_info"][0, 0][0, 0][0]
                for i in range(len(points_data)):
                    points.append([float(points_data[i][0]), float(points_data[i][1])])
            # 其他可能的.mat格式
            elif 'annPoints' in mat:
                points_data = mat['annPoints']
                for i in range(len(points_data)):
                    points.append([float(points_data[i][0]), float(points_data[i][1])])
            # 如果points直接是数组
            elif 'points' in mat:
                points_data = mat['points']
                if len(points_data.shape) == 2:
                    for i in range(len(points_data)):
                        points.append([float(points_data[i][0]), float(points_data[i][1])])
        except Exception as e:
            print(f"Warning: Failed to load .mat file {gt_path}: {e}")
            print(f"Trying to read as txt format...")
            # 如果读取失败，尝试作为txt处理
            try:
                with open(gt_path) as f_label:
                    for line in f_label:
                        line = line.strip()
                        if len(line) < 2:
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            x = float(parts[0])
                            y = float(parts[1])
                            points.append([x, y])
            except:
                print(f"Error: Cannot read annotation file {gt_path}")
    else:
        # 处理.txt格式
        try:
            with open(gt_path) as f_label:
                for line in f_label:
                    line = line.strip()
                    if len(line) < 2:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        x = float(parts[0])
                        y = float(parts[1])
                        points.append([x, y])
        except Exception as e:
            print(f"Error: Failed to load .txt file {gt_path}: {e}")

    return img, np.array(points) if len(points) > 0 else np.array([]).reshape(0, 2)

def random_crop(img, den, num_patch=4):
    # 【修改5】random_crop 的尺寸也必须同步修改
    half_h = CROP_SIZE
    half_w = CROP_SIZE
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []

    # 防止极少数情况下图片小于 crop size 导致报错
    # 理论上前面的 scale 逻辑已经保护了，但这里加个安全兜底
    if img.shape[1] < half_h or img.shape[2] < half_w:
       # 如果真的比 224 还要小，就只能 Padding 或者 Resize 硬拉
       # 简单策略：Pad 0 补齐
       pad_h = max(0, half_h - img.shape[1])
       pad_w = max(0, half_w - img.shape[2])
       img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))

    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h

        result_den.append(record_den)

    return result_img, result_den
```
> 为什么要这样修改?
> 
> 这是一个非常好的问题！理解“为什么要改”比“知道怎么改”更重要，因为这涉及到深度学习中数据、模型和损失函数之间的<u>对齐（Alignment）</u>原则。
> 
> 之所以要这改这 5 处代码，是因为你把模型的输入尺寸（Input Size）从 **128 改为了 224**。这一变动引发了**连锁反应**，必须在数据处理的每一个环节都进行同步调整，否则会导致模型“看”到的是错的图片，或者“学”到的是错的标签。
> 
> 我把这 5 个修改点背后的**数学和逻辑原因**拆解给你看：
> 
> ---
> 
> 1. 修改 `random_crop` 函数中的尺寸 (128 -> 224)
> *   **原因**：为了喂给 Transformer (ViT) 骨干网络。
> *   **解释**：DSU-Net 用的 Dinov2 和 SAM2 是基于 Vision Transformer 的。它们把图片切成一个个 $14 \times 14$ 的小方块（Patch）。
>     *   如果你用原来的 **128**：$128 \div 14 \approx 9.14$。切不整齐，边缘必须补零（Padding），这会引入无意义的黑边，干扰特征提取。
>     *   改为 **224**：$224 \div 14 = 16$。刚好切成 $16 \times 16$ 个 Patch，完美适配。这也是 ImageNet 预训练的标准尺寸。
> 
> 2. 修改 Scale 缩放逻辑 (`> CROP_SIZE`)
> *   **原因**：防止程序崩溃（Index Error）。
> *   **解释**：`random_crop` 是从大图里随机抠一个小图。
>     *   如果你把图缩放到只剩 150 像素宽，然后让程序去抠一个 224 像素的块，程序就会报错（因为图不够大，抠不出来）。
>     *   因此，必须保证缩放后的图**至少**比我们要抠的尺寸（224）大。
> 
> 3. 修改 Flip 翻转坐标 (`CROP_SIZE - x`)
> *   **原因**：保证坐标变换的几何正确性。
> *   **解释**：这就像照镜子。
>     *   如果你在一张宽 128 的纸上画个点，坐标是 $x=10$。水平翻转后，新坐标是 $128 - 10 = 118$。
>     *   现在纸变宽了（224），同样的翻转，新坐标应该是 $224 - 10 = 214$。
>     *   如果你还用旧公式 $128 - 10 = 118$，这个点就跑到图片中间左边去了，而图片内容其实已经翻转到了最右边。**点和人就对不上了**，模型会学废。
> 
> 4. 修改 GT Density Map 生成逻辑 (Sum Pooling Stride)
> *   **这是最关键的修改！**
> *   **原因**：保证 Ground Truth (标签) 和 模型预测输出 (Prediction) 的**总能量守恒**。
> *   **解释**：
>     *   你的模型输出是固定的 $16 \times 16$ 网格。
>     *   **旧逻辑**：输入 $128 \times 128$。要把 128 压缩成 16，每个网格就要“管” $128/16 = \mathbf{8}$ 个像素宽的区域。所以我们对 GT 进行 $8 \times 8$ 的求和池化。
>     *   **新逻辑**：输入 $224 \times 224$。要把 224 压缩成 16，每个网格就要“管” $224/16 = \mathbf{14}$ 个像素宽的区域。
>     *   如果你输入 224 的图，却还按 8倍下采样生成 GT，你的 GT 尺寸会变成 $224/8 = 28 \times 28$。
>     *   **后果**：模型输出 $16 \times 16$，标签却是 $28 \times 28$。Loss 函数（如 MSE Loss）一计算就会直接报错：`RuntimeError: The size of tensor a (16) must match the size of tensor b (28)`。
> 
> 总结
> 这一系列修改的核心逻辑是：
> **牵一发而动全身**。
> 一旦输入尺寸变了（为了适配新骨干），虽然输出尺寸想保持不变（为了适配旧后端），但中间所有的**坐标映射**（翻转）、**空间映射**（Crop）和**标签映射**（GT生成）的比例系数全都要跟着变。
> 
> 不做这些修改，代码要么跑不起来（报错），要么跑起来训练效果极差（坐标错位）。

输入搞定了，接下来就是中间的对接了，已知DSU-Net的特征提取后的分辨率为`11x11`，而目标分辨率为`16x16`，故而需要一个适应层来调整分辨率。

在参考了ai的建议后，我决定使用`RFB+Bilinear`的方案，通过RFB模块调整通道数，利用双线性插值调整分辨率。

参考代码如下：
```python
import torch.nn as nn
from models.RFB import RFB_modified

class Adapter_RFB(nn.Module):
    def __init__(self, in_channels=1152, out_channels=256, target_size=(16, 16)):
        super().__init__()
        # 1. 先用 RFB 降维并提取多尺度特征 (1152 -> 256)
        # RFB 内部维持 spatial size 不变 (11x11 -> 11x11)
        self.rfb = RFB_modified(in_channels, out_channels)
        
        # 2. 空间对齐 (11x11 -> 16x16)
        self.upsample = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        x = self.rfb(x)
        x = self.upsample(x)
        return x
```

## 跑代码

## 写论文

## 参考论文

1. Xu, Yimin, Fan Yang和Bin Xu. 《DSU-Net:An Improved U-Net Model Based on DINOv2 and SAM2 with Multi-Scale Cross-Model Feature Enhancement》. arXiv:2503.21187. 预印本, arXiv, 2025年3月31日. https://doi.org/10.48550/arXiv.2503.21187.
2. Wu, Yihong, Jinqiao Wei, Xionghui Zhao, 等. 《DSGC-Net: A Dual-Stream Graph Convolutional Network for Crowd Counting via Feature Correlation Mining》. arXiv:2509.02261. 预印本, arXiv, 2025年9月2日. https://doi.org/10.48550/arXiv.2509.02261.

