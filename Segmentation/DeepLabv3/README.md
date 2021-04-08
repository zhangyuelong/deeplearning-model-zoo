### 注意事项

当网络的backbone选择mobilenetv2时, 由于mobilenetv2中的倒残差结构(InvertedResidual)采用了F.pad操作, 该操作可以转成onnx, 但onnx转trt会出错, 报错如下:

```
While parsing node number 24 [Pad]:
ERROR: /workspace/TensorRT/parsers/onnx/builtin_op_importers.cpp:2251 In function importPad:
[8] Assertion failed: inputs.at(1).is_weights()
[04/07/2021-19:29:32] [E] Failed to parse onnx file
[04/07/2021-19:29:32] [E] Parsing model failed
[04/07/2021-19:29:32] [E] Engine creation failed
[04/07/2021-19:29:32] [E] Engine set up failed
```

解决方案是: **对生成的onnx采用onnx-simplifier进行优化, 生成的新的优化完之后的onnx就可以转trt了.**

### 数据集和预训练权重

- dataset: 链接: `https://pan.baidu.com/s/1PQ5dmR9vfWfRb_w_DQhDUQ`  密码: af8f
- mobilenetv2: `https://download.pytorch.org/models/mobilenet_v2-b0353104.pth`
- resnet50: `https://download.pytorch.org/models/resnet50-19c8e357.pth`
- resnet101: `https://download.pytorch.org/models/resnet101-5d3b4d8f.pth`

### 模型训练

训练前, 记得将`model = deeplabv3plus_mobilenet(num_classes=class_num, output_stride=8, pretrained_backbone=pre_trained)`这句改成你想要的网络和backbone.

`torch.save(model.state_dict(), weights + "DeepLabV3plus_mobilenet_weights" + ".pth")`改成对应的名字.

预测代码与其保持一致, 不再赘述.

```bash_script
python3 train.py
```

模型训练完毕之后, 在./weights/目录下生成模型的权重`DeepLabV3plus_mobilenet_weights.pth`.

### 模型推理

```bash_script
python3 predict.py
```

模型推理的过程中也顺便转成了onnx模型fcnet.onnx, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim deeplabv3+_mobilenet.onnx deeplabv3+_mobilenet_opt.onnx --input-shape 1,3,352,480 --dynamic-input-shape --enable-fuse-bn
```

### ONNX转换TensorRT脚本

```bash_script
trtexec --onnx=deeplabv3+_mobilenet_opt.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x352x480 \
        --maxShapes=image:8x3x352x480 \
        --minShapes=image:1x3x352x480 \
        --shapes=image:4x3x352x480 \
        --saveEngine=deeplabv3+_mobilenet_opt.engine
```

### trt模型性能测试

测试环境:

- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```bash_script
trtexec --loadEngine=deeplabv3+_mobilenet_opt.engine --shapes=image:?x3x352x480
```

**deeplabv3+ mobilenet**

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 38.6580 | 44.2886 | 40.5436 | 39.5461 |
| 2 | 75.3174 | 88.5777 | 78.7434 | 79.3057 |
| 3 | 111.152 | 122.507 | 112.504 | 111.520 |
| 4 | 146.377 | 159.806 | 148.695 | 147.575 |
| 5 | 184.721 | 195.042 | 187.262 | 186.222 |
| 6 | 222.069 | 234.063 | 224.771 | 223.849 |
| 7 | 258.401 | 273.927 | 262.354 | 261.326 |
| 8 | 296.910 | 308.036 | 299.635 | 298.342 |

**deeplabv3+ resnet50**

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 129.220 | 143.695 | 132.090 | 131.052 |
| 2 | 263.174 | 268.114 | 264.963 | 264.963 |
| 3 | 376.586 | 387.113 | 380.224 | 380.486 |
| 4 | 506.564 | 510.324 | 508.450 | 508.420 |
| 5 | 622.829 | 639.910 | 628.496 | 627.132 |
| 6 | 745.717 | 762.307 | 749.720 | 748.935 |
| 7 | 790.234 | 794.756 | 792.709 | 792.864 |
| 8 | 898.688 | 925.750 | 904.848 | 902.534 |

**deeplabv3+ resnet101**

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 184.179 | 194.847 | 187.279 | 186.800 |
| 2 | 354.165 | 365.621 | 357.952 | 356.564 |
| 3 | 525.736 | 539.051 | 529.008 | 528.110 |
| 4 | 697.054 | 708.871 | 702.906 | 703.478 |
| 5 | 875.642 | 881.435 | 878.135 | 878.263 |
| 6 | 1038.10 | 1045.90 | 1041.66 | 1041.50 |
| 7 | 1135.31 | 1138.70 | 1137.48 | 1137.78 |
| 8 | 1290.48 | 1306.31 | 1295.60 | 1293.79 |
