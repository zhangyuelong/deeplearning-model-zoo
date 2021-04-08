### 注意事项

当网络选择DeepLabv3, backbone选择mobilenetv2时, 由于mobilenetv2中的倒残差结构(InvertedResidual)采用了F.pad操作, 该操作可以转成onnx, 但onnx转trt会出错, 报错如下:

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
| 1 | 36.7666 | 40.0117 | 37.9431 | 37.4152 |
| 2 | 73.6550 | 78.5356 | 74.6074 | 74.0884 |
| 3 | 110.585 | 117.506 | 112.872 | 111.710 |
| 4 | 147.337 | 156.523 | 150.515 | 149.135 |
| 5 | 185.056 | 196.943 | 188.970 | 187.092 |
| 6 | 222.645 | 234.137 | 224.942 | 223.827 |
| 7 | 259.439 | 273.521 | 262.877 | 261.810 |
| 8 | 297.152 | 310.719 | 299.689 | 298.531 |

**deeplabv3+ resnet50**

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 36.7666 | 40.0117 | 37.9431 | 37.4152 |
| 2 | 73.6550 | 78.5356 | 74.6074 | 74.0884 |
| 3 | 110.585 | 117.506 | 112.872 | 111.710 |
| 4 | 147.337 | 156.523 | 150.515 | 149.135 |
| 5 | 185.056 | 196.943 | 188.970 | 187.092 |
| 6 | 222.645 | 234.137 | 224.942 | 223.827 |
| 7 | 259.439 | 273.521 | 262.877 | 261.810 |
| 8 | 297.152 | 310.719 | 299.689 | 298.531 |

**deeplabv3+ resnet101**

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 36.7666 | 40.0117 | 37.9431 | 37.4152 |
| 2 | 73.6550 | 78.5356 | 74.6074 | 74.0884 |
| 3 | 110.585 | 117.506 | 112.872 | 111.710 |
| 4 | 147.337 | 156.523 | 150.515 | 149.135 |
| 5 | 185.056 | 196.943 | 188.970 | 187.092 |
| 6 | 222.645 | 234.137 | 224.942 | 223.827 |
| 7 | 259.439 | 273.521 | 262.877 | 261.810 |
| 8 | 297.152 | 310.719 | 299.689 | 298.531 |
