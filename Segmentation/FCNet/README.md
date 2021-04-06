### 数据集和预训练权重

- dataset: 链接: https://pan.baidu.com/s/1PQ5dmR9vfWfRb_w_DQhDUQ  密码: af8f
- `vgg16_bn`: https://download.pytorch.org/models/vgg16_bn-6c64b313.pth

### 模型训练

```bash_script
python3 train.py
```

模型训练完毕之后, 在./weights/目录下生成模型的权重`FCNet_weights.pth`.

### 模型推理

```bash_script
python3 predict.py
```

模型推理的过程中也顺便转成了onnx模型fcnet.onnx, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim fcnet.onnx fcnet_opt.onnx --input-shape 1,3,352,480 --dynamic-input-shape --enable-fuse-bn
```

### ONNX转换TensorRT脚本

```bash_script
trtexec --onnx=fcnet.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x352x480 \
        --maxShapes=image:8x3x352x480 \
        --minShapes=image:1x3x352x480 \
        --shapes=image:4x3x352x480 \
        --saveEngine=fcnet.engine
```

### trt模型性能测试

测试环境:

- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```bash_script
trtexec --loadEngine=fcnet.engine --shapes=image:?x3x352x480
```

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
