### 数据集和预训练权重

- dataset: 链接: https://pan.baidu.com/s/1PQ5dmR9vfWfRb_w_DQhDUQ  密码: af8f
- mobilenetv2: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth

### 模型训练

```bash_script
python3 train.py
```

模型训练完毕之后, 在./weights/目录下生成模型的权重`PSPNet_weights.pth`.

### 模型推理

```bash_script
python3 predict.py
```

模型推理的过程中也顺便转成了onnx模型pspnet.onnx, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim pspnet.onnx pspnet_opt.onnx --input-shape 1,3,473,473 --dynamic-input-shape --enable-fuse-bn
```

### ONNX转换TensorRT脚本

```bash_script
trtexec --onnx=pspnet.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x473x473 \
        --maxShapes=image:8x3x473x473 \
        --minShapes=image:1x3x473x473 \
        --shapes=image:4x3x473x473 \
        --saveEngine=pspnet.engine
```

### trt模型性能测试

测试环境:

- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```bash_script
trtexec --loadEngine=pspnet.engine --shapes=image:?x3x473x473
```

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 7.35913 | 8.42233 | 7.51580 | 7.41479 |
| 2 | 13.8574 | 15.4993 | 14.0551 | 13.9346 |
| 3 | 20.2526 | 22.7707 | 20.5888 | 20.3817 |
| 4 | 26.8716 | 28.6900 | 27.2464 | 27.0315 |
| 5 | 33.3037 | 35.4966 | 33.7325 | 33.5261 |
| 6 | 39.8170 | 43.8732 | 40.3400 | 39.9674 |
| 7 | 46.4858 | 50.1269 | 47.0649 | 46.7119 |
| 8 | 52.7318 | 56.6480 | 53.2767 | 52.9101 |
