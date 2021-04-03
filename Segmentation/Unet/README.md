### 数据集和预训练权重

- dataset: 链接: https://pan.baidu.com/s/1PQ5dmR9vfWfRb_w_DQhDUQ  密码: af8f
- vgg16: https://download.pytorch.org/models/vgg16-397923af.pth

### 模型训练

```bash_script
python3 train.py
```

模型训练完毕之后, 在./weights/目录下生成模型的权重`UNet_weights.pth`.

### 模型推理

```bash_script
python3 predict.py
```

模型推理的过程中也顺便转成了onnx模型unet.onnx, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim unet.onnx unet_opt.onnx --input-shape 1,3,224,224 --dynamic-input-shape --enable-fuse-bn
```

### ONNX转换TensorRT脚本

```bash_script
trtexec --onnx=unet.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x224x224 \
        --maxShapes=image:8x3x224x224 \
        --minShapes=image:1x3x224x224 \
        --shapes=image:4x3x224x224 \
        --saveEngine=unet.engine
```

### trt模型性能测试

测试环境:

- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```bash_script
trtexec --loadEngine=unet.engine --shapes=image:?x3x224x224
```

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 30.3757 | 32.6938 | 30.7708 | 30.6770 |
| 2 | 59.2369 | 62.3008 | 60.0166 | 59.8248 |
| 3 | 88.8722 | 94.6840 | 91.1178 | 91.0913 |
| 4 | 117.408 | 122.917 | 120.373 | 120.580 |
| 5 | 149.578 | 156.414 | 153.962 | 153.923 |
| 6 | 176.703 | 186.560 | 181.562 | 181.436 |
| 7 | 213.342 | 220.542 | 216.478 | 216.215 |
| 8 | 240.008 | 247.321 | 243.794 | 244.878 |
