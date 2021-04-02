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
| 1 | 21.8986 | 23.9066 | 22.3813 | 22.1873 |
| 2 | 43.7179 | 48.3323 | 44.5957 | 44.2279 |
| 3 | 65.1090 | 69.9022 | 66.2255 | 65.9062 |
| 4 | 87.3655 | 92.8395 | 88.2939 | 87.9507 |
| 5 | 109.834 | 116.431 | 111.787 | 110.748 |
| 6 | 131.556 | 139.965 | 133.770 | 132.418 |
| 7 | 153.440 | 161.788 | 155.994 | 154.767 |
| 8 | 176.166 | 185.773 | 178.224 | 177.190 |
