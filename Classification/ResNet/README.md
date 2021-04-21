### 模型训练

```bash_script
python3 train.py
```

模型训练完毕之后, 在./weights/目录下生成模型的权重`resnet18.pth`.

### 模型推理

```bash_script
python3 predict.py
```

模型推理的过程中也顺便转成了onnx模型`resnet18.onnx`, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim resnet18.onnx resnet18_opt.onnx --input-shape 1,3,224,224 --dynamic-input-shape --enable-fuse-bn
```

### ONNX转换TensorRT脚本

```shell script
trtexec --onnx=resnet18.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x224x224 \
        --maxShapes=image:8x3x224x224 \
        --minShapes=image:1x3x224x224 \
        --shapes=image:4x3x224x224 \
        --saveEngine=resnet18.engine
```

### trt模型性能测试

测试环境:
- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```shell script
trtexec --loadEngine=resnet18.engine --shapes=image:?x3x224x224
```

**ResNet34**

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 4.51672 | 5.11865 | 4.67817 | 4.61871 |
| 2 | 6.73474 | 7.65327 | 7.00408 | 6.90625 |
| 3 | 10.3239 | 11.3786 | 10.8011 | 10.6954 |
| 4 | 12.9340 | 14.1838 | 13.4075 | 13.3064 |
| 5 | 15.1935 | 17.3551 | 15.7719 | 15.5737 |
| 6 | 18.4297 | 21.4432 | 19.2834 | 19.0796 |
| 7 | 20.9768 | 23.2745 | 21.8240 | 21.5263 |
| 8 | 24.8654 | 27.5272 | 25.9683 | 25.6482 |



