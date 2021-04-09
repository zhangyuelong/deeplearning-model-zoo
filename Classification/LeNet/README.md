### 注意事项

LeNet网络太简单了, 没什么好说的. 原网络的输入是28x28, 我把它改成了224x224了.

### 模型训练

```bash_script
python3 train.py
```

模型训练完毕之后, 在./weights/目录下生成模型的权重`lenet.pth`.

### 模型推理

```bash_script
python3 predict.py
```

模型推理的过程中也顺便转成了onnx模型`lenet.onnx`, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim lenet.onnx lenet_opt.onnx --input-shape 1,3,224,224 --dynamic-input-shape --enable-fuse-bn
```

### ONNX转换TensorRT脚本

```shell script
trtexec --onnx=lenet.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x224x224 \
        --maxShapes=image:8x3x224x224 \
        --minShapes=image:1x3x224x224 \
        --shapes=image:4x3x224x224 \
        --saveEngine=lenet.onnx
```

### trt模型性能测试

测试环境:
- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```shell script
trtexec --loadEngine=lenet.engine --shapes=image:?x3x224x224
```

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 0.94397 | 1.29453 | 0.98092 | 0.97112 |
| 2 | 1.66681 | 1.97000 | 1.69860 | 1.69043 |
| 3 | 2.12979 | 2.61389 | 2.25801 | 2.26782 |
| 4 | 2.44080 | 2.86865 | 2.49938 | 2.47144 |
| 5 | 2.83398 | 3.24310 | 2.89018 | 2.86011 |
| 6 | 3.24670 | 3.77527 | 3.32212 | 3.27856 |
| 7 | 3.78796 | 4.23984 | 3.87150 | 3.82983 |
| 8 | 4.21509 | 4.77679 | 4.30399 | 4.25067 |



