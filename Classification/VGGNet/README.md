### 模型训练

```bash_script
python3 train.py
```

模型训练完毕之后, 在./weights/目录下生成模型的权重`vgg{11, 13, 16, 19}net.pth`.

### 模型推理

```bash_script
python3 predict.py
```

模型推理的过程中也顺便转成了onnx模型`vgg{11, 13, 16, 19}net.onnx`, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim vgg{11, 13, 16, 19}net.onnx vgg{11, 13, 16, 19}net_opt.onnx --input-shape 1,3,224,224 --dynamic-input-shape --enable-fuse-bn
```

测试一下onnx模型的推理:

```bash_script
python3 onnx_inference.py
```

### ONNX转换TensorRT脚本

```shell script
trtexec --onnx=vgg{11, 13, 16, 19}net.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x224x224 \
        --maxShapes=image:8x3x224x224 \
        --minShapes=image:1x3x224x224 \
        --shapes=image:4x3x224x224 \
        --saveEngine=vgg{11, 13, 16, 19}net.engine
```

### trt模型性能测试

测试环境:
- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```shell script
trtexec --loadEngine=vgg{11, 13, 16, 19}net.engine --shapes=image:?x3x224x224
```

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 15.5127 | 17.5924 | 16.1016 | 16.4409 |
| 2 | 26.5659 | 28.4825 | 27.4404 | 26.9713 |
| 3 | 36.5498 | 40.6161 | 38.3390 | 38.9777 |
| 4 | 47.3956 | 51.4304 | 49.9427 | 50.4408 |
| 5 | 58.5609 | 62.7156 | 60.5530 | 59.8679 |
| 6 | 73.1405 | 78.7151 | 75.9924 | 77.0665 |
| 7 | 84.1257 | 89.8939 | 86.9006 | 85.3907 |
| 8 | 95.2374 | 102.807 | 98.9013 | 99.6387 |



