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
| 1 | 2.06519 | 2.54504 | 2.14450 | 2.10638 |
| 2 | 3.14221 | 3.73276 | 3.17795 | 3.16382 |
| 3 | 4.16248 | 4.58820 | 4.22804 | 4.18750 |
| 4 | 5.23090 | 5.80704 | 5.31340 | 5.28021 |
| 5 | 6.27850 | 7.20695 | 6.42659 | 6.36743 |
| 6 | 7.74976 | 8.88440 | 7.92282 | 7.86670 |
| 7 | 8.93689 | 10.3424 | 9.14666 | 9.04321 |
| 8 | 9.89062 | 10.8291 | 10.0688 | 10.0179 |



