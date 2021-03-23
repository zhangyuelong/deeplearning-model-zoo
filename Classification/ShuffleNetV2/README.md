### 模型训练

```bash_script
python3 train.py
```

模型训练之后, 在./weights/目录下生成模型的权重`model-29.pth`.

### 模型推理

模型推理的过程中也顺便转成了onnx模型`shfflenetv2.onnx`, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim shufflenetv2.onnx shufflenetv2_opt.onnx --input-shape 1,3,224,224 --dynamic-input-shape --enable-fuse-bn
```

### ONNX转换TensorRT脚本

```shell script
trtexec --onnx=shufflenetv2.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x224x224 \
        --maxShapes=image:8x3x224x224 \
        --minShapes=image:1x3x224x224 \
        --shapes=image:4x3x224x224 \
        --saveEngine=shufflenetv2.engine
```

### trt模型性能测试

测试环境:
- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```shell script
trtexec --loadEngine=shufflenetv2.engine --shapes=image:?x3x224x224
```

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 1.25983 | 1.62988 | 1.29929 | 1.28735 |
| 2 | 1.73804 | 2.06033 | 1.76845 | 1.75342 |
| 3 | 2.30930 | 2.62163 | 2.34361 | 2.32690 |
| 4 | 2.86548 | 3.54205 | 3.02537 | 3.01086 |
| 5 | 3.38013 | 3.84396 | 3.44286 | 3.41919 |
| 6 | 4.10327 | 4.76601 | 4.16680 | 4.13458 |
| 7 | 4.66455 | 5.20610 | 4.72185 | 4.70972 |
| 8 | 5.34534 | 6.12262 | 5.43741 | 5.39801 |
