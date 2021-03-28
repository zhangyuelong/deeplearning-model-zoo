### 模型训练

```bash_script
python3 train.py
```

模型训练完毕之后, 在./weights/目录下生成模型的权重`SegNet_weights.pth`.

### 模型推理

```bash_script
python3 predict.py
```

模型推理的过程中也顺便转成了onnx模型segnet.onnx, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim segnet.onnx segnet_opt.onnx --input-shape 1,3,224,224 --dynamic-input-shape --enable-fuse-bn
```

### ONNX转换TensorRT脚本

```bash_script
trtexec --onnx=segnet.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x224x224 \
        --maxShapes=image:8x3x224x224 \
        --minShapes=image:1x3x224x224 \
        --shapes=image:4x3x224x224 \
        --saveEngine=segnet.engine
```

### trt模型性能测试

测试环境:

- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```bash_script
trtexec --loadEngine=segnet.engine --shapes=image:?x3x224x224
```

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 3.26678 | 3.79514 | 3.32356 | 3.29834 |
| 2 | 5.24634 | 6.40442 | 5.32755 | 5.27112 |
| 3 | 7.19745 | 8.22992 | 7.35077 | 7.23688 |
| 4 | 9.09976 | 9.75244 | 9.23072 | 9.13574 |
| 5 | 11.0481 | 12.5439 | 11.2229 | 11.0879 |
| 6 | 13.0544 | 15.3620 | 13.3100 | 13.1490 |
| 7 | 14.9752 | 16.0486 | 15.1918 | 15.0629 |
| 8 | 17.0109 | 19.0373 | 17.3353 | 17.1364 |
