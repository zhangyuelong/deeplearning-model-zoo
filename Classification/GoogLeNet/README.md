### 模型训练

```bash_script
python3 train.py
```

模型训练完毕之后, 在./weights/目录下生成模型的权重`googlenet.pth`.

### 模型推理

```bash_script
python3 predict.py
```

模型推理的过程中也顺便转成了onnx模型`googlenet.onnx`, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim googlenet.onnx googlenet_opt.onnx --input-shape 1,3,224,224 --dynamic-input-shape --enable-fuse-bn
```

测试一下onnx模型的推理:

```bash_script
python3 onnx_inference.py
```

### ONNX转换TensorRT脚本

```shell script
trtexec --onnx=googlenet.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x224x224 \
        --maxShapes=image:8x3x224x224 \
        --minShapes=image:1x3x224x224 \
        --shapes=image:4x3x224x224 \
        --saveEngine=googlenet.engine
```

### trt模型性能测试

测试环境:
- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```shell script
trtexec --loadEngine=googlenet.engine --shapes=image:?x3x224x224
```

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 3.10156 | 3.57745 | 3.18600 | 3.14182 |
| 2 | 4.59973 | 5.14984 | 4.73750 | 4.70129 |
| 3 | 6.22266 | 6.88831 | 6.43065 | 6.32520 |
| 4 | 7.73401 | 9.51575 | 8.18480 | 8.20160 |
| 5 | 9.09290 | 10.1533 | 9.41635 | 9.37268 |
| 6 | 10.9880 | 12.2847 | 11.4170 | 11.3737 |
| 7 | 12.4402 | 13.7941 | 13.1599 | 13.2289 |
| 8 | 14.3892 | 15.9209 | 14.9327 | 14.6689 |



