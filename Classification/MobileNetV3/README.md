### 注意事项

原代码中使用的F.hardsigmoid和nn.Hardswish是无法转成onnx和trt的, 因此我将其替换成了自己实现的HSigmoid和HSwish, 这样就能成功转成onnx和trt了.

### 模型训练

```bash_script
python3 train.py
```

模型训练完毕之后, 在./weights/目录下生成模型的权重`MobileNetV3_Large{Small}.pth`.

### 模型推理

```bash_script
python3 predict.py
```

模型推理的过程中也顺便转成了onnx模型`mobilenetv3_large{small}.onnx`, 利用此onnx模型, 就可以用trtexec转成trt模型.

生成的onnx有可能比较冗余, 可以先用onnxsim工具进行简化.

```bash_script
python3 -m onnxsim mobilenetv3_large{small}.onnx mobilenetv3_large{small}_opt.onnx --input-shape 1,3,224,224 --dynamic-input-shape --enable-fuse-bn
```

### ONNX转换TensorRT脚本

```shell script
trtexec --onnx=mobilenetv3_large{small}.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=image:4x3x224x224 \
        --maxShapes=image:8x3x224x224 \
        --minShapes=image:1x3x224x224 \
        --shapes=image:4x3x224x224 \
        --saveEngine=mobilenetv3_large{small}.engine
```

### trt模型性能测试

测试环境:
- GPU: GeForce GTX 1050 Ti 4GB
- TensorRT: 7.2.2.1

测试脚本:

```shell script
trtexec --loadEngine=mobilenetv3_large{small}.engine --shapes=image:?x3x224x224
```

**large**:

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 2.25897 | 2.55261 | 2.29798 | 2.28145 |
| 2 | 3.35632 | 3.74405 | 3.40076 | 3.38123 |
| 3 | 4.47656 | 4.98587 | 4.53382 | 4.50464 |
| 4 | 5.51404 | 5.76178 | 5.56155 | 5.55396 |
| 5 | 6.62134 | 6.97662 | 6.67031 | 6.65381 |
| 6 | 7.95410 | 8.81519 | 8.02716 | 7.99072 |
| 7 | 8.96045 | 9.95914 | 9.07347 | 9.01697 |
| 8 | 10.1868 | 11.6007 | 10.3117 | 10.2404 |

**small**:

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 1.26233 | 1.48163 | 1.29064 | 1.28241 |
| 2 | 1.74115 | 1.97743 | 1.77768 | 1.76733 |
| 3 | 2.20844 | 2.59647 | 2.26215 | 2.24194 |
| 4 | 2.66260 | 2.97470 | 2.70386 | 2.68326 |
| 5 | 3.12524 | 3.65015 | 3.25350 | 3.25586 |
| 6 | 3.64682 | 3.87903 | 3.69076 | 3.67587 |
| 7 | 4.09003 | 4.63895 | 4.15559 | 4.12738 |
| 8 | 4.61621 | 4.83774 | 4.65540 | 4.64624 |

### Triton Server部署

配置文件:

- pt模型

```
name: "mobilenetv3_pt"
platform: "pytorch_libtorch"
max_batch_size: 8

input [
  {
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [5]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

- trt模型

```
name: "mobilenetv3_trt"
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "outputs"
    data_type: TYPE_FP32
    dims: [ 5 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

性能报告(perf_client测试):

**small**

| Model/Batch_Size | Throughput(infer/sec) | Avg latency(ms) | p50 latency(ms) | p90 latency(ms) | p90 latency(ms) | p95 latency(ms) |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| torch/1 | 205.6 | 4.859 | 4.807 | 5.118 | 5.260 | 5485 |
| trt/1 | 344.4 | 2.902 | 2.995 | 3.263 | 3.426 | 3.783 |
| torch/2 | 366 | 5.466 | 5.401 | 5.739 | 5.930 | 6.297 |
| trt/2 | 420 | 4.761 | 4.770 | 5.839 | 5.902 | 6.206 |
| torch/3 | 448.8 | 6.684 | 6.189 | 8.144 | 9.125 | 11.177 |
| trt/3 | 563.4 | 5.325 | 4.979 | 7.549 | 7.921 | 9.080 |
| torch/4 | 576 | 6.944 | 6.897 | 7.099 | 7.220 | 7.714 |
| trt/4 | 577.6 | 6.922 | 6.869 | 8.138 | 9.623 | 10.033 |
| torch/5 | 588 | 8.490 | 8.387 | 8.811 | 9.118 | 9.429 |
| trt/5 | 615 | 8.134 | 7.853 | 11.349 | 11.782 | 11.998 |
| torch/6 | 591.6 | 10.148 | 9.883 | 10.906 | 11.058 | 11.373 |
| trt/6 | 596.4 | 10.062 | 11.058 | 11.322 | 13.744 | 14.060 |
| torch/7 | 623 | 11.239 | 11.183 | 11.463 | 11.636 | 12.183 |
| trt/7 | 712.6 | 9.832 | 9.603 | 12.520 | 14.694 | 15.847 |
| torch/8 | 628.8 | 12.704 | 12.628 | 12.818 | 13.435 | 13.657 |
| trt/8 | 659.2 | 12.126 | 12.277 | 14.346 | 14.538 | 17.834 |
