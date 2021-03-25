### 注意事项
原代码中使用的F.hardsigmoid和nn.Hardswish是无法转成onnx和trt的, 因此我将其替换成了自己实现的HSigmoid和HSwish, 这样就能成功转成onnx和trt了.

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

large:

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 1.98755 | 2.61694 | 2.04606 | 2.01050 |
| 2 | 2.94849 | 3.55392 | 3.00092 | 2.97656 |
| 3 | 4.01013 | 4.24976 | 4.04865 | 4.04352 |
| 4 | 4.91440 | 5.23875 | 5.00087 | 4.98407 |
| 5 | 5.83350 | 6.42996 | 5.89457 | 5.86884 |
| 6 | 7.08032 | 8.79092 | 7.41305 | 7.47440 |
| 7 | 8.00305 | 8.91763 | 8.08931 | 8.04187 |
| 8 | 9.13406 | 9.97479 | 9.27938 | 9.24243 |

small:

| Batch_Size | min(ms) | max(ms) | mean(ms) | median(ms) |
|:----:|:----:|:----:|:----:|:----:|
| 1 | 1.98755 | 2.61694 | 2.04606 | 2.01050 |
| 2 | 2.94849 | 3.55392 | 3.00092 | 2.97656 |
| 3 | 4.01013 | 4.24976 | 4.04865 | 4.04352 |
| 4 | 4.91440 | 5.23875 | 5.00087 | 4.98407 |
| 5 | 5.83350 | 6.42996 | 5.89457 | 5.86884 |
| 6 | 7.08032 | 8.79092 | 7.41305 | 7.47440 |
| 7 | 8.00305 | 8.91763 | 8.08931 | 8.04187 |
| 8 | 9.13406 | 9.97479 | 9.27938 | 9.24243 |
