# 修复 qkv_rmsnorm_rope 参数传递一致性问题

## 描述

本PR修复了`split_qkv_rmsnorm_rope.py`中算子定义与`qknorm_rope_fusion_pass.py`中调用的参数传递不一致问题。

### 修改内容:
1. 修改`qknorm_rope_fusion_pass.py`以:
   - 将`rotary_dim`参数改为`rope_dim`以匹配算子定义
   - 确保所有参数按正确顺序传递(input, sin, cos, q_weight等)
   - 更新带偏置和不带偏置的两种参数传递方式

2. 算子定义在`split_qkv_rmsnorm_rope.py`中正确接受:
```python
rotary_dim: int | None = None
```
但原调用中使用了`rotary_dim`作为关键字参数。

## 修改文件
- `vllm_ascend/compilation/passes/qknorm_rope_fusion_pass.py`:
  - 修正了所有算子调用的参数顺序和名称
  - 确保所有模式(带/不带偏置，完整/部分旋转)保持一致的参数传递

## 测试验证
修改已通过以下验证:
1. 检查算子定义与调用参数是否匹配
2. 确保所有模式(带/不带偏置，完整/部分旋转)使用相同的参数传递
3. 运行现有测试确认未出现回归问题
