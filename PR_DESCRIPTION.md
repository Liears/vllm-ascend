# 优化 mtp_proposer.py 的 speculative decoding 实现

## 变更描述

本PR优化了`mtp_proposer.py`中speculative decoding的实现，主要涉及以下方面：

### 主要修改内容：
1. 重构了`_propose()`方法：
   - 优化了多token并行预测逻辑
   - 改进了pcp/dcp并行计算的slot管理
   - 增强了对超出max_model_len情况的处理

2. 完善了`dummy_run()`方法：
   - 统一了与`_propose()`方法的aclgraph处理
   - 优化了padding和all_gather/unpad逻辑

3. 关键参数处理：
   - 显式处理了`aclgraph_runtime_mode`
   - 统一了`num_tokens`和`num_input_tokens`的计算方式

## 代码改进点
- 更健壮的位置id处理，避免超出模型最大长度
- 优化了并行计算中的slot索引管理
- 完善了多token预测的batch处理逻辑
- 统一了不同运行模式下的输入处理

## 测试验证
修改已通过以下验证:
1. 单元测试覆盖所有speculative decoding场景
2. CUDA graph和eager模式下的功能测试
3. 并行计算(pcp/dcp)配置下的回归测试
4. max_model_len边界情况测试
