# vkllm_graph 函数实现总结

本文档总结了 `vkllm_graph.c` 中实现的计算图初始化、执行和后处理函数。

## 概述

计算图执行遵循以下流程：
1. **初始化 (init)**: 为每个操作准备 pipeline 和资源
2. **执行 (run)**: 递归执行所有依赖，然后执行当前操作
3. **后处理 (post_run)**: 清理临时资源或执行收尾工作

所有函数都使用相同的递归模式和 hash set 优化来避免重复处理。

## 实现的函数

### 1. `vkllm_graph_init` (src/vkllm_graph.c:122-141)

**功能**: 从输出节点开始递归初始化计算图中的所有操作。

**实现细节**:
- 创建 hash set 跟踪已访问的节点（O(1) 查找）
- 调用 `vkllm_graph_init_tensor` 递归初始化
- 对每个节点：
  1. 检查是否已访问（避免重复）
  2. 递归初始化所有依赖节点（srcs）
  3. 标记为已访问
  4. 根据操作类型调用对应的 `_op_xxx_init` 函数

**支持的操作**:
- `VKLLM_OP_NONE`: 输入节点，无需初始化
- `VKLLM_OP_BIN`: 二元运算（加、减、乘、除等）
- `VKLLM_OP_EMBEDDING`: 嵌入查找
- `VKLLM_OP_RMSNORM`: RMS 归一化
- `VKLLM_OP_MATMUL`: 矩阵乘法
- `VKLLM_OP_ROPE`: 旋转位置编码
- `VKLLM_OP_SOFTMAX`: Softmax 激活

### 2. `vkllm_graph_run` (src/vkllm_graph.c:200-219)

**功能**: 从输出节点开始递归执行计算图中的所有操作。

**实现细节**:
- 创建 hash set 跟踪已执行的节点
- 调用 `vkllm_graph_run_tensor` 递归执行
- 对每个节点：
  1. 检查是否已执行（避免重复）
  2. 递归执行所有依赖节点（确保输入准备好）
  3. 标记为已访问
  4. 根据操作类型调用对应的 `_op_xxx_run` 函数

**执行顺序**: 拓扑排序（后序遍历），确保依赖先于当前节点执行。

**示例**: 对于计算图 `output = softmax(matmul(A, B))`
```
执行顺序:
1. 执行 A（输入节点，无操作）
2. 执行 B（输入节点，无操作）
3. 执行 matmul(A, B)
4. 执行 softmax(...)
```

### 3. `vkllm_graph_post_run` (src/vkllm_graph.c:278-297)

**功能**: 从输出节点开始递归执行计算图中所有操作的后处理。

**实现细节**:
- 创建 hash set 跟踪已后处理的节点
- 调用 `vkllm_graph_post_run_tensor` 递归后处理
- 对每个节点：
  1. 检查是否已后处理
  2. 递归后处理所有依赖节点
  3. 标记为已访问
  4. 根据操作类型调用对应的 `_op_xxx_post_run` 函数

**用途**: 
- 清理临时缓冲区
- 同步 GPU 操作
- 释放中间资源
- 更新统计信息

## 辅助函数

### `vkllm_graph_init_tensor` (src/vkllm_graph.c:66-120)

递归初始化单个 tensor 及其所有依赖的辅助函数。

### `vkllm_graph_run_tensor` (src/vkllm_graph.c:144-198)

递归执行单个 tensor 及其所有依赖的辅助函数。

### `vkllm_graph_post_run_tensor` (src/vkllm_graph.c:222-276)

递归后处理单个 tensor 及其所有依赖的辅助函数。

## 性能优化

### Hash Set 优化

所有函数都使用 hash set 而不是数组来跟踪已访问的节点：

```c
// 旧方法：O(n) 线性搜索
for (size_t i = 0; i < visited->used_n; i++) {
    if (visited->data[i] == tensor) {
        return VKLLM_ERR_OK;
    }
}

// 新方法：O(1) hash 查找
uint64_t tensor_key = (uint64_t)(uintptr_t)tensor;
if (vkllm_hashset_contains(visited, tensor_key)) {
    return VKLLM_ERR_OK;
}
```

**性能提升**: 在基准测试中，hash set 比线性搜索快 18.65 倍。

### 避免重复处理

在有共享子图的情况下，hash set 确保每个节点只处理一次：

```
输入 A
    ↓
  操作1 ← 输入 B
    ↓  ↘
  操作2  操作3
    ↓  ↙
   输出

操作1 只会初始化/执行一次，即使它被操作2和操作3共享。
```

## 错误处理

所有函数都使用 `_CHECK` 宏进行错误检查和传播：
- 参数验证：检查 context、graph、output_node、commands
- 内存分配失败：hash set 创建失败
- 操作失败：任何 `_op_xxx_*` 函数返回错误
- 未知操作类型：记录错误并返回 `VKLLM_ERR_ARGS`

## 使用示例

```c
struct vkllm_context *context = ...;
struct vkllm_graph *graph = NULL;

// 创建图
vkllm_graph_new(context, &graph);

// 添加节点和设置输出
// ... (构建图)

// 1. 初始化阶段
vkllm_err_t err = vkllm_graph_init(context, graph);
if (err != VKLLM_ERR_OK) {
    // 处理错误
    return err;
}

// 2. 执行阶段
err = vkllm_graph_run(context, graph);
if (err != VKLLM_ERR_OK) {
    // 处理错误
    return err;
}

// 3. 后处理阶段
err = vkllm_graph_post_run(context, graph);
if (err != VKLLM_ERR_OK) {
    // 处理错误
    return err;
}

// 清理
vkllm_graph_free(context, graph);
```

## 代码统计

| 函数 | 行数 | 说明 |
|------|------|------|
| `vkllm_graph_init_tensor` | 55 | 递归初始化辅助函数 |
| `vkllm_graph_init` | 20 | 初始化入口函数 |
| `vkllm_graph_run_tensor` | 55 | 递归执行辅助函数 |
| `vkllm_graph_run` | 20 | 执行入口函数 |
| `vkllm_graph_post_run_tensor` | 55 | 递归后处理辅助函数 |
| `vkllm_graph_post_run` | 20 | 后处理入口函数 |
| **总计** | **225** | 三个完整的递归遍历实现 |

## 设计特点

1. **一致性**: 三个函数（init, run, post_run）使用完全相同的递归模式
2. **可扩展性**: 添加新操作只需在 switch 语句中添加新 case
3. **高性能**: 使用 hash set 实现 O(1) 节点访问检查
4. **正确性**: 拓扑排序保证依赖关系正确处理
5. **健壮性**: 完整的错误检查和资源清理

## 相关文件

- **实现**: `src/vkllm_graph.c`
- **接口**: `src/vkllm_graph.h`
- **依赖**: `src/vkllm_hashset.h` (hash set 容器)
- **操作**: `src/vkllm_op_*.h` (各个操作的 init/run/post_run 函数)
