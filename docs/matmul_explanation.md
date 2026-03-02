# Vulkan Compute Shader 矩阵乘法详解

## 一、分层次的分块策略

整个算法采用了**三层分块**策略，从粗到细：

```
层级1: WorkGroup 级别 (BM × BN × BK 块)
层级2: Warp 级别 (WM × WN 块)  
层级3: Thread 级别 (TM × TN 块)
```

### 1.1 WorkGroup 级别分块

```
矩阵 A (M × K)          矩阵 B (K × N)          矩阵 C (M × N)
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│             │        │             │        │             │
│  BM×BK 块   │   ×    │  BK×BN 块   │   =    │  BM×BN 块   │
│             │        │             │        │             │
└─────────────┘        └─────────────┘        └─────────────┘
```

**参数说明：**
- `BM = 64`: 每个工作组处理 64 行（M 维度）
- `BN = 64`: 每个工作组处理 64 列（N 维度）
- `BK = 32`: K 维度每次处理 32 个元素

**工作组分配：**
```glsl
const uint blocks_m = (p.M + BM - 1) / BM;  // M 维度需要多少个块
const uint ir = gl_WorkGroupID.x % blocks_m;  // 当前块在 M 维度的索引
const uint ik = gl_WorkGroupID.x / blocks_m;  // K 维度的分块索引
const uint ic = gl_WorkGroupID.y;             // N 维度的块索引
```

### 1.2 Warp 级别分块

每个 WorkGroup 包含 64 个线程 = 2 个 Warp（假设 WARP=32）

```
BM×BN 块 (64×64)
┌─────────┬─────────┐
│ Warp 0  │ Warp 1  │  每个 Warp 处理 WM×WN (32×32)
│ 32×32   │ 32×32   │
├─────────┼─────────┤
│ Warp 2  │ Warp 3  │
│ 32×32   │ 32×32   │
└─────────┴─────────┘
```

**Warp 定位：**
```glsl
const uint warp_i = gl_LocalInvocationID.x / WARP;  // 0-3, 当前是第几个 warp
const uint warp_r = warp_i % (BM / WM);  // Warp 在 M 维度的位置 (0-1)
const uint warp_c = warp_i / (BM / WM);  // Warp 在 N 维度的位置 (0-1)
```

### 1.3 Thread 级别分块

每个线程处理 TM×TN 的小块：

```
WM×WN 块 (32×32) - 由一个 Warp 的 32 个线程处理
┌──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  每个格子是 TM×TN (4×2)
├──┼──┼──┼──┼──┼──┼──┼──┤  由一个线程处理
│  │  │  │  │  │  │  │  │
└──┴──┴──┴──┴──┴──┴──┴──┘
```

**线程定位：**
```glsl
const uint tiw = gl_LocalInvocationID.x % WARP;  // Warp 内的线程 ID (0-31)
const uint tiwr = tiw % (WSUBM / TM);  // 线程在 M 维度的位置
const uint tiwc = tiw / (WSUBM / TM);  // 线程在 N 维度的位置
```

## 二、主要优化技术

### 2.1 共享内存优化 (Shared Memory)

**核心思想：** 将全局内存的数据加载到共享内存，供工作组内所有线程重复使用。

```glsl
shared vec2 buf_a[BM * SHMEM_STRIDE];  // 64 × 17 = 1088 个 vec2
shared vec2 buf_a[BN * SHMEM_STRIDE];  // 64 × 17 = 1088 个 vec2
```

**为什么使用 vec2？**
- 一次存储 2 个 float，批量操作
- 减少内存事务次数

**SHMEM_STRIDE 的作用：**
```glsl
#define SHMEM_STRIDE (BK / 2 + 1)  // 32/2 + 1 = 17
```

这是为了**避免 bank conflict**：
- 共享内存分为 32 个 bank
- 如果多个线程同时访问同一个 bank 的不同地址，会产生冲突
- 通过添加 padding (+1)，使连续行错开 bank，避免冲突

**数据加载模式：**

```
全局内存 A (M×K)                共享内存 buf_a
┌────────────┐                 ┌────────────┐
│ BM 行      │ ───加载───>     │ BM × BK/2  │
│ × BK 列    │                 │ (vec2)     │
└────────────┘                 └────────────┘
        ↓
  64个线程协同加载
  每个线程负责一部分
```

### 2.2 寄存器缓存 (Register Cache)

**两级缓存策略：**

```
全局内存 (DRAM, 慢)
    ↓
共享内存 (Shared Memory, 中速)
    ↓
寄存器 (Registers, 快)
```

```glsl
vec2 sums[WMITER * TM * WNITER * TN/2];  // 累加器，存在寄存器
vec4 cache_a[WMITER * TM];                // A 的缓存
vec4 cache_b;                             // B 的缓存
```

**为什么用 vec4？**
- 一次从共享内存读 4 个 float
- 减少共享内存访问次数
- 提高数据重用率

### 2.3 循环展开 (Loop Unrolling)

```glsl
[[unroll]] for (uint i = 0; i < BK / BK_STEP; i++) {
    // 加载和计算
}
```

**好处：**
- 减少循环开销（条件判断、跳转）
- 增加指令级并行性（ILP）
- 让编译器更好地优化

### 2.4 批量加载 (Vectorized Loading)

```glsl
#define LOAD_VEC_BATCH_A 2
#define LOAD_VEC_BATCH_B 2

// 一次加载 2 个 float，存为 vec2
buf_a[...] = vec2(data_a[idx], data_a[idx + 1]);
```

**好处：**
- 减少内存事务次数
- 提高内存带宽利用率
- 合并内存访问（memory coalescing）

### 2.5 FMA 指令 (Fused Multiply-Add)

```glsl
sums[idx].x = fma(cache_a[...].x, cache_b.x,
              fma(cache_a[...].y, cache_b.y,
              fma(cache_a[...].z, cache_b.z,
              fma(cache_a[...].w, cache_b.w, sums[idx].x))));
```

**优势：**
- 单指令完成乘法和加法：`a * b + c`
- 减少舍入误差
- 提高计算吞吐量

## 三、详细执行流程

### 3.1 初始化阶段

```glsl
// 1. 计算当前工作组/线程的位置
const uint ir = ...;  // M 维度块索引
const uint ic = ...;  // N 维度块索引
const uint ik = ...;  // K 维度分块索引

// 2. 初始化累加器
[[unroll]] for (uint i = 0; i < WMITER*TM*WNITER*TN/2; i++) {
    sums[i] = vec2(0.0f, 0.0f);
}
```

### 3.2 主循环：分块计算

```glsl
for (uint block = start_k; block < end_k; block += BK) {
    // 步骤 1: 加载 A 和 B 的块到共享内存
    for (uint l = 0; l < BM; l += loadstride_a) {
        load_a_to_shmem(...);  // 64 个线程协同加载
    }
    for (uint l = 0; l < BN; l += loadstride_b) {
        load_b_to_shmem(...);
    }
    
    barrier();  // 同步，确保所有线程都加载完成
    
    // 步骤 2: 从共享内存到寄存器，进行计算
    for (uint i = 0; i < BK / BK_STEP; i++) {
        // 2a: 加载 A 的数据到寄存器
        for (uint wsir = 0; wsir < WMITER; wsir++) {
            for (uint j = 0; j < TM; j++) {
                cache_a[wsir * TM + j].xy = buf_a[...];
                cache_a[wsir * TM + j].zw = buf_a[...];
            }
        }
        
        // 2b: 对于 B 的每一列，加载并计算
        for (uint wsic = 0; wsic < WNITER; wsic++) {
            for (uint cc = 0; cc < TN; cc++) {
                cache_b.xy = buf_b[...];
                cache_b.zw = buf_b[...];
                
                // 2c: 执行矩阵乘法累加
                for (uint wsir = 0; wsir < WMITER; wsir++) {
                    for (uint cr = 0; cr < TM / 2; cr++) {
                        sums[idx] += cache_a[...] * cache_b;
                    }
                }
            }
        }
    }
    
    barrier();  // 同步，准备下一个块
}
```

### 3.3 写回结果

```glsl
// 将累加器的结果写回全局内存
for (uint wsic = 0; wsic < WNITER; wsic++) {
    for (uint wsir = 0; wsir < WMITER; wsir++) {
        for (uint cc = 0; cc < TN; cc++) {
            for (uint cr = 0; cr < TM / 2; cr++) {
                // 边界检查
                if (dr_warp + 2*cr < p.M && dc_warp + cc < p.N) {
                    data_d[...] = sums[idx].x;
                }
                if (dr_warp + 2*cr+1 < p.M && dc_warp + cc < p.N) {
                    data_d[...] = sums[idx].y;
                }
            }
        }
    }
}
```

## 四、内存访问模式分析

### 4.1 合并访问 (Coalesced Access)

**加载 A 矩阵：**
```
线程 ID:  0    1    2    3   ...   31
访问地址: [0]  [1]  [2]  [3] ... [31]  ← 连续地址
         └────────┬────────┘
              单次内存事务
```

**为什么重要？**
- GPU 内存以 32/64/128 字节为单位事务
- 如果 32 个线程访问连续地址，可以合并成一次事务
- 否则需要多次事务，性能下降

### 4.2 数据重用

**A 矩阵的重用：**
```
A 的一行 (BK 个元素) 被同一工作组的所有线程重用
└─> 加载到共享内存，避免重复从全局内存读取
```

**重用因子计算：**
- A 的每个元素被 BN 个线程使用：重用 64 次
- B 的每个元素被 BM 个线程使用：重用 64 次

## 五、性能分析

### 5.1 计算强度 (Arithmetic Intensity)

```
计算量：2 × BM × BN × BK = 2 × 64 × 64 × 32 = 262,144 次浮点运算
内存访问：
  - 读 A: BM × BK × 4 bytes = 64 × 32 × 4 = 8,192 bytes
  - 读 B: BN × BK × 4 bytes = 64 × 32 × 4 = 8,192 bytes  
  - 写 C: BM × BN × 4 bytes = 64 × 64 × 4 = 16,384 bytes
  总计: 32,768 bytes

计算强度 = 262,144 / 32,768 ≈ 8 FLOP/byte
```

这是一个**计算密集型**操作，适合 GPU。

### 5.2 占用率 (Occupancy)

**资源使用：**
- 每个工作组：64 个线程
- 共享内存：~34 KB (2 × 1088 × vec2 × 4 bytes)
- 寄存器：~100 个/线程

**限制因素：**
- 通常受共享内存限制
- 需要平衡分块大小和占用率

### 5.3 优化效果对比

| 优化技术 | 性能提升 |
|---------|---------|
| 无优化（naive） | 1× 基准 |
| + 分块 | 3-5× |
| + 共享内存 | 8-12× |
| + 寄存器缓存 | 15-20× |
| + 循环展开 + FMA | 25-35× |

## 六、关键参数调优

### 6.1 分块大小选择

```
BM、BN、BK 的选择需要平衡：
- 更大 → 更多数据重用，但更多共享内存
- 更小 → 更高占用率，但数据重用少

典型值：
- NVIDIA: BM=128, BN=128, BK=32
- AMD: BM=64, BN=64, BK=16
```

### 6.2 Warp 配置

```
WM、WN 影响：
- 单个 Warp 的工作量
- 寄存器压力
- 指令级并行

常见配置：WM=32, WN=32 (1024 个元素/warp)
```

### 6.3 线程粒度

```
TM、TN 决定：
- 每个线程的计算量
- 寄存器使用量

典型：TM=4-8, TN=2-4
```

## 七、调试和验证

### 关键检查点：

1. **索引正确性**
   - 验证每个线程访问的数据位置
   - 检查边界条件

2. **同步正确性**
   - barrier() 必须在正确位置
   - 避免死锁

3. **数值精度**
   - 检查 FMA 的累加顺序
   - 考虑浮点舍入误差

4. **性能分析**
   - 使用 Nsight 等工具
   - 查看内存吞吐量、计算利用率

## 八、总结

这个 shader 的核心优化策略是：

1. **分层分块** - 充分利用 GPU 的层次结构
2. **内存层次** - 全局→共享→寄存器，逐级加速
3. **数据重用** - 共享内存实现块内重用
4. **向量化** - vec2/vec4 批量操作
5. **指令优化** - FMA、循环展开

这些技术使得 GPU 矩阵乘法比 naive 实现快 **20-30 倍**！
