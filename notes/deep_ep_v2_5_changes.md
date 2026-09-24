# DeepEP V2.5 详解:架构重构、通信内核与新增能力

> **分析对象**: `v2.5` 分支,tip commit `8c1d13a`(2026-09-23,`remove warpget opt`),版本号 `2.5.0`
> **对比基线**: 基线 commit `76f1427`(V2.0 公开前快照,version 2.0.0:`DeepEP V2` 的 elastic 形态,含 V1 legacy、自带构建期 JIT 与 NVSHMEM 依赖)
> **方法**: 基于两个 commit 的完整代码阅读(全部 16 个增量提交、162 个变更文件),文中所有结论均标注代码出处
> **分析日期**: 2026-09-24

---

## 目录

- [1. 版本概述](#1-版本概述)
- [2. 总体架构](#2-总体架构)
- [3. 统一 Buffer 框架:V2.5 的核心重构](#3-统一-buffer-框架v25-的核心重构)
- [4. 通信后端:全面迁移到 NCCL Gin](#4-通信后端全面迁移到-nccl-gin)
- [5. DeepJIT 运行时编译与构建体系重构](#5-deepjit-运行时编译与构建体系重构)
- [6. EPBuffer:MoE 通信内核详解](#6-epbuffermoe-通信内核详解)
- [7. BucketBuffer:CP/DP 集合通信详解](#7-bucketbuffercpdp-集合通信详解)
- [8. EngramBuffer:远程记忆访问详解](#8-engrambuffer远程记忆访问详解)
- [9. PPBuffer:流水线并行通信详解](#9-ppbuffer流水线并行通信详解)
- [10. 公开 API 面变化(v2.0 → v2.5)](#10-公开-api-面变化v20--v25)
- [11. 构建与部署需求变化](#11-构建与部署需求变化)
- [12. 被移除的功能与迁移说明](#12-被移除的功能与迁移说明)
- [13. 测试体系](#13-测试体系)
- [14. 总结:设计哲学与功能清单](#14-总结设计哲学与功能清单)

---

## 1. 版本概述

### 1.1 一句话总结

**V2.5 把 V2.0 时期"一个 `ElasticBuffer` 包打天下"的实验形态,重构为 EP / Bucket / PP / Engram 四个职责单一的专用 Buffer(共享统一生命周期),彻底删除 V1 全部代码与 NVSHMEM 依赖,将 GPU 内核的编译方式从自带的构建期 JIT 切换到独立的 DeepJIT 运行时编译,并在此架构上新增了动态冗余专家负载均衡(服务 MoonEP/UltraEP 类方案)与 CP/DP 桶式集合通信两大能力。**

### 1.2 变更规模

| 指标 | 数值 |
|---|---|
| 增量提交数 | 16 个(`76f1427..8c1d13a`) |
| 变更文件数 | 162 个 |
| 新增 / 删除行数 | +12,612 / −17,445 |
| 核心提交 | `1a1ffba DeepEP V2.5`(单次 squash 了 Nightly 仓库相对公开 main 的 **534 个提交**) |
| 删除的主要代码 | `csrc/kernels/legacy/`(约 7,000 行 V1 内核)、`csrc/legacy/`(1,984 行 V1 Buffer,其中 `buffer.hpp` 恰为 1,794 行)、`csrc/elastic/`(1,307 行 ElasticBuffer)、`csrc/jit/`(786 行、8 个头文件的自带 JIT)、`deep_ep/buffers/{legacy,elastic}.py` |
| 新增的主要代码 | `csrc/buffers/`(BufferBase + 四个专用 Buffer 共 2,004 行)、`deep_ep/buffers/{ep,bucket,engram,pp}.py`(1,545 行)、`deep_ep/include/deep_ep/impls/bucket/`(约 1,770 行 kernel)、`csrc/kernels/{comm,ep,bucket,engram,pp,driver}/`(内核启动层,`driver.cpp` 所在) |

### 1.3 提交时间线

V2.0(4 月底 EPv2 公开)到 V2.5 的 16 个增量提交可分为四段:

| 阶段 | 时间 | 提交 | 内容 |
|---|---|---|---|
| EPv2 修复期 | 04-30 ~ 08-04 | `b306af0` ~ `01dc3aa`(13 个) | EPv2 公开发布后的稳定性修复:单节点初始化(`#630`)、NCCL Device API 运行时版本兼容(`#688`)、跨 NVLink+RDMA scale-up 域时 GIN barrier 前的 system-scope release(`#715`)、mbarrier 与 TMA load 之间的 `fence.proxy.async.shared::cta`(`#642`)等 |
| 插入修复 | 09-16 | `a56d615 Ensure that the last get rings the doorbell`(`#752`) | Engram fetch doorbell 时序修复:最后一批 get 必须敲 doorbell,否则 flush 不完整(见 §8.2);该提交夹在 `01dc3aa` 与 `1a1ffba` 之间,日期在修复期区间之外 |
| V2.5 主干 | 09-23 | `1a1ffba DeepEP V2.5` | 一次 squash 合并 Nightly 仓库 534 个提交,即本文分析的绝大部分:四 Buffer 拆分、DeepJIT、删 V1/删 NVSHMEM、LB 原语、Bucket 集合通信、多层 Engram |
| 收尾 | 09-23 | `8c1d13a remove warpget opt` | 移除 Engram fetch 中的 `ncclGinOptFlagsWarpGet` 优化路径(该优化依赖扩展版 NCCL 构建,移除后 Engram 对特殊 NCCL 构建的依赖解除) |

### 1.4 版本演进脉络

```
V1.x (legacy, 1.2.1)          V2.0 (76f1427)                V2.5 (v2.5 分支)
─────────────────────         ─────────────────────         ─────────────────────
Buffer (V1 API)               ElasticBuffer                 EPBuffer        MoE dispatch/combine
  + NVSHMEM 后端                (EP + Engram + PP            EngramBuffer    远程记忆 (多层, GPU/CPU 存储)
  + intranode/internode/        混合在一个 buffer 里)         PPBuffer        流水线并行 send/recv
    low_latency 内核            + 仍携带 V1 legacy 代码        BucketBuffer    CP/DP 集合通信 (全新)
                                + 自带构建期 JIT               + 统一 BufferBase 生命周期
                                                                  + BufferAllocator 分配规划
                                                                  + DeepJIT 运行时编译
                                                                  + NCCL Gin 为唯一通信后端
                                                                  + 动态冗余专家 LB 原语 (全新)
```

---

## 2. 总体架构

### 2.1 分层结构

V2.5 的代码组织是严格分层的,自底向上为:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Python API 层  deep_ep/                                                 │
│   buffers/  EPBuffer · BucketBuffer · EngramBuffer · PPBuffer           │
│             BufferBase(生命周期) · BufferAllocator(分配规划)             │
│   comm/     NCCLCommHandle · barrier · stream · 域大小查询               │
│   utils/    EventOverlap · 带宽探测 · 参考实现(refs.py)                  │
├─────────────────────────────────────────────────────────────────────────┤
│ C++ host 扩展层  csrc/  →  deep_ep._C.so (install 时编译, 仅 3 个源文件) │
│   python_api.cpp · kernels/comm/context.cpp · kernels/driver/driver.cpp │
│   buffers/  四个 Buffer 的 host 侧实现(参数校验/内存/workspace 管理)      │
│   runtime/jit.hpp  DeepJIT 运行时初始化                                  │
├─────────────────────────────────────────────────────────────────────────┤
│ 运行时 JIT 层  third-party/deep_jit (submodule → deepseek-ai/DeepJIT)   │
│   按 GPU 架构 + 模板参数组合, 首次调用时 nvcc 编译 .cuh 内核并缓存         │
├─────────────────────────────────────────────────────────────────────────┤
│ Device 层  deep_ep/include/deep_ep/                                     │
│   comm/     NCCLGin(设备侧通信句柄) · barrier                            │
│   layout/   各 buffer 的内存布局与信号工作区(token/eplb/workspace/...)    │
│   impls/    全部 __global__ 内核(ep/ bucket/ engram/ pp/ comm/)          │
│   common/   PTX 原语封装 · 数学工具 · 异常                                │
├─────────────────────────────────────────────────────────────────────────┤
│ 传输层  NCCL(≥2.32.3, Device API + Gin) + NVLink(对称指针) + RDMA(GDAKI)│
└─────────────────────────────────────────────────────────────────────────┘
```

**关键事实:host 扩展在 install 时只编译 3 个源文件**(`setup.py:81-83`: `csrc/python_api.cpp`、`csrc/kernels/comm/context.cpp`、`csrc/kernels/driver/driver.cpp`),所有 `__global__` 内核都不进 wheel,而是由 DeepJIT 在运行时针对当前 GPU 编译。这意味着 **安装一台机器不需要 GPU,也不需要 `TORCH_CUDA_ARCH_LIST`**——这是 V2.5 构建体系的根本性变化(见 §5)。

### 2.2 域模型:scale-up / scale-out 与三种 team

V2.5 的通信域抽象建立在 NCCL 的三种 team 之上(`deep_ep/include/deep_ep/comm/handle.cuh:22`):

| Team | 含义 | 典型范围 |
|---|---|---|
| `ncclTeamTagWorld` | 全部 rank | 整个进程组 |
| `ncclTeamTagLsa` (LSA, NVLink 域) | 节点内 NVLink 互通的 rank | 8 卡节点 |
| `ncclTeamTagRail` | 跨节点的"同轨"rank(相同节点内编号) | 跨节点同位置 GPU |

Python 层暴露两级视角(`EPBuffer.__init__`):

- **逻辑域**:`num_scaleout_ranks × num_scaleup_ranks`(scale-up = 节点内 NVLink 域,scale-out = 节点间 RDMA 域);
- **物理域**:`num_rdma_ranks × num_nvlink_ranks`。

所有内核模板同时支持"纯 NVLink、纯 RDMA、hybrid(两级)"三种拓扑路径,由 rank 数在运行时/编译期分流。V2.0 已有该模型,V2.5 将其推广到全部四个 buffer。

### 2.3 内存模型:对称内存窗口 + 对齐体系

V2.5 的通信内存全部基于 **NCCL 对称内存窗口(symmetric window)**:每个 buffer 申请一块对称内存,注册为 `ncclWindow_t`,任意 rank 可通过窗口内偏移寻址同位置内存——NVLink 域内用对称指针直访(`ncclGetLsaPointer`),域外用 Gin 的 RDMA get/put。

对齐体系在 `deep_ep/include/deep_ep/common/compiled.cuh:61-74` 集中定义,并保证整除链:

| 常量 | 值 | 用途 |
|---|---|---|
| `kNumTMAAlignmentBytes` | 32 B | TMA 指令对齐(V2.0 为 16 B,收紧) |
| `kNumRDMAAlignmentBytes` | 64 B | RDMA 传输单元对齐 |
| `kNumAllocationAlignmentBytes` | 2 MiB | 对称分配粒度(BufferAllocator 的对齐基准) |
| `kNumMaxRanks` | 1024 | scale-out/scale-up/RDMA/NVLink rank 数统一上限 |
| `kNumMaxContexts` / `kNumMaxQPs` / `kDefaultQPDepth` | 8 / 1024 / 1024 | GIN 上下文(QP)体系 |
| `kNumMaxSignalBytes` | 16 MiB | 信号工作区上限 |

另一个容量上限变化:`kNumMaxExpertsPerRank`(每 rank 专家数)从 V2.0 的 256(`common/layout.cuh:21`)放宽到 V2.5 的 512(`layout/ep/workspace.cuh:11`),单 rank 可承载更多专家。

**`BufferAllocator` 的"规划—物化"模式**(§3.3)让多个张量(通信 buffer、LB 区、storage 等)先以 meta tensor 规划布局,再一次性物化到同一段连续、2 MiB 对齐的对称存储中——这是四个 buffer 共享内存管理的基础设施。

---

## 3. 统一 Buffer 框架:V2.5 的核心重构

### 3.1 背景:V2.0 的 ElasticBuffer 问题

V2.0 的全部功能(EP dispatch/combine、Engram、PP)挤在一个 `ElasticBuffer` 里:`deep_ep/buffers/elastic.py`(883 行) + `csrc/elastic/buffer.hpp`(1,307 行)。问题在于:

1. **生命周期耦合**:只想用 Engram 的用户也必须接受 EP 的 buffer 尺寸计算与初始化协议;
2. **内存不可规划**:各功能各自申请对称内存,无法统一规划对齐布局;
3. **测试与裁剪困难**:无法单独构建/验证某一类通信。

### 3.2 拆分方案:BufferBase + 四个专用 Buffer

V2.5 把 `ElasticBuffer` 拆成四个 Buffer,各自只带自己的运行时:

| Buffer | 职责 | Python | C++ host | 典型构造参数 |
|---|---|---|---|---|
| `EPBuffer` | MoE dispatch/combine + 冗余专家 LB | `deep_ep/buffers/ep.py`(906 行) | `csrc/buffers/ep.hpp`(1,100 行) | MoE 配置(或裸 `num_bytes`)、`lb_allocation_plan_or_num_bytes`、hybrid/overlap 开关 |
| `BucketBuffer` | 批量 all-gather / reduce-scatter / all-reduce(CP/DP) | `deep_ep/buffers/bucket.py`(398 行) | `csrc/buffers/bucket.hpp`(374 行) | 一个或多个 process group、分配规划 |
| `EngramBuffer` | 远程记忆(Engram)RDMA 读取 | `deep_ep/buffers/engram.py`(169 行) | `csrc/buffers/engram.hpp`(319 行) | GPU 区字节数、每 rank RDMA storage 字节数、CPU/GPU storage、QP 配置 |
| `PPBuffer` | 流水线并行相邻 rank send/recv | `deep_ep/buffers/pp.py`(72 行) | `csrc/buffers/pp.hpp`(148 行) | 单 tensor 最大字节数、最大 in-flight 数 |

共同生命周期由抽象基类 `BufferBase` 承载(`deep_ep/buffers/base.py`):

- `main_context`:首个通信上下文;
- `print_memory_usage()`:打印本 rank 的 workspace / 通信 buffer / RDMA storage 占用;
- `destroy()`:释放 C++ 运行时,仅当构造时 `explicitly_destroy=True` 时允许调用(否则由析构函数释放)。

**迁移对照表**(V2.0 API → V2.5 API):

| V2.0(`ElasticBuffer`) | V2.5 |
|---|---|
| `ElasticBuffer.dispatch / combine` | `EPBuffer.dispatch / combine`(API 语义基本保留,新增 deferred epilogue 等参数,见 §6) |
| `ElasticBuffer.engram_*`(单层) | `EngramBuffer.set_config / write / fetch`(多层化,见 §8) |
| `ElasticBuffer.pp_send / pp_recv` | `PPBuffer.send / recv`(参数收敛为 ring 相邻 rank) |
| ——(不存在) | `BucketBuffer` 全部为新增 |
| 构造时统一分配 | 各 buffer 独立分配,可用 `BufferAllocator` 预先规划 |

### 3.3 BufferAllocator:先规划、后物化

`deep_ep/buffers/allocator.py` 提供了一个两阶段分配器:

```python
plan = BufferAllocator()                       # 对齐基准 = get_num_allocation_alignment() (2 MiB)
t1 = plan.allocate((1024, 7168), torch.bfloat16)   # 返回 meta tensor,记录 offset
t2 = plan.allocate((512,), torch.float32)
buffer = EPBuffer(group, ..., lb_allocation_plan_or_num_bytes=plan)
# buffer 构造完成后, EPBuffer 内部调用 plan.materialize(runtime.lb_storage):
# 把每个 meta tensor 用 storage.narrow(...) 切成真实视图, torch.utils.swap_tensors 原地替换
```

要点:

- `allocate()` 只登记 `(shape, dtype, offset)`,张量位于 meta device,不占显存;
- `materialize(storage)` 要求传入一段连续 uint8 CUDA 存储且字节数严格等于规划总长,随后逐个张量做 `narrow → view(dtype) → view(shape) → swap_tensors`;
- 规划一旦物化即冻结(`assert not materialized`),保证同组内各 rank 的规划顺序一致;
- 四个 buffer 全部接受 `BufferAllocator` 或裸字节数两种构造形式,`EPBuffer` 的 LB 区、`BucketBuffer` 的主存储都是典型使用者。

这个模式的工程价值:**对称内存的 2 MiB 对齐、多张量共存、以及"所有 rank 规划一致"的约束,都收敛到一处强制**,而不是散落在各 buffer 的构造代码里。

### 3.4 通信上下文(context)复用

四个 buffer 共享同一套 host 侧通信上下文机制(`csrc/comm/` + `csrc/kernels/comm/context.cpp`):

- `NCCLCommHandle`(`deep_ep/comm/handle.py`):包装 NCCL communicator,**可直接复用 PyTorch 进程组已有的 NCCL comm**(`get_nccl_comm_handle(group)`),避免重复初始化;
- 每个 buffer 持有一个或多个 context(对应一个或多个 process group),`BucketBuffer` 明确支持**一个 buffer 挂多个 group**,通过 `group` 参数在调用时选择;
- `barrier`、`get_comm_stream`、`get_physical/logical_domain_size` 作为通用函数暴露在 `deep_ep.comm`,各 buffer 以属性形式继承(见 `ep.py:237-240` 的类属性别名)。

---

## 4. 通信后端:全面迁移到 NCCL Gin

### 4.1 Gin 设备侧原语

V2.5 的 device 侧通信全部通过 **NCCL Gin**(NCCL 2.32+ 的轻量级 device-side communication API)完成。核心操作集(封装于 `comm::NCCLGin`,`deep_ep/include/deep_ep/comm/handle.cuh`):

| 原语 | 语义 | DeepEP 中的典型用途 |
|---|---|---|
| `gin.get` | RDMA 读(远端 → 本地),可跨不同 window、区分 segment(GPU/CPU 混合) | Engram fetch(§8) |
| `gin.put` / `gin.putValue` | RDMA 写(本地 → 远端),可携带远端动作 `remote_action` | PP send 的 `put + StrongVASignalAdd`(§9);bucket RS 的 `put + 远端 tail 自增` |
| `gin.signal` | 无数据的远端原子信号(滚动计数器) | 各 buffer 的 arrival/release 通知 |
| `gin.waitSignal` | 本地等待远端信号达到目标值(带 shadow 计数器与超时诊断) | 接收侧等待 |
| `gin.flush` / `flushAsync` | 保证已发请求的远端可见性(同步/异步,异步返回 request 句柄) | Engram 每层结束时的 QP 冲刷 |
| `getSignalShadowPtr` | 信号 shadow(本地镜像)指针 | 滚动计数等待 |

请求可带 `ncclGinOptFlagsAggregateRequests`(批量聚合,不立刻敲 doorbell)与 `ncclGinOptFlagsDefault`(立刻门铃)等选项,Engram 的 doorbell 阈值机制(§8.3)就建立在这对选项上。

**资源共享模式**:`NCCLGin` 构造时可选 `NCCL_GIN_RESOURCE_SHARING_GPU`(整个 GPU 共享该 QP)或 `NCCL_GIN_RESOURCE_SHARING_CTA`(CTA 独占),PP/Engram 的 QP 切分策略依赖它。

### 4.2 NCCLGin 封装:NVLink 快路径 + RDMA 慢路径

`NCCLGin` 不是对 Gin 的薄包装,它做了一件关键的事——**每次操作自动选择传输路径**:

```cpp
// handle.cuh:66-92  对称指针解析
dtype_t* get_sym_ptr(dtype_t* ptr, dst_rank_idx) {
    if (not is_nvlink_accessible<team_t>(dst_rank_idx)) return nullptr;  // 域外 → 走 RDMA
    if (dst_nvl_rank_idx == team_lsa.rank) return ptr;                   // 本地 bypass
    return ncclGetLsaPointer(nccl_window, get_sym_offset(ptr), dst_nvl_rank_idx);  // NVLink 直访
}
```

上层原语据此分流,例如 `red_add_rel`(远程原子加):NVLink 可达时用 `red.add.rel.sys`(跨 NVLink 的 PTX 原子),不可达时退化为 `gin.signal(VASignalAdd)` 的远端原子;`put_value` 同理(`st.relaxed.sys` vs `gin.putValue`)。这使得**同一份内核代码同时覆盖 NVLink 域内、域外与混合拓扑**,无需像 V1 那样为 intranode/internode 各写一套。

NVLink 可达性判定按 team 分派:LSA team 恒可达;World team 用 rail×lsa 区间判定;Rail team 仅本 rank 可达(`handle.cuh:37-54`)。

### 4.3 GPU barrier

`comm::gpu_barrier`(impls/comm/barrier.cuh + layout/common/barrier.cuh)是各内核的入口/出口栅栏:基于对称内存中的 `BarrierSignals`,支持超时(`kNumTimeoutCycles`,默认由构造参数 `num_gpu_timeout_secs` 换算)、带 tag 防跨内核干扰,并可选择是否冲刷全部已分配 QP(`kFlushAllAllocatedQPs`)。

两个有代表性的修复进入了本区间:

- `#715`:当 scale-up 域跨 NVLink 与 RDMA 时,GIN barrier 前需要一次 **system-scope release**,否则 NVLink 侧与 RDMA 侧的可见性顺序不成立;
- `#642`:mbarrier 等待与 TMA load 之间插入 `fence.proxy.async.shared::cta`,保证共享内存的异步代理可见性。

### 4.4 为什么可以删掉 NVSHMEM

V1 的跨节点通信(含 IBGDA)与 V2.0 的 legacy 路径都依赖 NVSHMEM;EPv2 自 4 月公开起就改用了 NCCL Gin(NCCL 内置的 GDAKI——GPU-initiated RDMA,即 IBGDA 类能力,见 `NCCL_GIN_GDAKI_ENABLE=1` 编译宏,`csrc/runtime/jit.hpp`)。到 V2.5:

1. 所有 device 侧 RDMA 操作(get/put/signal/flush)都有 Gin 对应原语;
2. QP 数量、QP depth、RD atomic 限制等调参项改由 NCCL 环境变量/构造参数承载(如 `NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC`);
3. 于是 NVSHMEM 相关的构建探测(`nvidia-nvshmem` pip 包,`setup.py`)、`csrc/kernels/backend/nvshmem.cu`、`NVSHMEM_*` 环境变量体系全部删除,**NVSHMEM 不再是依赖**(README News 与 `setup.py` 均无 NVSHMEM 痕迹)。

---

## 5. DeepJIT 运行时编译与构建体系重构

### 5.1 拆分:install 时编 host,运行时编 kernel

| | V2.0 | V2.5 |
|---|---|---|
| GPU 内核编译时机 | 构建期(自带 `csrc/jit/` JIT 在首次调用时编译,但 host 扩展本身在 install 时用 torch cpp_extension 全量编译 CUDA 源) | **install 时完全不编 GPU 内核**;运行时由 DeepJIT 编译 |
| JIT 实现 | 自带(`csrc/jit/{compiler,cache,kernel_runtime,launch_runtime,...}`,786 行、8 个头文件) | 独立库 **DeepJIT**(`third-party/deep_jit` submodule → `deepseek-ai/DeepJIT`) |
| 安装前置 | 需要可见 GPU + `TORCH_CUDA_ARCH_LIST` | **不需要 GPU、不需要 arch list**;只需 CUDA toolkit(≥13.1)与 C++20 编译器可用 |
| host 扩展源文件 | 多个 .cu/.cpp | 仅 3 个:`python_api.cpp`、`kernels/comm/context.cpp`、`kernels/driver/driver.cpp` |

### 5.2 DeepJIT 初始化细节

`csrc/runtime/jit.hpp` 展示了集成方式:

- `init_jit(library_root, nccl_root)` 在 `deep_ep/__init__.py` 导入时调用,构造 `deep_jit::Runtime<deep_jit::CUDA>`;
- 配置:库根目录、前缀 `"EP"`、额外 include(`deep_ep/include`)、源码前缀 `"deep_ep/"`(用于缓存 key 的内容追踪);
- nvcc 参数:`--diag-suppress=...`、`-I<nccl>/include`(**NCCL 头路径参与缓存 key**,而 DeepEP 自身 include 靠内容追踪做到路径无关)、`-DEP_NUM_TOPK_IDX_BITS`、以及一组 Gin 开关宏:`NCCL_GIN_GDAKI_ENABLE=1`、`NCCL_GIN_PROXY_ENABLE=0`、`NCCL_GIN_GPI_ENABLE=0`、`NCCL_GIN_EFA_GDA_ENABLE=0`;
- `deep_ep.__init__` 导入时还执行 `check_nccl_so()`:扫描 `/proc/self/maps`,断言进程内只有一份 `libnccl` 且与 DeepEP 链接的二进制**逐字节一致**(`filecmp`),防止 PyTorch 加载了另一份 NCCL。

### 5.3 部署含义

- **一次构建、多机分发**:wheel 不含任何目标架构的 GPU 内核,在 H100/H800/H200/B200 上运行时各自 JIT;
- **缓存**:JIT 产物缓存在 `EP_JIT_CACHE_DIR`(可持久化到共享存储),集群内只有首台机器付编译成本;
- 代价:首次调用每个"模板实例"有一次编译延迟;所有内核的模板参数(§6-§9 中大量 `kNumXxx`)因此保持编译期常量以最大化缓存命中。

---

## 6. EPBuffer:MoE 通信内核详解

### 6.1 张量内存布局(TokenLayout / BufferLayout)

`deep_ep/include/deep_ep/layout/ep/token.cuh` 定义每个 token 在通信 buffer 中的紧凑布局,三段各按 32 B 对齐、总和按 64 B 对齐,末尾可选挂一个 mbarrier:

```
[ hidden (BF16 或 FP8) ][ scale factors ][ metadata ] [ mbarrier? ]
                                    metadata =
                                    topk_idx      : int32 × num_topk   (恒为 32 位)
                                    topk_weights  : float × num_topk
                                    src_token_global_idx : int (可选)
                                    linked_list_idx      : int × num_topk (可选, hybrid 模式)
```

`BufferLayout` 在其上组织 `num_ranks × num_max_tokens_per_rank` 个 token 槽,并提供 rank 视图与 channel 视图(`get_channel_buffer`,hybrid 模式下每个 channel 一段)。

### 6.2 信号工作区(EPSignals)

`layout/ep/workspace.cuh` 的 `EPSignals`(≤16 MiB)是所有 EP 内核的同步原语集合:

| 字段 | 用途 |
|---|---|
| `barrier_signals` | 入口/出口 GPU barrier |
| `notify_reduction[ranks+experts]` | 接收侧计数归约 |
| `scaleup_rank_expert_count[2][...]`、`scaleout_rank_expert_count[2][...]` | 双缓冲(send/recv)的 rank 级/expert 级 token 计数 |
| `scaleup_atomic_sender_count` | NVLink 侧原子发送者计数 |
| `scaleout_channel_signaled_tail[1280][ranks]`、`channel_scaleup_tail[1280][ranks]` | channel 级进度(最大 1280 channel) |
| `lb_chunk_idx_counter` | LB kernel 的 chunk 抢占计数器(§6.8) |

容量上限:总 expert ≤ 2048、每 rank expert ≤ 512。

### 6.3 dispatch 与 combine

`EPBuffer.dispatch(x, topk_idx, ...)` / `combine(x, handle, ...)` 的语义与 V2.0 的 elastic 版本一致(高吞吐与低延迟统一接口、FP8 dispatch + BF16 combine、cached handle 复用布局),V2.5 的增量参数:

| 参数 | 说明 |
|---|---|
| `handle`(cached) | 反向传播/二次调用时跳过布局重算;expand 模式下缓存的 expanded layout 也可复用 |
| `do_expand` / `do_zero_padding` | expanded layout(每 expert-slot 一行,面向 grouped GEMM);`do_zero_padding` 将 expert 间对齐填充槽清零,保证 GEMM 数值确定 |
| `use_tma_aligned_col_major_sf` | FP8 scale factor 用 TMA 对齐的列主序布局 |
| `defer_epilogue` | **延迟 epilogue**:CPU 侧接收计数等待与 copy/reduce epilogue 推迟到 `event.current_stream_wait()` 时执行(要求 `async_with_compute_stream=True`),让通信主 kernel 与计算重叠得更彻底;函数此时直接返回 `EventOverlap`,其 `wait()` 的返回值就是原本的五元组/三元组 |
| `deterministic` | 确定性模式:`EPHandle.deterministic_sort` 在 wait 之后按"源 rank + 源 token 全局序"重排接收 token(expand 模式为双键排序:先 expert、后 expert 内源序),消除 RDMA/NVLink 到达顺序带来的非确定性;`check_torch_deterministic()` 要求 PyTorch 确定性环境 |

`EPHandle`(ep.py:27-205)是 dispatch 与 combine 之间的路由凭证,字段包括:每 scale-up rank 的接收前缀和 `psum_num_recv_tokens_per_scaleup_rank`、按 `expert_alignment` 对齐的 expert 前缀和 `psum_num_recv_tokens_per_expert`、`recv_src_metadata`(源 token 与槽位)、`dst_buffer_slot_idx`、hybrid 模式专属的 `token_metadata_at_forward` / `channel_linked_list` 等;handle 对 `topk_idx` 做**引用保存 + 版本号校验**(`tensor._version`),防止使用中偷改。

### 6.4 分析式 SM/QP 估算(免 autotune)

V2.5 延续了 EPv2 的"解析估算代替自动调参"路线,公式可读(`ep.py:431-560`):

**SM 数 `get_theoretical_num_sms`**:以"平衡 gate"假设计算每 token 的期望 top-k 命中数(组合数公式 `get_expected_topk`),然后按 dispatch/combine 两个方向分别累加 SM 读量(`sm_read`)、SM 写量(`sm_write`)、RDMA 流量、NVLink 流量;取 RDMA/NVLink 中**带宽受限的一侧**(`bounded_traffic/bounded_gbs`),解出满足 `SM 带宽 ≥ 链路带宽 × 流量比` 的最小 SM 数,乘 1.3 余量、向上取 4 的倍数、至少 4;若 `prefer_overlap_with_compute=False` 则再取 `max(64)`,最后不超过设备 SM 数的一半。

**QP 数 `get_theoretical_num_qps`**:direct 模式 `min(num_sms, 9)`(减少 doorbell 开销);hybrid 模式 `num_sms × 16 + 1`(每 channel 独立 QP + 1 个 notify QP);均不超过构造时的 `num_allocated_qps`(默认 hybrid 65 或 129、direct 17,取决于 NIC 是否支持 fast RDMA atomic)。

### 6.5 动态冗余专家负载均衡(LB)——V2.5 最重要的新原语

**动机**:热门专家在线复制(MoonEP、UltraEP 路线)把过载 expert 的副本放到 NVLink 域内其他 GPU 上计算。这要求两个通信原语:前向计算前把**原始专家权重(+ 量化 scale)推给持有副本的邻居**;反向时把**副本的 FP32 梯度累加回原始专家**。V2.5 在 `EPBuffer` 上直接提供:

```python
weights_ready = buffer.lb_prefetch_weights(redundant_expert_weights,  # [num_redundant, *shape] 在 LB 区
                                            expert_weights,           # [num_local, *shape] 普通 CUDA
                                            redundancy_mapping)       # [num_nvlink_ranks, num_redundant] int32
weights_ready.wait()          # 之后可安全读取冗余槽位
...                            # 专家计算
grads_ready = buffer.lb_reduce_grads(redundant_expert_grads,          # [num_redundant, hidden] fp32, LB 区
                                     expert_grads,                    # [num_local, hidden] fp32, 原地累加
                                     redundancy_mapping)
```

约定:两个操作都是 LSA(NVLink)域内**集合通信**(无冗余槽位的 rank 也必须调用);`redundancy_mapping[r, c]` 给出"域内 rank r 的冗余槽 c 承载的 expert 全局 id"(`owner_rank × num_local_experts + local`),`-1` 表示空槽;副本只能指派给**邻居**,自己的 expert 不得出现在自己的冗余槽;权重/梯度张量须连续,每 expert 字节数为 32 B 的倍数,允许成对传多个张量(权重 + scale 一起搬,`kNumMaxWeightEntries = 16`,见 `layout/ep/eplb.cuh:69-87`)。LB 存储独立于 EP 主 buffer,用 `BufferAllocator` 规划进构造参数的 `lb_allocation_plan_or_num_bytes`。

**内核实现**(`impls/ep/prefetch_weights.cuh` / `reduce_grads.cuh`,各 95 行,结构同构):

1. **任务构建**(host 无关,纯 device):`EPLBSharedMemoryLayout::build_tasks` 把整张 `redundancy_mapping` 用 `cub::BlockScan` 做一次前缀和,挑出"本 rank 是 owner"的交换任务,压缩进共享内存的 `EPLBTask{local_expert_idx, dst_nvl_rank_idx, redundant_expert_idx}` 数组(32 KB 共享内存,复用 TMA staging 区做 BlockScan 临时存储);
2. **工作分配**:`EPWeightIterator` 用工作区里的 `lb_chunk_idx_counter` 做 `atomicAdd` 抢占式领取 32 KB 级 chunk,按 (weight → task → chunk) 单调推进,各 warp 负载均衡;
3. **数据移动**:
   - prefetch:`tma_load_1d`(HBM 源权重 → smem)→ mbarrier → `tma_store_1d`(smem → 邻居的冗余槽,**经 `gin.get_sym_ptr<ncclTeamTagLsa>` 解析的 NVLink 对称指针**)→ `tma_store_wait_read<0>`;
   - reduce_grads:`tma_load_1d`(邻居冗余梯度 → smem)→ **`tma_store_reduce_add_f32`**(TMA 硬件 reduce-add 直接累加进本 rank 的 expert 梯度),全程 SM 只做调度,搬运与加和都由 TMA 引擎完成;
4. **栅栏**:入口 `gpu_barrier`(等所有 rank 任务就绪)+ 出口 `gpu_barrier`(保证邻居侧可见性),配合调用方 `EventOverlap.wait()`。

SM 数估算 `lb_get_theoretical_num_sms` 只依赖带宽:`ceil(max(nvlink_gbs/sm_read_gbs, nvlink_gbs/sm_write_gbs))`,与冗余映射无关(注释说明:环流量或 rank 偏斜可能先于 SM 容量撞上传输瓶颈)。

### 6.6 其他行为约束

- EP dispatch/combine **必须消耗 SM**(zero-SM RDMA EP 不支持,README Notes);
- 所有 rank 的 `num_max_tokens_per_rank` 必须一致(构造时给定);
- `sl_idx` 可覆盖 `EP_DEFAULT_RDMA_SL`,再被 `EP_OVERRIDE_RDMA_SL` 覆盖;
- CPU/GPU 双侧超时(`num_cpu_timeout_secs` 默认 300 s、`num_gpu_timeout_secs` 默认 100 s)贯穿所有内核的 `timeout_while` 等待。

---

## 7. BucketBuffer:CP/DP 集合通信详解

### 7.1 定位与切块模型

`BucketBuffer` 面向训练侧的**批量、张量序列级**集合通信(典型:CP/DP 的梯度/参数同步),与 NCCL 集合通信的差异在于:GPU 主导、可预测带宽模型、支持 FP32 通信精度与就地/非就地语义。

核心抽象是 **chunk**(`layout/bucket/chunk.cuh`):

- 全部 bucket(一次调用里的张量序列)被展平成长度固定的 **32 KB chunk 序列**(`ChunkIterator`,桶间连续编号,仅每桶尾 chunk 可能不满);
- 接收侧固定一块 **256 MiB `ChunkStorage`**,以 `ChunkView2D<ranks, slots_per_rank, 32KB>` 切成"每 rank 若干环形槽位"——这是 credit-ring 协议的物理载体;
- 每个调用者(warp)以 stride 方式认领 chunk(`start + i×stride`),天然均匀。

### 7.2 三个算法 × 三种传输

| 算子 | nvlink 传输 | rdma 传输 | hybrid 传输 |
|---|---|---|---|
| `all_gather` | 对称指针直读 + **copy engine** 驱动(`num_sms=0`,不占 SM) | **JIT 编译 `rdma_all_gather` 内核并 cooperative launch**(grid = 全部设备 SM) | **每 RDMA peer 1 SM 发射 RDMA**(`kNumQPsPerChunk` 个 QP)+ 拷贝引擎经 NVLink 转发到达的 chunk |
| `reduce_scatter` | warp 读邻居槽 + 本地归约 | **credit-ring 双角色 warp**(§7.3) | 两级:NVLink 域内归约 + 跨节点 ring |
| `all_reduce` | multimem(节点内硬件归约) | 本地归约 + 点对点 | **multimem + 跨节点 ring + 广播**三阶段(§7.4) |

每个算子 × 拓扑有独立的 SM 数解析公式(`bucket.py:236-293`),例如 hybrid reduce-scatter 取 `min(NVLink 带宽/nvl_ranks, RDMA 带宽 × n/(n-1))` 为瓶颈带宽,再按"multimem 与本地归约各需一读一写"放大 2 倍反推 SM;公式中的 ring 算法流量系数 `n/(n-1)` 即经典 ring 的带宽利用率。

### 7.3 RDMA reduce-scatter 的 credit-ring 协议(最复杂的新内核)

`impls/bucket/reduce_scatter/rdma.cuh`(637 行)是一个**warp 特化**内核,把 warp 分为两种角色并显式分配寄存器预算(`warpgroup_reg_realloc`:issue warp 112 寄存器、reduce warp 144 寄存器):

**Issue 角色(发送侧)**:
- 每个 issue warp 固定映射到 QP(4 个 lane 条带化覆盖 4 个 send QP);
- 按 chunk 迭代:目标槽位 `(dst_rank, slot_idx = chunk_idx % kNumSlotsPerRank)`;
- 等待远端该槽的 **head 信用**(`ld_volatile(head_ptr) >= head_target`)——远端 reduce 完才会写回 head;
- `gin.put<ncclTeamTagRail>` 把 chunk 推到远端 `ChunkStorage` 槽位,并携带 `StrongVASignalAdd{tail_ptr, +1}` 远端自增 tail(数据 + 通知一次完成);
- 8 个 peer 一批(`kNumIssueChunksPerBatch=8`)lane 并行发射,`__ballot`/`__syncwarp` 同步。

**Reduce 角色(接收侧)**:
- 按同样的 chunk 迭代,在**逆环形序**(`rank - peer_offset - j`)等待各 peer 的 tail 到达(`ld_acquire_sys(tail_ptr) >= tail_target`);
- 到达后把"本地 buffer 中自己的分片 + 各 peer 推来的 chunk 槽"在寄存器里做 `float4` 向量化累加(每 lane 8 个 float4 深展开),末尾乘以 `scale`(仅最后一批);
- 结果写回本地 buffer 的对应分片;
- **归还信用**:若该槽还有后续 chunk 要来,`gin.put_value(head_ptr, tail_target, release_rank)` 通知远端可以重用该槽(最后一次使用无需归还)。

协议性质:每 (peer, slot) 一个 in-flight 窗口的 **有界环形缓冲**,head/tail 都是滚动计数器;`kNumSlotsPerRank` 由 256 MiB 存储与 chunk 数推出,信号区上限有 `EP_STATIC_ASSERT` 兜底。入口/出口各一次 `gpu_barrier`(出口带全 QP 冲刷)。

### 7.4 Hybrid all-reduce 与 multimem

`impls/bucket/all_reduce/hybrid.cuh`(340 行)把节点内归约交给 **multimem**(`ptx::multimem_cp_async_bulk`,即 NVLink 域的多播/归约硬件能力),跨节点走 RDMA ring,warp 角色分四类:multimem warp(节点内归约)、issue warp(跨节点发射)、reduce warp(远端 chunk 归约)、broadcast warp(结果经 TMA 双缓冲流水线 + multimem 播回本域,见 `hybrid_ar_broadcast`)。`buffer_multimem` 是一块额外的 multimem 对称视图。

### 7.5 BucketSession:外部张量支持

`BucketSession`(`with buffer.session():`)临时接管三个集合方法,使其接受**未注册张量**:

- 进入 session 后,调用自动把 src/dst 登记/分配到 buffer 存储内(`_allocate` 按 64 B 对齐顺序切块),外部张量先 `_foreach_copy_` 进存储,内核在注册区执行,`wait()` 时再经 hook 拷回原 dst(就地调用则直接在注册区出结果);
- 约束:不支持嵌套 session、一个 session 内只能用同一个 group;`all_gather` 在纯 NVLink 域(`num_rdma_ranks == 1`)时允许外部 src 免拷贝(拷贝引擎直接读);
- 该机制用 `inspect.signature.bind` 解析调用参数,是典型的"注册制存储 + 透明暂存"设计。

---

## 8. EngramBuffer:远程记忆访问详解

### 8.1 存储模型:多层、GPU/CPU 可混布

V2.0 的 Engram 是单层 GPU→GPU 的 RDMA 读取;V2.5(`csrc/buffers/engram.hpp` + `impls/engram/`)升级为:

- **多层**:`set_config(num_entries_per_layer: List[int], hidden, ...)` 按层配置 entry 数,布局模板参数 `kNumLayers`;
- **存储介质可选**:`use_cpu_rdma_storage` 允许每 rank 的 storage 分片放在 **CPU 内存**并注册 RDMA(`gin.get` 的 `ncclGin_SegmentMixed` 段类型支持 GPU+CPU 混合寻址)——Engram 语义(大表、稀疏远端读)天然适合 CPU 内存承载;
- 尺寸规划:`get_storage_size_hint` 返回"主 GPU buffer(接收区 + 本 rank 的 FP8 SF 分片)+ 每 rank RDMA storage"两块对齐大小;构造时若总注册量超过显存,自动设置 `NCCL_WIN_STRIDE`(以 4 GiB 对齐);
- hybrid 模式下 CPU storage 通过**跨进程共享句柄**(`create_shared_handle` 返回 `(pid, fd)` + all_gather_object)在同节点各进程间复用,避免重复注册;
- QP 配置有解析公式 `get_theoretical_config(num_layers, num_max_tokens, num_entries_per_token)` → `(num_qps, qp_depth, restrict_rd_atomic)`,以 NIC 读请求率(Mpps,fast RDMA atomic 90 / 否则 45)为输入。

### 8.2 fetch 内核:每层一个 wait hook

`fetch(indices: [L, T, E], num_qps)` 一次为**所有层**发射 RDMA get,返回 L 个 wait hook(每层一个 callable,`engram_fetch_wait.cuh`),调用方可按层流水线地消费——这是"多层 + 每层独立同步"的直接体现。

**发射侧结构**(`impls/engram/engram_fetch.cuh:53-207`):

- warp 划分为 issue warp 与 SF warp 两组;issue warp 通过 `EngramIssueWarpLayout` 映射到 `(rdma_peer, gin_context/QP)`:warp 先按 peer 分、再按 QP 分,同一 peer 的多 warp 再按 token tile 条带;
- 每个 issue warp 持有一个 **2×32 的环形 pending-request 缓冲**(共享内存),lane 级把 `(global_idx << 32) | entry_idx` 压入,满 32 个即成批;
- **doorbell 阈值**:`doorbell_threshold = kFlushDepth / 活跃 warp 数`,未满阈值的批次带 `ncclGinOptFlagsAggregateRequests`(聚合、不敲铃),末批或越阈才用默认选项敲 doorbell——这是把"请求聚合降 NIC 开销"与"及时可见"折中的核心;`#752`("last get 必须敲 doorbell")修复的正是末批漏敲导致 flush 不完整的时序;
- **每层结束**:单 warp 独占的 peer-context 直接 `flush_async`;多 warp 的则用 workspace 里的 `issue_counter` 做原子到达计数,最后一个 warp `fence_acquire_gpu` 后统一 flush 并把计数器归零,其余 warp 带超时地等待计数器清零(层间依赖由此建立);
- SF warp(仅 FP8):不经 RDMA,而是经 **NVLink 对称指针**(`ncclTeamTagLsa`)把 SF 分片行 `__ldg` 读入、按 TMA 对齐的列主序散射到 `fetched_sf`(`copy_sf_entry`)——SF 表小、命中域内,走 NVLink 最省。

### 8.3 收尾提交:移除 warpget 优化

V2.5 的最后一个提交 `8c1d13a` 删除了发射批次中 `ncclGinOptFlagsWarpGet` 的使用(整 warp 单次 get 的优化,依赖扩展版 NCCL 构建)。移除后 Engram 只依赖标准 Gin 接口。注:README Notes 中"Engram requires a NCCL build providing `ncclGinOptFlagsWarpGet`"的表述在该提交后已过时,属文档残留。

---

## 9. PPBuffer:流水线并行通信详解

### 9.1 模型:ring 相邻 rank 的有界槽位

`PPBuffer(group, num_max_tensor_bytes, num_max_inflight_tensors)` 为每对相邻 rank 维护 `num_max_inflight_tensors` 个固定大小槽位(`layout/pp/workspace.cuh` 的 `PPSignals` 记录 `send_count / recv_count / arrival[rank][qp] / release[rank]`)。`send(x, dst)` / `recv(x, src)` 只允许 **ring 上的前驱/后继**(`get_pp_buffer_offset` 返回本 rank 与 peer 的槽位半边索引)。

### 9.2 发送/接收协议

`impls/pp/pp_send_recv.cuh`:

**send**(`pp_send_impl`,单 warp/SM):
1. 等目标槽的 **release 信用**(`ld_acquire_sys(release) >= send_count - in_flight + 1`,超时打印诊断);
2. `pp_tma_copy` 把 `x` 经 TMA 搬到本 rank 的 send 槽(§9.3);
3. `cooperative_groups::this_grid().sync()`(整个 grid 的 TMA 完成后);
4. 把发送数据切到 ≤2 个 QP,逐 QP `gin.put` 到 peer 的 recv 槽并携带 `StrongVASignalAdd{arrival, +1}`;**空数据也必须 signal**(接收方每个 QP 都等);
5. SM0 递增 `send_count`。

**recv**(`pp_recv_impl`):
1. 前 `kNumQPs` 个 thread 各等一个 QP 的 arrival 达到 `recv_count + 1`;
2. `pp_tma_copy` 把 recv 槽内容搬出到 `x`;grid sync;
3. `gin.signal(release, +1)` 归还槽位给 peer,递增 `recv_count`。

这是一个干净的**生产者-消费者环形队列**:arrival 是数据就绪信号,release 是槽位归还信号,两者都是滚动计数器;`num_max_inflight_tensors` 即队列深度。

### 9.3 pp_tma_copy:两级 TMA 流水线

`pp_tma_copy`(同文件 34-102 行)是典型的 double-buffered TMA 拷贝:smem 分 `kNumStages=2` 级,每级一个 mbarrier;SM 间按 32 B 块条带分工;稳态下"等本级 load → 发 store → 等 store 读完成 → 发下一轮 load",实现 load/store 全重叠。该模式同时出现在 PP 与 LB kernel 中,是 V2.5 内核的通用数据搬运构件。

---

## 10. 公开 API 面变化(v2.0 → v2.5)

**导入面对比**(`deep_ep/__init__.py`):

```python
# V2.0
from deep_ep import ElasticBuffer, EventOverlap, EventHandle, topk_idx_t, get_physical_domain_size, ...

# V2.5
from deep_ep import (
    EPBuffer, EPHandle,
    BucketBuffer, BucketSession,
    EngramBuffer,
    PPBuffer,
    BufferBase, BufferAllocator,
    EventOverlap, EventHandle, topk_idx_t,
    get_num_allocation_alignment, get_num_rdma_alignment, get_num_tma_alignment,
    comm,  # NCCLCommHandle / barrier / get_comm_stream / 域查询
)
```

| 变化 | 说明 |
|---|---|
| 移除 | `ElasticBuffer`(全部方法迁移到四个新 Buffer)、V1 的 `Buffer`/`Config`(V2.0 中尚以 legacy 形式存在) |
| 新增 | `EPHandle`(handle 从裸 tuple 升级为带文档与版本校验的类)、`BucketSession`、`BufferAllocator`、对齐查询三件套、`EPBuffer.lb_*` 两原语、dispatch/combine 的 `defer_epilogue`/`do_zero_padding`/`use_tma_aligned_col_major_sf` 参数 |
| 保持 | `EventOverlap` 的 `current_stream_wait()` + hook 语义(V2.5 的 deferred epilogue 与确定性排序都挂在 `hook_after_wait` 上) |

---

## 11. 构建与部署需求变化

| 项 | V2.0 | V2.5 |
|---|---|---|
| GPU | Ampere(Hopper 推荐) | **仅 Hopper 及以后**(SM90 特性不再可选,`DISABLE_SM90_FEATURES` 删除) |
| CUDA toolkit | 12.x | **13.1+**(DeepJIT 编译需要,安装时可用、运行机需保留) |
| 编译器 | C++17 | **C++20**(需 `std::format`) |
| PyTorch | 2.10+ | **2.10+**(未变化) |
| NCCL | 2.30.4+ | **2.32.3+**(Gin 完整特性;`ncclDevCommCreate` 运行时版本兼容,`#688`) |
| NVSHMEM | 跨节点必需 | **彻底移除** |
| 网络 | NVLink + RDMA | NVLink + RDMA;`BucketBuffer` 要求 NVLink 与 RDMA 带宽**都能被探测到**(`nvidia-smi`/`ibstat`),即便只用单传输 |
| 安装 | 需 GPU + arch list | **免 GPU**;`git submodule update --init`(DeepJIT)+ `bash install.sh` |
| NCCL 一致性 | 无强制 | 导入时二进制级校验 `libnccl` 唯一且一致(`check_nccl_so`) |

关键环境变量:`EP_JIT_CACHE_DIR`(JIT 缓存)、`EP_NCCL_ROOT_DIR`(自定义 NCCL)、`EP_NUM_TOPK_IDX_BITS`、`EP_DEFAULT_RDMA_SL` / `EP_OVERRIDE_RDMA_SL`、`NCCL_GIN_TYPE` / `NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC`(Engram)、`NCCL_WIN_STRIDE`(超显存注册)。

---

## 12. 被移除的功能与迁移说明

1. **V1 全部删除**:`Buffer`/`Config`/`EventOverlap(V1 语义)` API、`kernels/legacy/` 的 intranode/internode/low-latency 内核(约 7,000 行)、`csrc/legacy/`、`docs/legacy.md` 对应的旧文档。仍在用 V1 的用户需升级到 EPv2 语义:`Buffer.get_dispatch_layout + dispatch` → `EPBuffer.dispatch`(布局计算内置,handle 化);low-latency 路径并入 `EPBuffer` 的统一接口(低延迟 = 小 batch 下的同一组内核)。
2. **NVSHMEM 后端删除**:所有 `NVSHMEM_*` 环境变量、`csrc/kernels/legacy/ibgda_device.cuh`、`csrc/kernels/backend/nvshmem.cu`、`docs/nvshmem.md` 消失;等价能力由 `NCCL_GIN_GDAKI_*` 承担。
3. **自带 JIT 删除**:`csrc/jit/` 整体移除,换 DeepJIT;依赖旧 JIT 缓存路径的用户改 `EP_JIT_CACHE_DIR`。
4. **Ampere 支持删除**:FP8/SM90 路径成为硬依赖。
5. **zero-SM EP 不可用**:EP 通信必须给 SM(Bucket 的 all_gather 除外——且仅纯 NVLink 拓扑走拷贝引擎,RDMA/hybrid 拓扑仍需 SM 发射内核)。

---

## 13. 测试体系

V2.5 的测试目录按 buffer 重组(`tests/`):

| 目录 | 内容 |
|---|---|
| `tests/ep/` | `test_ep.py`(dispatch/combine 正确性 + 带宽,含 hybrid/direct、expand、cached handle)、`test_prefetch_weights.py`(250 行,覆盖多张量权重 + scale、映射边界)、`test_reduce_grads.py`(256 行,FP32 累加语义) |
| `tests/bucket/` | `test_all_gather.py` / `test_reduce_scatter.py` / `test_all_reduce.py` / `test_mixed.py`(373 行,混合张量序列 + session)/ `test_session.py` / `test_perf.py`(273 行,带宽基准) |
| `tests/engram/` | `test_engram.py`(155 行,多层、GPU/CPU storage) |
| `tests/pp/` | `test_pp.py`(ring send/recv,要求纯 RDMA group) |
| `tests/comm/`、`tests/buffer/` | barrier 语义、分配规划 |
| `tests/legacy/` | V1 回归(随 V1 代码保留于 fork 的 main,供对照) |
| `tests/utils/` | `test_gate.py`(参考 gate 实现) |

测试约定:`init_dist` 基于 `MASTER_ADDR/PORT + WORLD_SIZE/RANK`,单机多卡 spawn 或多机同脚本;默认 `NCCL_IB_SL=1`、`EP_OVERRIDE_RDMA_SL=1`;数值验证用 `deep_ep/utils/refs.py` 的参考实现比对。

---

## 14. 总结:设计哲学与功能清单

### 14.1 设计哲学

1. **一个域模型,四种负载**:World/LSA/Rail 三级 team + scale-up/scale-out 两级域,让同一套内核模板覆盖 EP、集合通信、PP、Engram 四类负载,消灭了 V1"按拓扑写 N 套内核"的乘法复杂度。
2. **对称内存 + 自动路径选择**:`NCCLGin` 在每个操作点做 NVLink 对称指针 vs RDMA 的分流,拓扑感知下沉到通信原语层。
3. **TMA 优先的数据移动**:LB 的权重搬运、PP 的槽位拷贝、bucket 的广播,数据面几乎全部由 TMA 引擎(load/store/reduce-add)执行,SM 只承担调度与等待——这是"通信尽量不占算力"路线的延续。
4. **warp 特化 + 显式资源预算**:复杂内核(RDMA reduce-scatter、Engram fetch)用 `warpgroup_reg_realloc` 给不同角色 warp 分配不同寄存器预算,角色间用共享内存环形缓冲 + mbarrier 交接。
5. **解析调参取代自动调优**:SM/QP/QP-depth/槽位深度全部由带宽/速率模型闭式计算(公式在 Python 层可读、可覆盖),免掉 V1 时代依赖查表 + 集群内 autotune 的黑盒。
6. **构建与部署解耦**:host 扩展 3 文件 + DeepJIT 运行时编译,安装免 GPU,缓存集群共享。

### 14.2 功能清单

| 类别 | 能力 | 状态 |
|---|---|---|
| EP | dispatch/combine(BF16/FP8、expand layout、zero padding、cached handle、deferred epilogue、确定性模式) | 稳定(核心) |
| EP | 动态冗余专家 LB(`lb_prefetch_weights` / `lb_reduce_grads`,NVLink,TMA reduce-add) | 稳定(核心),支撑 MoonEP/UltraEP 类方案 |
| 集合通信 | 批量 all-gather / reduce-scatter / all-reduce,nvlink/rdma/hybrid 三传输 + 外部张量 session | 实验性 |
| Engram | 多层、GPU/CPU 混合存储、每层 wait hook、doorbell 聚合 | 实验性 |
| PP | ring 相邻 send/recv、in-flight 槽位、双级 TMA 流水线 | 实验性 |
| 基础设施 | 四 Buffer 统一生命周期、BufferAllocator、NCCL comm 复用、分析式 SM/QP 估算、二进制级 NCCL 一致性校验、CPU/GPU 双侧超时诊断 | 稳定 |

### 14.3 遗留问题与观察

- `Bucket/Engram/PP` 在 README 中仍标注 **experimental**,生产使用应以 EPBuffer 为准;
- README Notes 关于 Engram 依赖 `ncclGinOptFlagsWarpGet` NCCL 构建的说明,在 `8c1d13a` 之后已过时;
- `EPBuffer.get_theoretical_num_sms` 的 docstring 明确:group-limited gate(如 V3.0)的负载分布不满足"平衡 gate"假设,估算函数不适配(TODO 保留);
- LB 梯度归约内核(`impls/ep/reduce_grads.cuh:77-78`)中 `tma_load_1d` 带 `L2CacheHint::kEvictNormal` 处留有 `TODO: why this is faster?`,说明部分性能选择仍属经验性结论。

---

*文档基于 commit `8c1d13a` 的静态代码阅读;行号引用以该 commit 的 `deep_ep/` 与 `csrc/` 树为准。*
