# DeepEP main 分支(v2.1.0 / EPv2 ElasticBuffer)代码实现详解

> 本文基于 `s0okiym/DeepEP` main 分支(HEAD ≈ `63d80ff`,`deep_ep.__version__ == '2.1.0'`)的代码逐文件精读撰写,目标是把"这个仓库现在到底是怎么实现的"讲清楚:背景与动机、核心名词、分层架构、传输层、设备端公共层、V1 legacy kernel、EPv2 ElasticBuffer 数据通路、JIT 编译系统、工具层,最后汇总优化要点与测试体系。
>
> 姊妹篇:[deep_ep_v2_5_changes.md](./deep_ep_v2_5_changes.md) 分析的是后来出现的 **V2.5 分支线**(ElasticBuffer 拆分为 4 个 buffer、V1/NVSHMEM 移除、DeepJIT 取代内置 JIT)。本文分析的是 **main 分支的 V2.1.0 线**,两条线并存但代码不同,阅读时请注意区分:本文中的"EPv2"均指 main 分支的 `ElasticBuffer` 体系。

---

## 目录

1. [背景](#1-背景)
2. [名词速查表](#2-名词速查表)
3. [总体架构](#3-总体架构)
4. [传输层:backend](#4-传输层backend)
5. [设备端公共层:handle / comm / layout / ptx](#5-设备端公共层handle--comm--layout--ptx)
6. [V1 legacy kernel 体系](#6-v1-legacy-kernel-体系)
7. [EPv2:ElasticBuffer 数据通路](#7-epvelelasticbuffer-数据通路)
8. [JIT 编译系统](#8-jit-编译系统)
9. [工具层与构建](#9-工具层与构建)
10. [优化要点汇总](#10-优化要点汇总)
11. [测试体系](#11-测试体系)
12. [附录:关键文件速查](#12-附录关键文件速查)

---

## 1. 背景

### 1.1 MoE 与专家并行下的 All-to-All

DeepSeek-V3/R1 采用 MoE(混合专家)架构:每层有数百个 expert(如 256 个),每个 token 由 router 选出 top-k 个 expert(如 top-8)计算。在多卡/多机部署时使用 **EP(Expert Parallelism,专家并行)**:expert 分布在不同的 GPU 上,于是每一步都不可避免地出现两次全交换通信:

- **dispatch(分发)**:token 从自己所在的 rank 发到"被选中 expert 所在的 rank",并按 expert 组织好顺序;
- **combine(合并)**:expert 算完之后,token 的专家输出要回到原 rank,并对一个 token 命中的多个 expert 的输出按 router 权重加权求和。

这两次操作合起来就是一个 **all-to-all**(每对 rank 之间都可能有数据)。在大 EP 度(几百到几千 rank)下,all-to-all 的带宽/时延直接决定了 MoE 训练与推理的步时。DeepEP 的定位就是把这个 all-to-all 做到"占用尽可能少的 SM、跑满 NVLink 与 RDMA 带宽"。

DeepEP V1(1.x)基于 NVSHMEM IBGDA 手写 RDMA kernel,已经很强,但有两个结构性痛点:

1. **SM 占用固定且偏高**:V1 的 kernel 要求 24 个 SM 量级的固定开销,而 MoE 的 dense 计算(kernel)本身也希望独占 SM,通信 kernel 抢 SM 会拖累整体;
2. **规模与模式受限**:跨节点通信路径、QP 数量、SM 数量都是按固定布局硬编码的,难以适配 NVL72 这类"机内 NVLink 域 + 机间 RDMA"的两级拓扑。

DeepEP V2(本仓库 main 分支的 `ElasticBuffer`,v2.1.0)针对性重构:

- 通信 SM 数降到 **4~6 个**(README 中对比:24 → 4-6),且 SM/QP 数量在运行时按规模解析式计算,不再硬编码;
- 通信从"纯 RDMA"扩展为 **direct(全 RDMA)+ hybrid(NVLink 域内直连 + 域间 RDMA 两级流水线)** 两种模式,匹配 scale-up/scale-out 两级拓扑;
- 设备端通信原语从自研 NVSHMEM IBGDA 调用切换到 **NCCL 的 Gin(GPU-Initiated Network)设备 API**(GDAKI 后端),统一走 NCCL 的对称内存窗口管理;
- 数据搬运全面转向 **TMA(异步批量拷贝)+ mbarrier**,kernel 内部用 **warp 角色分化**(notify/dispatch/forward 等)+ **PDL(Programmatic Dependent Launch)** 串联主 kernel 与 epilogue kernel。

### 1.2 硬件前提

| 硬件/特性 | 在 DeepEP 中的角色 |
|---|---|
| Hopper(H800,sm_90) | 唯一正式支持目标(`TORCH_CUDA_ARCH_LIST` 默认 `9.0`);依赖 TMA、`red.add` 系统作用域原子、warp 寄存器重分配等 SM90 特性 |
| NVLink(机内 scale-up 域) | 同 NVLink 域 rank 间走 `red.add.rel.sys` / 直接 store 的对称指针,完全绕开 NIC |
| RDMA(IB/RoCE) | 跨 scale-up 域的数据搬运与计数同步 |
| GDAKI / IBGDA(NCCL Gin 后端) | **GPU 直接发起 RDMA**:kernel 内的线程自己写 doorbell,免 CPU 代理,是 V2 低 SM 占用的关键 |
| TMA | 隐藏向量(hidden)的大块 1D 异步 load/store,配合 mbarrier 做流水线 |
| PDL | 主 dispatch kernel 与 copy epilogue kernel 之间零间隙衔接 |

### 1.3 V1 → V2(main 线)的演进对照

| 维度 | V1(legacy,仍在 main 中) | V2.1.0(main 线,ElasticBuffer) |
|---|---|---|
| 后端 | NVSHMEM(IBGDA) | NCCL Gin(GDAKI)+ NVLink 对称内存;NVSHMEM 仅作 legacy 链接保留 |
| SM 占用 | 固定(典型 24) | 解析式计算,4~6 起 |
| 拓扑 | 单级 all-to-all | direct / hybrid(scale-up × scale-out 两级) |
| 数据搬运 | 寄存器/共享内存 + `st`/`red` | TMA 1D + mbarrier 流水线 |
| 内存 | NVSHMEM symmetric heap | NCCL window(严格顺序)+ 可选 GPU+CPU 混合缓冲 |
| 编译 | AOT(随 `.so` 预编译) | **JIT**:运行时按模板参数即时编译 cubin |
| 附加能力 | internode LL 低时延 kernel | Engram 显存拉取、PP send/recv、AGRS 集合通信(均可 0 SM/拷贝引擎) |

> 注意:main 分支是"双栈"结构——V1 legacy kernel 仍然以 AOT 方式编译进 `deep_ep._C`,V2 的全部 kernel 走 JIT。V2.5 分支线才把 V1/NVSHMEM 彻底移除。

---

## 2. 名词速查表

| 名词 | 含义 |
|---|---|
| **token / hidden** | 一个 MoE token 的特征向量;`hidden` 是其维度,`hidden_bytes` = hidden × 元素字节数 |
| **expert / topk** | 专家;每个 token 选 `topk` 个专家(如 8),`topk_idx` 记录所选专家号 |
| **dispatch / combine** | EP 的两次 all-to-all:token 送往专家所在 rank / 结果加权回传 |
| **scale-up 域(NVLink 域)** | 机内 NVLink 全互联的一组 rank(如 NVL72 的 72 卡);NCCL 中对应 **LSA 团队**(Local Scaleup Access,`ncclTeamLsa`) |
| **scale-out 域** | 跨机 RDMA 部分;多个 scale-up 域组成全集群 |
| **物理/逻辑域大小** | `get_physical_domain_size` = (world 中 RDMA 连接的 rail 数, NVLink 域大小);`get_logical_domain_size` 将其规整为规整的 `scaleout_ranks × scaleup_ranks` 矩形(允许 NVLink 域小于 rail 数时"折叠"),所有 kernel 按逻辑域计算 |
| **direct 模式** | 所有 rank 对直接走 RDMA(或 NVLink),单层 all-to-all |
| **hybrid 模式** | 两级流水线:域内 NVLink + 域间 RDMA。"scale-out warp"先做跨域搬运,"forward warp"再做域内分发,两级重叠 |
| **channel** | V2 kernel 里"一条 warp = 一个 channel",channel 之间互不依赖地搬运 token 子集;channel 总数 = SM 数 × 每 SM channel 数(≤ 8) |
| **QP** | RDMA 队列对。每个 QP 一条硬件队列;DeepEP 通过 `get_qp_mode` 决定 SM 与 QP 的分配/共享关系(整 QP 独占给某 SM,或全 SM 共享) |
| **Gin** | NCCL 的设备端网络原语:`ncclGin` 提供 `get/put/putValue/signal/waitSignal/flush` 等 GPU 直接发起的网络操作,后端为 **GDAKI**(GPU-initiated,IBGDA 的 NCCL 实现) |
| **GDAKI signal table** | GDAKI 后端维护的 GPU 可见信号计数器表,DeepEP barrier 直接轮询该表实现带超时的跨 rank 同步 |
| **对称内存 / window** | 每个 rank 在同一逻辑偏移处都有相同布局的内存块;NCCL 通过 `ncclCommWindowRegister` 注册 **window**,设备端用"偏移 + 目标 rank"即可得到对端指针,无需每地址一次注册 |
| **LSA 指针(`ncclGetLsaPointer`)** | 同 NVLink 域内,按 window 偏移取对端 GPU 的可直读写指针(NVLink P2P) |
| **slot** | 接收端缓冲中一个 token 的位置编号;dispatch 时按 expert 顺序原子分配 slot,combine 时原路寻址 |
| **expand layout** | 一种接收布局:按 expert 展开存储(每个 (token, expert) 占一行),供不需要"聚合回 token"的下游直接消费;与"聚合布局"相对 |
| **SF / scale factor / FP8** | dispatch 可把 token 以 FP8 发送(数据 + 缩放因子);`sf_pack_t` 打包 4 个 UE8M0(E8M0 格式的 scale)为一包,`KNumSFPacks` 为每 token 的 SF 包数 |
| **EPHandle** | 一次 dispatch 返回的句柄(`psum` 偏移、每 expert 计数、`recv_src_metadata`、`dst_buffer_slot_idx`、`token_metadata_at_forward`、`channel_linked_list` 等),供后续 combine 精确寻址,免去 combine 重新计算布局 |
| **PDL** | Programmatic Dependent Launch:前 kernel 用 `cudaTriggerProgrammaticLaunchCompletion()` 提前放行后继 kernel 的启动,后继 kernel 用 `cudaGridDependencySynchronize()` 等待真正需要的前置数据,消除 kernel 间隙 |
| **mbarrier** | SM90 的异步屏障;TMA load/store 的完成通知与 warp 间流水线同步都靠它 |
| **EventOverlap** | V1 时代的工具:让通信 kernel 与一个计算 event 区间重叠,用户可"通信 ∥ 计算" |
| **Engram** | 实验特性:RDMA 拉取远端"记忆条目"(embedding 式查表),可配置 CPU 存储,**0 SM** 完成(纯拷贝引擎/网络) |
| **PP** | 实验特性:Pipeline Parallelism 的点对点 send/recv 环(环形 slot + arrival/release 信号) |
| **AGRS** | 实验特性:All-Gather Reduce-Scatter 会话;NVLink 域内、基于拷贝引擎,0 SM |
| **SL(Service Level)** | RDMA 的流量等级(类似 InfiniBand SL),`sl_idx` 传给 NCCL Gin 作为 traffic class,实现 VL(虚拟通道)隔离 |
| **0 SM kernel** | 不需要常驻计算 SM 的传输:走 PCIe/拷贝引擎或纯网络引擎,不占 SM 资源 |
| **JIT / AOT** | 运行时即时编译(EPv2 kernel,按模板参数生成 cubin) / 安装期预编译(V1 legacy 与 backend) |
| **doorbell** | RDMA 的"按铃"寄存器:GPU 写门铃寄存器直接让 NIC 取走 WQE,即 IBGDA/GDAKI 的核心 |
| **golden layout** | hybrid kernel 中预计算好的"每 channel 每 token 的去向表",forward warp 照表搬运避免运行时反复计算 |

---

## 3. 总体架构

### 3.1 目录导览

```
DeepEP/
├── deep_ep/                    # Python 包
│   ├── __init__.py             # v2.1.0;导出 Buffer(legacy) / ElasticBuffer / EPHandle / Config …
│   ├── buffers/
│   │   ├── legacy.py           # V1 API:dispatch / internode_* / low_latency_* / Config
│   │   └── elastic.py          # ★ V2 API:ElasticBuffer(dispatch/combine/engram/pp/agrs)
│   ├── include/deep_ep/        # 设备端头文件(随 pip 包发行,供 JIT 编译)
│   │   ├── common/             # handle.cuh / comm.cuh / layout.cuh / ptx.cuh / exception.cuh
│   │   ├── impls/              # dispatch / combine / *_copy_epilogue / hybrid_* / engram / pp / agrs / barrier
│   │   ├── jit/  (如存在)      # JIT 相关的设备端片段
│   │   └── ...
│   ├── utils/                  # envs.py(带宽探测/初始化)、gate.py、refs.py、event_overlap 等
│   └── ...
├── csrc/
│   ├── python_api.cpp          # pybind 入口(pybind11 → C++ API)
│   ├── elastic/buffer.hpp      # ★ ElasticBuffer 的 C++ host 类(workspace 布局、host 同步)
│   ├── kernels/
│   │   ├── backend/            # api.cuh / nccl.cu / nvshmem.cu / symmetric.hpp / cuda_driver.cu
│   │   └── legacy/             # V1 kernel:AOT 编译(intranode / internode / internode_ll / layout / config)
│   ├── jit/                    # ★ JIT 子系统:compiler / cache / kernel_runtime / launch_runtime / include_parser
│   └── indexing/main.cu        # "编译冒烟测试":#include 全部 EPv2 impl + 空 main()
├── tests/                      # elastic / legacy / utils 三套测试
├── setup.py                    # 构建:决定哪些源 AOT、链接 NVSHMEM/NCCL、烘焙 persistent 环境变量
└── notes/                      # 本文档所在目录
```

### 3.2 分层与调用链

```
用户代码
  │  ElasticBuffer(group, ...).dispatch(x, topk_idx, ...)
  ▼
deep_ep/buffers/elastic.py          ── Python 参数校验、事件(EventOverlap)编排、理论 SM/QP 计算
  ▼
deep_ep._C (pybind: python_api.cpp) ── C++ ElasticBuffer(csrc/elastic/buffer.hpp):
  │                                    workspace 解析、CPU 同步(映射 host 内存)、kernel launch 参数装配
  ▼
csrc/jit (Compiler/Cache/LaunchRuntime) ── 以模板参数为 key 查找/编译 cubin,
  │                                        cuModuleLoad → cuLaunchKernelEx(可带 PDL 属性)
  ▼
deep_ep/include/deep_ep/impls/*.cuh   ── 设备端 kernel(warp 角色分化、TMA 流水线、barrier)
  ▼
common/{handle,comm,ptx}.cuh          ── NCCLGin 封装、barrier 族、PTX 工具箱
  ▼
NCCL(Gin/GDAKI → RDMA NIC doorbell;window/LSA → NVLink P2P)  +  可选 NVSHMEM(legacy)
```

### 3.3 关键结构性事实

1. **双 API 面**:`deep_ep.Buffer` = V1(legacy)API;`deep_ep.ElasticBuffer` = V2 API。两者共存于 main 分支。
2. **双编译模式**:
   - **AOT**:`setup.py` 中 `sources = ['csrc/python_api.cpp', 'csrc/kernels/legacy/layout.cu', 'csrc/kernels/legacy/intranode.cu', 'csrc/kernels/legacy/internode.cu', 'csrc/kernels/legacy/internode_ll.cu', 'csrc/kernels/backend/nvshmem.cu', 'csrc/kernels/backend/nccl.cu', 'csrc/kernels/backend/cuda_driver.cu']`——即 **V1 kernel 与 backend 全部预编译**进 `deep_ep._C`;
   - **JIT**:EPv2 的所有 kernel(`impls/*.cuh`)是**模板**,模板参数(SM 数、channel 数、rank 数、hidden 字节数、QP 数、超时……)运行时才确定,因此走 JIT。`csrc/indexing/main.cu` 用 `#include` 把全部 impl 头文件塞进一个空 `main()` 文件,作为"这套头文件能否全量编译"的冒烟测试。
3. **两个对称内存世界**:
   - EPv2:NCCL window(设备端 `ncclCommWindowRegister`,严格顺序)+ NVLink 域内 LSA 指针;
   - V1:NVSHMEM symmetric heap(独立链接,`-l:libnvshmem_host.so.3 -l:libnvshmem_device.a`)。
4. **persistent 环境变量**:`setup.py` 把 `EP_JIT_CACHE_DIR / EP_JIT_PRINT_COMPILER_COMMAND / EP_NUM_TOPK_IDX_BITS / EP_NCCL_ROOT_DIR` 在构建期**烘焙**进发行包(写进生成的 `envs.py`),避免运行环境差异导致 JIT 缓存 key 漂移或找错 NCCL。

---

## 4. 传输层:backend

传输层位于 `csrc/kernels/backend/`,对上层暴露三个命名空间(`api.cuh`):

### 4.1 三个命名空间

| 命名空间 | 内容 | 使用者 |
|---|---|---|
| `nvshmem` | `get_unique_id / init / alloc / free / barrier / finalize` | 仅 V1 legacy |
| `nccl` | `create_nccl_comm / destroy_nccl_comm / get_physical_domain_size / get_logical_domain_size / NCCLSymmetricMemoryContext` | EPv2 |
| `cuda_driver` | `batched_write / batched_wait / batched_write_and_wait`(stream batched memop) | AGRS 等 0-SM 特性 |

### 4.2 NCCL 通信与域划分

- `create_nccl_comm`:创建 NCCL communicator(尊重 `EP_SUPPRESS_NCCL_CHECK`、`EP_NIC_NAME`、`EP_OVERRIDE_RDMA_SL` 等环境变量)。
- **物理域大小** `get_physical_domain_size`:经 `ncclTeamWorld/ncclTeamLsa` 读出 (rail 数, NVLink 域大小)。
- **逻辑域大小** `get_logical_domain_size`:把物理 (rdma_ranks, nvl_ranks) 规整成规整矩形 (scaleout_ranks, scaleup_ranks)。允许 NVLink 域"装不满"一条 rail 的折叠情况,所有 EPv2 kernel 一律按逻辑域做模板参数。
- `EP_DISABLE_GIN=1` 是逃生门:强制回退(创建后 assert GIN 类型非 NONE 的断言被跳过),用于 NCCL/GDAKI 不可用时的诊断。

### 4.3 `NCCLSymmetricMemoryContext`(nccl.cu)

这是 EPv2 的"设备端通信上下文",构造流程:

1. **`ncclDevCommCreate`** 并携带 GIN 需求:
   - `ginContextCount = num_allocated_qps`,`ginExclusiveContexts = true`(每 QP 独占 context,避免 doorbell 争抢);
   - `ginQueueDepth = kGinQPDepth`(队列深度);
   - `ginTrafficClass = sl_idx`(SL 隔离);
   - `ginSignalCount = num_ranks + 2×2`(每 rank 一个 barrier 信号 + 预留);
   - `ginConnectionType = RAIL`(hybrid)或 `FULL`(direct);
   - `useRuntimeVersion`(NCCL ≥ 2.31 时走 runtime 版本)。
   随后断言 `ginType/railedGinType != NONE`——**GDAKI 是本版本的硬依赖**。
2. **对称内存分配**:调 `symmetric::alloc`(见 4.4)。
3. **window 注册**:`ncclCommWindowRegister`(集合通信,`NCCL_WIN_STRICT_ORDERING` 严格顺序标志,保证 put 的顺序语义),得到 `ncclWindow_t`。
4. **NVLink 域指针**:`ncclGetLsaDevicePointer` 把每个 NVLink 对端的 window 基址取回 host,设备端再按偏移换算(`nvl_window_ptrs` / `get_sym_ptr`)。

### 4.4 对称内存三兄弟(symmetric.hpp)

统一约束:**2 MiB 对齐**(`kNumAlignmentBytes`)、断言 `gpuDirectRDMACapable`(GPUDirect RDMA)、检查分配粒度。三个分配器对应三种内存形态:

| 类 | 形态 | 实现 |
|---|---|---|
| `GPUSymmetricMemory` | 纯 GPU | `ncclMemAlloc` |
| `ElasticSymmetricMemory` | **同一 VA 段 [GPU 前部][CPU 后部]** | CUDA VMM(`cuMemCreate/cuMemMap/cuMemAddressReserve`),GPU 段 + CPU(pinned)段连续映射;`cumem_create_with_fallback` 优先 FABRIC handle、失败退回 POSIX handle;`set_access` 对 peer GPU 与本 NUMA 放开 |
| `HybridElasticSymmetricMemory` | **[GPU 段][CPU 段: rank0\|rank1\|…] 每 rank 一段** | 每 rank 的 CPU 段在**本 rank 的 NUMA 节点本地**分配;跨进程共享用 **POSIX FD 传递**:rank 0 用 `create_cpu_handle` 导出 FD,各 NVLink 域内 peer 用 **`pidfd_open` + `pidfd_getfd`** 取回 FD 并 `cuMemMap` 进自己的 VA;用完 `release` 句柄。这是"GPU 显存 + 每卡 NUMA 本地内存"混合缓冲(Engram 的 CPU 存储就建在这上面)的基础 |

`alloc()` 工厂按"是否有 CPU 字节 + 是否需要 per-rank NUMA 本地"选择实现,并设置 `NCCL_ELASTIC_BUFFER_REGISTER=1`。

### 4.5 GPU+CPU 混合缓冲与超大 CPU 段

`ElasticBuffer` 构造参数 `num_bytes`(GPU 段)与 `num_cpu_bytes`(CPU 段)正交:EP 数据默认走 GPU 段;当 CPU 段大到超出 `NCCL_WIN_STRIDE` 表达能力时,Python 侧会分片处理,保证 window 偏移算术仍然成立。host 侧另有一块 `cudaMallocHost(cudaHostAllocMapped)` 的 **host_workspace**,用于 GPU kernel 把"接收 token 数"等标量直接写进 host 可见内存(kDoCPUSync 路径),CPU 无需额外同步即可读。

---

## 5. 设备端公共层:handle / comm / layout / ptx

### 5.1 `handle.cuh`:NCCLGin 封装(路径选择的"大脑")

`struct NCCLGin` 持 `ncclDevComm_t`、`ncclWindow_t`、三个团队(World/LSA/Rail)与 `lsa_base_ptr`。核心是**"能走 NVLink 直连就走 NVLink,否则走 Gin(RDMA)"** 的统一路径选择:

- `is_nvlink_accessible<team_t>(dst_rank)`:
  - LSA 团队:恒真;
  - World 团队:`rail.rank × lsa.nRanks ≤ dst < (rail.rank+1) × lsa.nRanks`(即"和自己在同一条 rail 的 NVLink 域内");
  - Rail 团队:仅自己(跨机默认不可直连)。
- `get_sym_ptr<team_t>(ptr, dst)`:返回对端可直接读写的指针——本地 bypass 返回原指针;NVLink 可达则 `ncclGetLsaPointer(window, 偏移, nvl_rank)`;不可达返回 `nullptr`(调用方转 RDMA)。
- 数据/计数原语都遵循"先 NVLink、后 RDMA"的双路:
  - `put(recv_sym, send_sym, bytes, dst)`:一律走 `gin.put`(**注释明确:本地/NVLink put 也经 NIC API 通道**,以便统一 remote action 语义);
  - `put_value(sym, value, dst)`:可达 → `st.relaxed.sys` 直写;否则 `gin.putValue`;
  - `red_add_rel(sym, value, dst)`:可达 → `red.add.rel.gpu/sys` 原子(系统作用域版本用于跨 NVLink 域);否则 `gin.signal(VASignalAdd)`(RDMA 远程原子加,仅限 64 位);
  - `get / flush / flush_async / wait`:RDMA 读与请求完成等待。

这一层的意义:**kernel 里只需要写"逻辑目的地 + 偏移",物理路径(NVLink P2P / RDMA / 本地)由模板参数与运行时判断共同决定**,direct 与 hybrid 两种模式因此可以共享大量代码。

### 5.2 `comm.cuh`:barrier 族与 QP 分配

**(a) 标签与超时。** 预定义 barrier 标签:`kDeviceBarrierTag=0, kKernelBarrierTag=1, kDispatchTag0/1, kCombineTag0/1, kHybridDispatchTag0/1, kHybridCombineTag0/1`(每个数据 kernel 前后各一次 barrier,标签区分用途以便超时报错定位)。`timeout_while` 用 `clock64()` 计时(按 2 GHz 近似,`kNumOneSecCycles = 2e9`),超时先**再等 1 秒让所有线程都打印诊断信息**,然后 `ptx::trap()`——集体自尽,死得"有信息量"。

**(b) QP 分配 `get_qp_mode`。** 规则:

- 只有 1 个 QP:全局共享(`NCCL_GIN_RESOURCE_SHARING_GPU`);
- notify warp 恒用 QP 0、CTA 内共享;
- SM 数 ≤ 可用 QP 数:**整 QP 独占给某 SM**(如 3 SM 10 QP:SM0 拿 0,3,6,9;SM1 拿 1,4,7;SM2 拿 2,5,8),CTA 共享模式——独占 QP 的 SM 之间零争抢;
- SM 数 > QP 数:全体 SM 按 `global_channel_idx % num_qps` 轮转**共享所有 QP**(GPU 共享模式)。

**(c) NVLink barrier(`nvlink_barrier_wo_local_sync`)。** 只用 **1 个 SM**;用**符号翻转协议**:counter 低 2 位编码 (phase, sign),每轮 barrier 所有 rank 向彼此的信号槽 `red_add_rel_sys(±1)`,目标值在 `0` 与 `kNumRanks` 之间交替,counter 每轮 +1——64 位 counter 按"每微秒一次"也要 57 万年才溢出,故可当终身计数器。等待用 `ld.acquire.sys` 轮询,带超时打印。

**(d) GIN barrier(`gin_barrier_wo_local_sync`)。** 三步:
1. **flush 所有 QP**(所有 SM 的所有 warp 分摊 `ncclGin(...).flush(ncclCoopWarp())`;world 团队还要 `fence_acq_rel_sys`,因为 barrier 前可能混有 NVLink 直写,必须系统可见);grid sync;
2. **SM0 用 QP0 对全体 rank `gin.signal(SignalInc)`**;
3. **等待**:轮询 GDAKI 的 **`signals_table.buffer`**(注释标明 TODO:等 NCCL 官方带超时的 wait API,当前直接 cast `gin._ginHandle` 读硬件信号表)+ `ld.acquire.sys`。

**(e) 组合 `gpu_barrier`。** 先 `tma_store_commit/wait`(防 TMA 代理通道未落盘)+ 可选 grid sync;hybrid 拓扑下 **SM0 做 scaleup barrier、其余 SM 并行做 scaleout barrier**(两级 barrier 重叠),结尾 grid sync。`do_scaleout/do_scaleup` 可按调用点裁剪。

### 5.3 `layout.cuh`:WorkspaceLayout

一块对称内存的固定布局(所有 rank 同构):NVLink barrier counter/signal(双相位)→ notify reduction 区(每 rank/每 expert 的 64 位打包计数)→ scaleup rank/expert 计数的 send/recv 双缓冲 → `scaleup_atomic_sender_counter`(slot 分配原子加)→ scaleout rank/expert 计数双缓冲 → channel 级 signaled tail(ranks × channels)→ AGRS 信号区。kernel 之间不传参数,全靠这块"黑板"。

### 5.4 `ptx.cuh`:PTX 工具箱(节选)

- **mbarrier**:`init/arrive/set_tx/wait_flip_phase`,TMA 流水线的节拍器;
- **TMA**:`tma_load_1d / tma_store_1d`、`tma_store_fence/wait<remaining>`、`cp_async_ca + cp_async_mbarrier_arrive`(SF 等小数据的异步拷贝);`TMACacheHint` 默认 `kEvictFirst`(通信数据不进 L2 缓存,保护计算流量的缓存);
- **谓词化访存**:`ldg/st` 带 gez/gtz 谓词(边界 token 的部分写入一条指令完成,不用分支);`ld_volatile / ld_acquire_sys / st_relaxed_sys / st_release_sys`;
- **原子**:`red_add`(relaxed,支持 int64 打包)、`red_add_rel_sys/gpu`;
- **warp 原语**:`exchange / gather / all / reduce_or / reduce_add / match / deduplicate / warp_inclusive_sum / warp_exclusive_sum / fadd2 / accumulate(bf162)`——dispatch 的计数前缀和、combine 的 topk 槽位解码都建立在这些原语上,不依赖 CUB;
- **`warpgroup_reg_alloc/dealloc`**:SM90 的每-warp 寄存器再分配,给不同角色的 warp 配置不同寄存器预算(数据搬运 warp 多要寄存器,控制 warp 少要);
- `named_barrier`、`fns/ffs`、`trap` 等。

---

## 6. V1 legacy kernel 体系

V1 kernel 全部 **AOT 编译**进 `deep_ep._C`(见 3.3),API 在 `deep_ep/buffers/legacy.py`(`Buffer`)。三个家族:

### 6.1 intranode(机内 NVLink)

- `num_channels = num_sms / 2`,每 channel 一对环形缓冲:发送端维护 per-channel 的 `start/end offset` 与 head/tail,接收端 warp 轮询 head/tail 消费——纯 NVLink P2P,无 RDMA。
- `notify_dispatch` 先做 `channel_prefix_matrix`(每 channel 每目的 rank 的 token 数与偏移),`dispatch` 再按矩阵搬运;`combine` 反向。
- 适合 NVLink 全互联单机。

### 6.2 internode(跨机 RDMA + NVLink 转发)

- kernel 以 `(kNumDispatchRDMASenderWarps + 1 + LEGACY_NUM_MAX_NVL_PEERS) × 32` 线程启动:**RDMA sender warp 直接打远端 NVSHMEM 缓冲 + NVLink forwarder warp 在机内转发**——V1 版的"两级流水",但角色数、QP 使用全部硬编码。
- 提供 `cached_notify`(复用上次 layout)与 combine 的 `kNumForwarders`。
- 这一代固定开销即 README 所称的 24 SM 量级。

### 6.3 low-latency(推理低时延)

- `clean_low_latency_buffer / dispatch / combine`(1024 线程)+ `query/update/clean_mask_buffer`。
- 布局 `LowLatencyLayout`:**2 个 slot(odd/even 乒乓)** 的 send/recv 数据 + recv count + combine send/recv + **flag**(signal 缓冲与 dispatch count 复用同一块,以 tag 区分)。
- 单条 dispatch 消息 = `int4` 头 + `max(hidden×bf16, hidden + scales×float)`;单条 combine 消息 = 每 128 通道一个 `nv_bfloat162` scale + `hidden×bf16`。send/recv/signal 三类缓冲各自双份。
- mask buffer 机制支持推理中动态屏蔽坏 rank。
- `Config`(lookup 表:按 (num_ranks, num_experts, num_tokens) 给出 SM/通道参数)在 legacy.py 中按规模查表,是 V1 的"准调参"方式。

### 6.4 EventOverlap

`EventHandle` + `EventOverlap` 让用户把一次通信"夹"在计算 event 之间重叠执行(`dispatch(..., async_with_compute_stream=True)` 一类接口),是 V1 时代掩盖通信延迟的主要手段;V2 侧对应 `prefer_overlap_with_compute` 与 PDL/0-SM 特性的分工。

---

## 7. EPv2:ElasticBuffer 数据通路

### 7.1 初始化与规模参数

`ElasticBuffer(group, num_bytes?, num_cpu_bytes, num_experts, num_max_tokens_per_rank, num_topk, expert_alignment, …, allow_hybrid_mode, deterministic, prefer_overlap_with_compute, sl_idx=3, num_allocated_qps, num_cpu_timeout_secs=300, num_gpu_timeout_secs=100, …)`。

**SM/QP 是解析式估算,不是自动调参**:

- `get_theoretical_num_sms`:按"搬多少字节 ÷ (SM 读带宽 + SM 写带宽)"反推最少 SM 数,默认 `sm_read_gbs=200 / sm_write_gbs=50`(可用 `EP_*` 覆盖或实测带宽),乘 **1.25 安全边际**,向上对齐到 2,下限 4,上限取 device SM 数——这就是"24 → 4-6"的来源;
- `get_theoretical_num_qps`:direct 模式 `min(num_sms, 9)`;hybrid 模式 `num_sms × 16 + 1`(两级流水线对 QP 的需求更大)。`num_allocated_qps` 的默认分配:hybrid 129 / direct 17(留出余量,实际 launch 时按需取子集)。

### 7.2 dispatch(direct 模式):notify + dispatch 两类 warp

`impls/dispatch.cuh`,`dispatch_impl<kIsScaleupNVLink, kDoCPUSync, kReuseSlotIndices, kNumSMs, kNumNotifyWarps, kNumDispatchWarps, kNumRanks, kNumHiddenBytes, kNumSFPacks, kNumMaxTokensPerRank, kNumExperts, kNumTopk, kExpertAlignment, kNumQPs, kNumTimeoutCycles>`。**channel = warp**,数据 kernel 之前的 notify 阶段:

1. **计数**:每个 token 的 topk → smem `atomicAdd` 得到每 expert 计数与**去重后的目的 rank 计数**(`ptx::deduplicate`,一个 token 命中同一 rank 的两个 expert 只算一次搬运);
2. **跨 SM 归约**:smem 计数经 64 位打包 `(1<<32)|count` 做 `red_add` 进 workspace 的 `notify_reduction` 区——**高 32 位当"到达标记"**(SM 全部到齐的栅栏)与低 32 位计数合一,一次原子完成两件事;
3. SM0 等待全体 SM 到达后:把**每 rank 计数**经 `gin.put_value(AggregateRequests)` 聚合发出、**每 expert 计数**按 NVLink 逐元素 put_value 或 RDMA 整块 put 发出;再等回全部 rank 的计数(以正数编码表示"已回填");
4. **对齐与前缀和**:expert 计数按 `kExpertAlignment` 取整(推理时对齐到 SM 倍数),warp 级 `warp_inclusive_sum` 链算前缀(不依赖 CUB),得到每 (rank, expert) 的接收区段;
5. `kDoCPUSync` 时把接收 token 总数写进 **mapped host workspace**(CPU 立刻可读,免同步);
6. notify warp 随后转为 slot 分配:`atomicAdd(scaleup_atomic_sender_counter)` 给每个待发 token 发 slot 号;`kReuseSlotIndices` 时直接复用 EPHandle 缓存的 slot 表。

**dispatch warp 主循环**(每 warp 一个 channel,token 按 channel 跨步切分):

- TMA 1D load:token hidden → smem(mbarrier 双缓冲流水);SF 用 `cp.async` 32-lane 跨步拷;`topk_idx/weights/src_token_global_idx` 一并进 smem;
- 按 slot 决定去向:**NVLink 可达** → `get_sym_ptr` 直写对端接收缓冲(TMA store);**否则** → 写本地 send 缓冲 + `gin.put`(RDMA);
- 收尾:`gpu_barrier(kDispatchTag1)` + **`cudaTriggerProgrammaticLaunchCompletion()`** 放行 epilogue kernel,并清理自己用过的原子计数器。

### 7.3 copy epilogue(独立 kernel,PDL 衔接)

`dispatch_copy_epilogue.cuh`:以 `cudaGridDependencySynchronize()` 开头(PDL 语义:epilogue 的 grid 早已启动,只是数据依赖在此汇合)。职责:

- 接收端把 send 缓冲(或 RDMA 落地缓冲)的 token **拷贝/整理**到最终输出(非 expand 布局:按 token 聚合;expand 布局:按 (token, expert) 展开),每 warp 一个 token,TMA 搬运;
- hybrid 模式另建 `channel_linked_list`(每 channel 的 token 链),供 combine 反查;
- `kDoZeroPadding`:把对齐补齐的 padding token 置零;
- `kCachedMode`(复用 handle)时跳过 CPU 计数读取,直接读 GPU 张量里的 num_recv。

### 7.4 hybrid 模式:scale-out × scale-up 两级流水线

`hybrid_dispatch.cuh` 引入第三种 warp 角色:

- **notify warp**:同上,但计数要**两级传播**——先算出 scaleout 计数(发往哪条 rail),scaleup 侧再做域内 per-rank/per-expert 分解;
- **scale-out warp**(每 channel 一个,独占一组 QP):跨域 RDMA 搬运,写对端 rail 的"中转缓冲";
- **forward warp**:域内 NVLink 接力,把中转数据按 **golden layout**(预计算的每 channel 去向表)分发到最终 slot。

两级之间用 workspace 的 channel tail(信号量)衔接,实现"域间 RDMA 在途"与"域内 NVLink 转发"重叠——这是 hybrid 相对 direct 在多机大 EP 下的吞吐来源,也是 QP 需求放大到 `num_sms×16+1` 的原因。收尾同样 `gpu_barrier(kHybridDispatchTag1)` + PDL trigger。

### 7.5 combine:三条路径

`combine_impl<kIsScaleupNVLink, kUseExpandedLayout, kAllowMultipleReduction, …>`,warp = channel,每 token 处理:

1. 读 `recv_src_metadata`(每 2+kNumTopk 元素一组)确定该 token 在各专家 slot 的位置;
2. **路径 A 无归约**(topk=1 或 `allow_multiple_reduction=False`):TMA load 专家输出 → NVLink 直达源 rank(TMA store 对称指针)或写 send 缓冲 + `gin.put`;
3. **路径 B 本地归约**(同一 token 的多个专家输出在**本 rank**):`compute_topk_slots`(`__ffs` 位扫描解码,避免 `BRA.DIV`)取出各 topk 槽,simem 内 `combine_reduce`:各 lane 拉 `int4` 向量、按权重累加至多 `kNumTopk/kNumRanks` 项,再一次 TMA store——**归约在 SMEM 完成,不经过全局内存往返**;
4. **路径 C expand 布局 send-all**:token 命中多个 rank 的专家,按槽位逐一发回。

topk 权重**写入 token metadata**随数据走,源 rank 端 combine kernel 消费时完成最终加权(权重归一也在 combine 侧)。`use_rank_layout`(`kAllowMultiple_reduction && ranks ≤ topk`)决定接收缓冲按"rank"还是按"topk 槽"组织,压缩缓冲体积。收尾 `gpu_barrier(kCombineTag0/1)`。

### 7.6 EPHandle 与确定性

`EPHandle` 缓存 dispatch 的全部布局信息:`do_expand`、`psum_num_recv_tokens_per_scaleup_rank / per_expert`、`num_unaligned_recv_tokens_per_expert`、`recv_src_metadata`、`dst_buffer_slot_idx`、`token_metadata_at_forward`、`channel_linked_list`、`deterministic_sort`。combine 直接消费它 → 免重算、且 `deterministic=True` 时整个 (dispatch, combine) 可复现。`dispatch(do_handle_copy=True)` 默认克隆 `topk_idx`(防用户原地改坏布局)。

### 7.7 实验特性(0 SM / 拷贝引擎)

- **Engram**:`engram_write / engram_fetch(indices[T,E])`——RDMA 拉取远端记忆条目;条目按 32B/LDG.256 对齐(`get_engram_storage_size_hint`);存储可放 GPU 或 **HybridElasticSymmetricMemory 的 NUMA 本地 CPU 段**;单次 kernel hook,传输走网络引擎,**不占常驻 SM**;
- **PP**:`pp_set_config/pp_send/pp_recv`——环形 slot + arrival/release 信号,点对点流水(`get_pp_buffer_size_hint`:prev/next × send/recv × 2);
- **AGRS**:`create_agrs_session / all_gather` 等会话 API——NVLink 域内 all-gather + reduce-scatter,基于 `cuda_driver::batched_write/wait` 的 stream memop(拷贝引擎),0 SM;信号区在 workspace(`kNumMaxInflightAGRS`)。

这些特性与 EP 本体共享同一块对称内存 workspace 与同一套 barrier 原语,是"一块内存、多种通信"设计的延伸。

### 7.8 CPU 同步与流控制

`ElasticBuffer.dispatch` 支持 `previous_event_before_epilogue`(epilogue 等待的计算事件)与 `prefer_overlap_with_compute`;C++ 侧 `stream_control_before_epilogue / stream_control_epilogue` 管理事件记录/等待。计数标量经 `kDoCPUSync` 路径写入 `cudaHostAllocMapped` 的 host_workspace,CPU 读计数零同步;`num_gpu_timeout_secs` 编译进 kernel 模板(超时即 trap,见 5.2)。

---

## 8. JIT 编译系统

EPv2 kernel 是重模板(`kNumSMs × kNumRanks × kNumHiddenBytes × kNumQPs × …`),AOT 不可行,因此自带一套 JIT(`csrc/jit/`):

### 8.1 编译(compiler.hpp)

- **编译器**:系统 `nvcc`(≥ 12.3;≥ 12.9 时按 arch 家族后缀),命令形如 `nvcc kernel.cu -cubin -o ...`(在临时目录内执行,**避免 cwd 头文件遮蔽**);
- **flags**:`-std=c++20 -O3 --expt-relaxed-constexpr --expt-extended-lambda --gpu-architecture=sm_90 --diag-suppress=39,161,174,177,186,940,3012 --ptxas-options=--register-usage-level=10`(寄存器压力拉满,配合 `warpgroup_reg_alloc`);`EP_NUM_TOPK_IDX_BITS`、`EP_GIN_GDAKI_DEBUG` 以宏注入;NCCL 头目录以 `-I` 加入;
- **缓存 key**:`name$signature$flags$code` 的 hex 摘要 → `~/.deep_ep`(或 `EP_JIT_CACHE_DIR`)下 `cache/kernel.<name>.<hash>/`,内含 `kernel.cu` + `kernel.cubin`;
- **落盘协议**:临时目录编译 → `fsync` 文件与目录 → **原子 `rename`**;重名冲突则整体 `safe_remove_all` 后重来(**分布式文件系统上禁止 `remove_all`**,只许安全删除)——多机共缓存目录时的正确性关键;
- **符号发现**:`cuobjdump -symbols` 要求**恰好一个** `STT_FUNC/STO_ENTRY` 入口(非法符号名先过滤),`EP_JIT_PTXAS_CHECK=1` 时额外断言无 local memory 溢出(寄存器用满的信号)。

### 8.2 加载与启动(kernel_runtime / launch_runtime / include_parser)

- `KernelRuntimeCache`:path → runtime 映射,valid 判定 = `kernel.cu` 与 `kernel.cubin` 双存在;
- 加载走 **driver API**:`cuModuleLoad / cuModuleGetFunction`;
- `LaunchRuntime`(CRTP:子类 `generate_impl()` 生成源码 + include 注释):`cuLaunchKernelEx` 启动,`cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` 放大共享内存;模板参数含 PDL 时置 `programmaticStreamSerializationAllowed = 1`——**PDL 属性与 kernel 代码在同一个 JIT 产物里闭环**;
- **IncludeParser**:源码里 `<deep_ep/*>` 的 include 会被**递归哈希**进 cache key(环检测;非 deep_ep include 直接报错)——改任何一个头文件,全部依赖它的 kernel 自动重编,且不会"改了头没重编"。

### 8.3 环境变量家族

`EP_JIT_CACHE_DIR`(缓存目录,构建期可烘焙)、`EP_JIT_PRINT_COMPILER_COMMAND`、`EP_JIT_PTXAS_CHECK`、`EP_JIT_SOURCE_DEBUG`;NCCL 侧 `EP_NCCL_ROOT_DIR`(烘焙)、`EP_DISABLE_GIN`、`EP_NIC_NAME`、`EP_OVERRIDE_RDMA_SL`、`EP_GIN_GDAKI_DEBUG`;运行侧 `EP_BUFFER_DEBUG`。

---

## 9. 工具层与构建

- **`deep_ep/utils/envs.py`**:`init_dist` 前对硬件做体检——`get_nvlink_gbs`(nvidia-smi 读链路带宽 ×0.9)、`get_rdma_gbs`(ibstat)、`check_fast_rdma_atomic_support`(原子能力)、`check_nvlink_connections`(拓扑断言)、`check_torch_deterministic`;读出的实测带宽即 7.1 解析式 SM 估算的输入;
- **`gate.py`**:测试用流量门——`generate_topk_idx / generate_rank_count` 支持指定某 rank 的 `ratio` 偏斜,用来压测"热点 rank";
- **`setup.py`**:除 3.3 的 AOT 源列表外——NVSHMEM 链接用 `_find_versioned_so` 动态解析 pip wheel 里的 `libnvshmem_host.so.3`(SONAME-only 发行)+ `-rpath` 钉住;NVCC 侧 `-rdc=true --ptxas-options=--register-usage-level=10`;`TORCH_CUDA_ARCH_LIST` 默认 `9.0`,`DISABLE_AGGRESSIVE_PTX_INSTRS` 控制激进 PTX(非 9.0 架构强制开启);
- **`tests/`**:见 §11。

---

## 10. 优化要点汇总

| # | 优化 | 机制 | 位置 |
|---|---|---|---|
| 1 | **SM 数解析式最小化(24→4-6)** | 带宽反推 + 1.25 边际 + 对齐,运行时按 (tokens, hidden, 拓扑) 算 | `elastic.py: get_theoretical_num_sms` |
| 2 | **NVLink 直连旁路 RDMA** | `get_sym_ptr` 能取到 LSA 指针就走 `red.add.rel.sys` / 直接 TMA store,零 NIC 开销 | `handle.cuh` |
| 3 | **TMA + mbarrier 双缓冲流水** | hidden 大块 1D 异步搬运,smem 双槽重叠;`TMACacheHint=EvictFirst` 不污染 L2 | `ptx.cuh`, `dispatch.cuh` |
| 4 | **PDL 消除 kernel 间隙** | dispatch 尾部 `cudaTriggerProgrammaticLaunchCompletion()`,epilogue 头部 `cudaGridDependencySynchronize()`,JIT 侧同步置 PDL launch 属性 | `impls/*`, `jit/launch_runtime.hpp` |
| 5 | **64 位打包原子** | `(1<<32)\|count` 一次原子同时完成"到达计数 + 数据计数" | `dispatch.cuh` notify |
| 6 | **整 QP 独占 / 共享双模** | SM ≤ QP 时整 QP 给单 SM(零争抢),否则全局轮转共享;notify 恒占 QP0 | `comm.cuh: get_qp_mode` |
| 7 | **hybrid 两级流水 + golden layout** | 域间 RDMA 在途时域内 NVLink 已转发;forward warp 照表搬运免重算 | `hybrid_dispatch.cuh` |
| 8 | **两级 barrier 并行** | hybrid barrier 里 SM0 做 scaleup、其余 SM 同时做 scaleout | `comm.cuh: gpu_barrier` |
| 9 | **符号翻转 NVLink barrier** | ±1 交替目标,counter 终身免复位,单 SM 完成 | `comm.cuh` |
| 10 | **SMEM 内 topk 归约** | combine 的加权求和在 smem 用 `int4` + `fadd2/accumulate` 完成,免全局往返;`__ffs` 解码免 `BRA.DIV` | `combine.cuh / combine_utils.cuh` |
| 11 | **谓词化访存** | `ldg/st` 带 gez/gtz 谓词,token 尾部部分写入免分支 | `ptx.cuh` |
| 12 | **warp 寄存器再分配** | `warpgroup_reg_alloc/dealloc` 给搬运 warp 加寄存器,`--register-usage-level=10` 兜底 | `impls/*` |
| 13 | **EPHandle 缓存** | combine 复用 dispatch 布局,免重算 + 确定性 | `EPHandle` |
| 14 | **0 SM 传输** | Engram/PP/AGRS 走网络引擎/拷贝引擎,与计算全重叠 | `engram/pp/agrs impls`, `cuda_driver.cu` |
| 15 | **JIT 原子落盘 + include 图哈希** | 分布式 FS 上缓存正确;改头文件必重编 | `csrc/jit/*` |
| 16 | **CPU 零同步读计数** | `cudaHostAllocMapped` workspace,kernel 直写 host 可见内存 | `elastic/buffer.hpp` |
| 17 | **每 rank NUMA 本地 CPU 内存** | POSIX FD + `pidfd_getfd` 跨进程共享 VMM 映射,Engram CPU 存储零跨 NUMA 访问 | `symmetric.hpp` |
| 18 | **GDAKI 信号表轮询 barrier** | 设备端直接读硬件信号表 + 超时 trap,免 CPU 介入 | `comm.cuh` |
| 19 | **SL/VL 流量隔离** | `sl_idx` → `ginTrafficClass`,与 VPC 其他流量分级 | `nccl.cu`, README 网络配置 |
| 20 | **去重 topk 目的 rank** | 同 token 命中同 rank 多 expert 只计一次搬运 | `dispatch.cuh` (`ptx::deduplicate`) |

---

## 11. 测试体系

```
tests/
├── elastic/     # EPv2:dispatch/combine 数值正确性(half/bf16/fp8 × expand/zero-padding × direct/hybrid × 确定性)
│                #          带宽吞吐(对比 NVLink/RDMA 理论带宽)、hybrid 两模式对照、Engram/PP/AGRS 冒烟
├── legacy/      # V1:intranode / internode / low_latency 正确性与 Config 表
└── utils/       # gate(流量偏斜)、带宽工具等
```

- 正确性基准:`refs.py` 提供 CPU/朴素 GPU 参考实现,dispatch 后 combine 的结果应与"本地按 expert 算完再加权"逐位(或容差内)一致;
- 确定性:`deterministic=True` 下两次运行 bit-exact;
- 性能:按 `tests/utils` 打印 GB/s 与 SM 数,和 README 表格(EP 8×2:RDMA 90/81 GB/s @12 SM;EP8 NVLink:726/740 GB/s @64 SM)对齐。

---

## 12. 附录:关键文件速查

| 文件 | 一句话 |
|---|---|
| `deep_ep/buffers/elastic.py` | V2 Python API:构造、hint、dispatch/combine、Engram/PP/AGRS、理论 SM/QP |
| `csrc/elastic/buffer.hpp` | ElasticBuffer C++ host:workspace、host 同步、launch 装配 |
| `csrc/kernels/backend/api.cuh` | 三命名空间后端门面(nvshmem/nccl/cuda_driver) |
| `csrc/kernels/backend/nccl.cu` | `NCCLSymmetricMemoryContext`:GIN 需求、window 注册、LSA 指针 |
| `csrc/kernels/backend/symmetric.hpp` | 2MiB 对齐的三类对称内存分配器(GPU / GPU+CPU / per-rank NUMA CPU) |
| `deep_ep/include/deep_ep/common/handle.cuh` | `NCCLGin`:NVLink 旁路 vs RDMA 的统一路径选择 |
| `deep_ep/include/deep_ep/common/comm.cuh` | barrier 族(NVLink 符号翻转 / GIN 信号表)、`get_qp_mode`、超时 trap |
| `deep_ep/include/deep_ep/common/layout.cuh` | `WorkspaceLayout`:全 rank 同构的"黑板"布局 |
| `deep_ep/include/deep_ep/common/ptx.cuh` | mbarrier/TMA/谓词访存/warp 原语/寄存器再分配工具箱 |
| `deep_ep/include/deep_ep/impls/dispatch.cuh` | direct dispatch:notify 阶段 + channel warp 主循环 + PDL trigger |
| `deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh` | PDL epilogue:拷贝/expand/linked-list/zero-padding |
| `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` | hybrid 三级 warp 角色 + golden layout 两级流水 |
| `deep_ep/include/deep_ep/impls/combine.cuh` + `combine_utils.cuh` | combine 三路径 + SMEM 归约 + `__ffs` 解码 |
| `csrc/jit/{compiler,cache,kernel_runtime,launch_runtime,include_parser}.hpp` | JIT 全链路:编译/缓存/原子落盘/driver 加载/PDL 启动/include 哈希 |
| `csrc/kernels/legacy/*` | V1 三家族 kernel(AOT) |
| `csrc/legacy/config.hpp` | V1 `Config` / `LowLatencyLayout`(2-slot 乒乓) |
| `deep_ep/buffers/legacy.py` | V1 Python API(`Buffer`) |
| `csrc/indexing/main.cu` | EPv2 头文件全量"编译冒烟测试" |
| `setup.py` | AOT 源列表、NVSHMEM/NCCL 链接、persistent env 烘焙、arch/PTX 开关 |
| `deep_ep/utils/envs.py` | 带宽/拓扑/原子能力体检 |

---

*本文与 [deep_ep_v2_5_changes.md](./deep_ep_v2_5_changes.md) 配套:前者讲 main 分支现状,后者讲 V2.5 分支相对它的增量(4-buffer 拆分、V1/NVSHMEM 移除、DeepJIT)。*
