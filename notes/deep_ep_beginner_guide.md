# DeepEP 零基础入门:从"什么是大模型"到"它为什么快"

> 文档版本:基于 DeepEP V2(main @ 614f119,`ElasticBuffer` + NCCL Gin backend)。
> 阅读对象:**零基础读者**——既不了解大模型,也不了解 GPU 高速通信。本文不假设任何前置知识,所有名词第一次出现时都会解释。
> 读完你能回答:① DeepEP 是什么、解决什么问题;② 它在大模型技术栈里处于什么位置;③ 它大致是怎么实现的;④ 它为什么能又快又省资源。
>
> 相关 notes:本文是其余 6 份文档的"第 0 层"。想看代码级细节,请按 §8 的路线图阅读:[主线详解](deep_ep_explained.md) · [实现导览](deep_ep_main_implementation.md) · [传输层](deep_ep_transport.md) · [NCCL GIN 内部](nccl_gin_internals.md) · [JIT/实验特性](deep_ep_runtime_and_experimental.md) · [V2.5 演进](deep_ep_v2_5_changes.md)

---

## 0. 先把答案放在最前面

**Q:DeepEP 是什么?**
一个 GPU 通信库,专门为大模型里的 **MoE(混合专家)架构**服务。它只做两件事,但把这两件事做到了极致:

- **dispatch(分发)**:把每张卡上的 token(可以理解为"一条待处理的数据")发送到"负责处理它的专家"所在的那些卡上;
- **combine(合并)**:专家算完之后,把结果送回原来的卡,并按权重加权合并。

**Q:谁在用?**
DeepSeek 开源(2025 年,MIT 协议),DeepSeek-V3 这类 MoE 大模型的训练/推理都依赖这类通信。社区还有支持 AMD GPU、异构网卡等多个分支(见 README 的 Community forks)。

**Q:它和 NCCL 是什么关系?**
NCCL 是 NVIDIA 的**通用**集合通信库(做 all-reduce 这类"规则"的通信);DeepEP 的 V2 复用了 NCCL 新提供的一个轻量引擎 **NCCL GIN**(让 GPU 上的程序直接驱动网卡),但通信的**模式**——token 到专家的不规则路由——是 DeepEP 自己设计的。类比:NCCL 提供发动机,DeepEP 造了一辆专门为 MoE 路况设计的整车。

**Q:它有多重要?**
在 MoE 模型的一层里,dispatch/combine 是绕不开的两步大流量搬运,而且**每一层、每一次训练的正反向都要做**。它们快不快、占多少资源,直接决定整个集群有多少算力真正花在"算模型"而不是"搬数据"上。DeepEP 是这个环节目前事实上的开源标杆实现之一。

带着这四个答案,下面从零开始把每个概念补齐。

---

## 1. 背景 A:大模型里的"专家"是什么(MoE)

### 1.1 大模型的最小背景

大语言模型(如 GPT、DeepSeek 系列)本质是一台"文字接龙机":把文字变成数字序列,经过几十层相同的加工(每层主要是"注意力 + 前馈网络"),最后预测下一个字。本文只需要知道两件事:

1. 模型有**几十层**结构相同的层,数据一层层流过;
2. 每层里的**前馈网络(FFN)**是参数量和计算量的大头。

### 1.2 MoE:分诊台 + 专科门诊

普通模型的 FFN 是"一个综合科":每个 token 都要把全部参数过一遍。参数想翻倍,计算量也得翻倍,成本撑不住。

**MoE(Mixture of Experts,混合专家)**的思路是把这个大 FFN 拆成 N 份独立的"专家网络",再配一个**路由器(gate/router)**:每个 token 到达时,路由器给所有专家打分,**只挑得分最高的 top-k 个专家**来处理它。

类比一家医院:病人(token)先到分诊台(router),分诊台从 256 个科室里挑 8 个最对口的,病人只去这 8 个科室,而不是把 256 个科室全跑一遍。

这样模型**参数量**可以做得很大(专家多),但每个 token 的**计算量**只增加一点点(只激活 k 个专家)。DeepSeek-V3 就是典型:每层两百余个路由专家、每个 token 每层激活 8 个(README 的性能测试就沿用这一配置:8K token/批、hidden 7168、top-8)。

### 1.3 代价:token 要"跨卡看病"

专家不会都长在同一张 GPU 上——256 个专家必然分散到集群的许多张卡上。于是问题来了:

> token 在 A 卡上,但它被分到的 8 个专家分布在 B、C、D 等卡上。要算,就必须先把 token **搬过去**;算完,还得把结果**搬回来**。

这个"搬"就是通信,而且是大规模、高频率的通信。DeepEP 的一切都围绕它展开。

---

## 2. 背景 B:卡很多,数据分散——all-to-all 从哪来

### 2.1 为什么要很多张卡

单个 GPU 的显存放不下大模型,算力也不够,所以训练/推理都在几十到几千张卡组成的集群上进行。把模型和数据切开、分给各卡,有几种并行方式,本文只展开与 DeepEP 直接相关的一种:

### 2.2 专家并行(EP,Expert Parallelism)

把全部专家**均分**到各卡上。例如 64 张卡、256 个专家 → 每张卡分到 4 个专家,这张卡就是这 4 个专家的"家"。同时,每张卡上也保管着一部分输入 token(batch 里的一批句子中分给它的那部分)。

### 2.3 all-to-all:dispatch 与 combine

每层的计算因此变成三步:

```
第 1 步 dispatch:每张卡把自己的 token 按路由结果"寄"给专家所在卡
第 2 步 本地计算:每张卡用自己家的专家,算收到的所有 token
第 3 步 combine:把专家输出按原路"寄回",并在原卡按权重加权求和
```

假设 4 张卡、每卡 4 个专家(共 16 个)、每 token 选 2 个专家,dispatch 就是这样一个"交叉投递":

```
卡 A 的 token ──┬── 2 个发给卡 B 的专家 ──→ 卡 B
                └── 2 个发给卡 D 的专家 ──→ 卡 D
卡 B 的 token ──→ 发给卡 A、卡 C
卡 C 的 token ──→ 发给卡 B、卡 D
卡 D 的 token ──→ 发给卡 A、卡 C
(每张卡都在同时收发,这就是 all-to-all:所有卡 ↔ 所有卡)
```

与"所有卡把同一份数据互相广播"的规则通信不同,MoE 的投递**极不规则**:每个 token 去哪些卡由路由结果决定,有的卡收得多(热点专家)、有的收得少,且每次都不一样。这正是通用通信库不擅长、DeepEP 要专门优化的地方。

### 2.4 通信量有多大(建立规模感)

以 README 的测试配置为例:每批 8K 个 token、每个 token 用 FP8 压缩后 7168 字节——光"完整 batch 一份"就有约 59 GB;而每个 token 还要投递到 top-8 专家所在的(最多 8 张)目标卡,单层 dispatch 的全集群流量可达数百 GB 量级;一个模型有几十层,训练一步(正向+反向)每层要做 4 次 all-to-all(dispatch、combine、以及它们各自的反向——巧合的是,dispatch 的反向恰好就是 combine,反之亦然)。如果通信不够快,昂贵的 GPU 就在排队等数据。这就是"通信库值得被单独做到极致"的原因。

---

## 3. 背景 C:GPU 之间怎么传数据(通信小白的最低限度)

### 3.1 两根管道

- **NVLink**:同一台服务器内部 GPU 之间的专用高速总线,带宽在数百 GB/s 量级。类比:同一间办公室里工人之间的传送带,又宽又快。
- **RDMA 网络**:跨服务器的网络(常见 InfiniBand 网卡 + 光纤)。它的特点是**远程直接内存访问**——一块网卡可以不经对方 CPU、直接读写对方显存,就像快递员不惊动收件人、直接把货塞进对方仓库的指定货架。带宽比 NVLink 低一个量级(README 实测中跨机路径约 61~90 GB/s,机内 NVLink 路径可达 600~740 GB/s,不同拓扑下的量级示意)。

大规模集群里两者并存:**机内走 NVLink,跨机走 RDMA**。这个"两速世界"是理解 DeepEP 一切设计的前提。

### 3.2 带宽 vs 延迟

- **带宽**:水管粗细,决定大块数据多长时间搬完(训练大 batch 时是主要矛盾);
- **延迟**:一滴水从出发到到达的时间,决定小消息要等多久(推理逐字生成时是主要矛盾)。

DeepEP 对两者都有专门优化:大 batch 走高吞吐路径,小 batch/解码走低延迟模式,统一在同一个接口里。

### 3.3 GPU 小词典(看懂后文只需这几个词)

| 术语 | 一句话解释 |
|---|---|
| GPU / 显存 | 并行计算设备;显存(HBM)是它自己的高速内存 |
| SM | GPU 上的"工人小组",一张卡有上百个;**计算和通信都要占用它们** |
| warp | 32 个线程组成的小队,SM 里干活的最小编制 |
| kernel | 一段在 GPU 上成千上万线程同时执行的程序 |
| TMA | Hopper 架构起的专用"搬运协处理器":只负责大批量内存搬运,几乎不占用算力 |
| stream(流) | GPU 的任务队列;DeepEP 用"计算流 + 通信流"两条队列并行干活 |
| CPU | 负责派活和控制,不干重活;通信库要尽量少让 GPU 等 CPU |
| QP(队列对) | RDMA 网卡上的"发送通道",像几条并行的传送带;每条都有独立容量的"深度" |

### 3.4 为什么通信"贵"

1. **占 SM**:通信 kernel 也要用工人小组,占得多了,算 GEMM 的工人就少了;
2. **有协议开销**:每次跨机发送都有固定成本(敲门、登记),小件多了成本占比极高;
3. **要同步**:多卡协同必须互等(屏障/barrier),等得多就慢。

所以 GPU 通信库的三大追求就是:**少占 SM、少来回趟数、多并行**。记住这三句,后面每个设计都能对上号。

---

## 4. DeepEP 是什么、处在哪

### 4.1 一句话与三个卖点

**DeepEP(DeepEveryParallel)= 专为 MoE 专家并行设计的高性能 GPU 通信库**,核心是 dispatch/combine 两个算子,同时附带实验性的 Engram(远程内存读取)、PP(流水线并行)、AGRS(集合通信原语)等能力。

README 亮出的三个卖点(均为官方声明):

1. **性能贴近硬件带宽上限**——多拓扑实测到达 61~740 GB/s 的瓶颈带宽(见 README Performance 表;性能数字无法从代码验证,以下同);
2. **极省 SM**——相比 V1,SM 占用从 24 降到 4~6 个,省下的算力还给计算;
3. **装完即用**——kernel 在运行时即时编译(JIT),安装阶段不需要编译 CUDA。

### 4.2 生态位置图

```
┌─────────────────────────────────────────────────────┐
│ 训练/推理框架(Megatron 系列、自定义 trainer 等)      │
│   └─ 模型的每一层:                                   │
│       注意力 → gate 打分选专家                        │
│            → ★ dispatch(DeepEP)★                   │
│            → 专家 GEMM 计算(矩阵乘法,计算库负责)      │
│            → ★ combine (DeepEP) ★                  │
│            → 下一层 ……                               │
├─────────────────────────────────────────────────────┤
│ DeepEP(本库):MoE 通信内核 + 缓冲区管理 + JIT        │
├─────────────────────────────────────────────────────┤
│ NCCL(GIN 引擎)/ RDMA 网卡驱动 / NVLink               │
├─────────────────────────────────────────────────────┤
│ 硬件:GPU 集群(NVLink 机内 + InfiniBand 跨机)        │
└─────────────────────────────────────────────────────┘
```

DeepEP 不做计算、不做路由决策(gate 是模型的一部分)、也不替代 NCCL 的通用集合通信;它只把"token 到专家"这段搬运做到最快、最省。

### 4.3 与底层的关系(一句话版)

- **机内(NVLink)**:DeepEP 用"对称内存"——所有卡把通信缓冲区注册成布局完全相同的镜像,之后写对方的缓冲区就像写自己的内存一样,一条指令直达,没有逐条消息的协商开销(详见 §6.1);
- **跨机(RDMA)**:通过 NCCL GIN 引擎,让 GPU 上正在跑的通信 kernel **直接**向网卡下指令(敲"门铃"),全程不经过 CPU(详见 §6.2)。

### 4.4 附赠的实验能力(知道即可)

同一个缓冲区体系上还搭了三样实验性功能:Engram(跨机读取"远程键值仓库",如别的卡的 KV 缓存)、PP send/recv(流水线并行相邻卡传激活)、AGRS(对称内存上的 all-gather/归约原语)。它们都追求"零 SM 或极少 SM"。细节见 [JIT/实验特性文档](deep_ep_runtime_and_experimental.md),本文不再展开。

---

## 5. 用户视角:10 分钟上手

### 5.1 只需要认识三个对象

| 对象 | 类比 | 作用 |
|---|---|---|
| `dist.ProcessGroup` | 集群通讯录 | 你是几号(rank)、队友有哪些卡 |
| `ElasticBuffer` | 一座中转仓库 + 搬运队 | 预先分配大块显存作通信缓冲区,提供 dispatch/combine;构造时按 MoE 规模自动算好仓库大小、用几个 SM、几条 QP |
| `EPHandle` | 一次发货的回执单 | dispatch 返回的路由元数据;combine 靠它原路寻址,推理时还能复用省同步 |

### 5.2 最小使用流程(简化自 README 示例)

```python
buffer = ElasticBuffer(group,
                       num_max_tokens_per_rank=8192, hidden=7168,
                       num_topk=8, use_fp8_dispatch=True)

# ① dispatch:x 是本卡的 token, topk_idx 是路由结果(每个 token 的 8 个专家编号)
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights,
    num_experts=256, num_max_tokens_per_rank=8192, expert_alignment=1,
    async_with_compute_stream=True)

# ② 本地专家计算:recv_x 已经按专家分好组,直接喂给本卡 4 个专家的 GEMM
#    (每个专家收多少 token 看 handle.num_recv_tokens_per_expert_list)

# ③ combine:把专家输出按 handle 寄回原卡,加权合并
combined_x, _, event = buffer.combine(expert_output, handle=handle,
                                      async_with_compute_stream=True)
```

### 5.3 读懂数据形状

- 发送:`x [本卡 token 数, 7168]`,`topk_idx [本卡 token 数, 8]`;
- dispatch 返回的 `recv_x`:**别人寄给我家专家的 token**,按专家排好序——这正好是本卡专家 GEMM 想要的输入布局,不用再整理;
- combine 返回的 `combined_x`:送回各原卡并加权合并后的结果。

### 5.4 常用开关(每个一句话)

- `use_fp8_dispatch`:发货前把 BF16 压成 FP8,流量省一半,精度由缩放因子补偿;
- `async_with_compute_stream=True`:通信与计算并行(配合 `event.current_stream_wait()` 等待),训练推理都建议开;
- `deterministic=True`:结果可复现(见 §6.7);
- 推理解码时把上一轮的 `handle` 传回 dispatch:路由没变就免掉重新数数和 CPU 同步;
- 训练时 dispatch 的反向就是 combine、combine 的反向就是 dispatch(README 明确此设计),所以一套接口覆盖正反向。

### 5.5 动手跑

按 README 的 Quick start 装好后,`python tests/elastic/test_ep.py` 即可在单机/多机上跑通 dispatch/combine;加 `EP_BUFFER_DEBUG=1` 能看到它自动估算的 SM/QP 数。

---

## 6. 实现全景:一次 dispatch 的旅程

从这里开始讲"怎么实现"。全程用一个类比:**快递分拣网络**。先看总览,再看七个关键设计;每一条都标注它解决了 §3.4 里的哪个"贵"。

### 6.0 总览:四步走

```
① 数数(notify):先不发货物,先互发"货单"
② 开格(前缀和):按货单把收货区分成一格一格,算好每格起点
③ 搬运(channel):搬运小队把 token 复制 top-k 份,写进目标卡收货格
④ 整理(epilogue):把到货摆成"按专家分组"的最终输出
```

**① 数数**:正式发货前,每张卡先统计"我总共要发给每张卡多少 token、发给每个专家多少",所有卡交换货单。这一步只发轻量计数,不发数据;每张卡上只派 4 个 warp(一支很小的队伍)干统计,几乎不占算力。
**② 开格**:拿到货单后,每张卡把"我将收到多少"按专家对齐、算前缀和,于是每个来件在收货区里的**格子位置提前就确定了**——后续搬运不需要任何协商,直接对格入座。
**③ 搬运**:每张卡派出若干"搬运小队"(SM 里的 warp,一队负责一条 channel),把每个 token 按 top-8 复制 8 份(同一目标只发一份,先去重),写到目标卡的格子。机内 NVLink 直写,跨机经 RDMA。
**④ 整理**:由一个独立的"整理 kernel"把到货摆进最终输出(按专家分组或按 token 展开)。combine 则是把这套流程反向再走一遍,外加按权重**加权求和**。

值得强调:combine 的搬运是 **push(推送)模型**——源卡把数据直接写进目标卡的收货格(NVLink)或写进本地发货架后主动发起 RDMA 写(跨机),目标卡只管等货到齐,不存在"目标卡去拉取"的往返。少一个往返,就少一半延迟。

### 6.1 设计一:对称内存——"先建一模一样的仓库"(省协议开销)

所有卡先把通信缓冲区注册成 NCCL 的"窗口"(window),**每张卡上这片内存的布局完全相同**。此后要写第 B 卡收货区的第 37 格,只需"窗口基址 + 同样的偏移",一条显存写指令(NVLink 直达)即可。类比:连锁仓库全部用同一张图纸建造,司机不需要问路,"去 3 号仓 B 区 37 格"在任何城市都是同一个门牌。这省掉了逐条消息"你在哪、你准备好了吗"的协商,尤其让海量小件(每 token 一件)变得可行。缓冲区按 2 MiB 对齐注册,并以严格顺序模式(NCCL_WIN_STRICT_ORDERING)保证跨卡读写一致。

### 6.2 设计二:NCCL GIN——GPU 直接敲网卡门铃(省 CPU、省趟数)

跨机通信传统上要 CPU 介入准备描述信息,而 DeepEP 的通信 kernel 借助 NCCL GIN(GPU-Initiated Networking)**在 GPU 上直接向网卡投递 RDMA 请求**:kernel 里一条指令敲"门铃"(doorbell),网卡就开始把本地发货架的内容写进对方收货格。配套两个降本手段:

- **多 QP 并行**:每个数据 channel 绑定独立 QP,像多开传送带;QP 总数按公式估算(单机模式 `min(SM数, 9)`,混合模式 `SM数×16+1`,上限由构造时的自动分配决定:65/129/17);
- **聚合小件**(AggregateRequests 选项):相邻的小写入合并成一次门铃,显著减少每件固定开销。

### 6.3 设计三:把 SM 用到极限地省(少占 SM)

这是 V2 最核心的卖点"24 → 4-6 个 SM"的来源,由三板斧组成:

1. **统计活外包给极少的 warp**:数数只占每 SM 4 个 warp,且大量工作可被 TMA(专用搬运工)承担——TMA 搬数据几乎不消耗算力单元;
2. **解析式估算 SM 数**:不靠试错调参,直接按"要搬多少字节 ÷ 链路带宽 ÷ 每个 SM 的搬运吞吐"反推最少 SM 数(内部公式:估算值 ×1.25 安全边际、向上取偶、下限 4;关掉"偏向计算"开关时抬高到 64 追求峰值带宽);
3. **SM 越少,通信和计算越能重叠**:通信只占 4~6 个 SM,剩下的全部留给专家 GEMM;两条 stream 并行,GEMM 在通信进行时就开工(`async_with_compute_stream` + 事件同步保证正确性)。

### 6.4 设计四:两级拓扑路由(应对"两速世界")

多机场景(混合模式)下,投递分两跳:**先跨机、后市内**。跨机的先由"scale-out 搬运队"整车 RDMA 送到目标机器(减少昂贵跨机趟数),同机的再由"forward 搬运队"用便宜的 NVLink 精细分发到各张卡;两跳用流水线方式重叠。单机场景(direct 模式)则一跳直达。跨机计数交换的"货单"同样走两级:先跨机汇总,再机内细分。

### 6.5 设计五:主 kernel + 整理 kernel 接力(PDL 消除空档)

搬运主 kernel 收尾时,通过 CUDA 的 **PDL(programmatic launch dependency,编程式启动依赖)**"预告"下一个整理 kernel 自动接力启动,让整理 kernel 的启动开销与主 kernel 的收尾重叠,消除 kernel 之间的空转间隙。(dispatch 侧主 kernel 结尾显式触发放行;combine 侧由整理 kernel 带启动属性实现同样效果。)

### 6.6 设计六:JIT 现场定制 kernel(装完即用 + 专用最优)

卡数、每卡 token 上限、hidden 大小、SM 数、拓扑模式……每个组合理论上是不同的最优 kernel。DeepEP 把这些做成模板参数,**第一次运行时按你的集群现场编译**(缓存在 `~/.deep_ep`,改了任何头文件会自动重编相关 kernel)。好处:安装包不用预编译(装完即用)、没出现过的组合零成本、每个用户拿到的都是为自己的规模定制的 kernel。

### 6.7 设计七:确定性与 handle 缓存(工程可用性)

- **确定性**:并行搬运的到货顺序天然不确定,`deterministic=True` 时在 dispatch 完成后由 Python 端按"源 token 全局编号"把输出重排成固定顺序(不是再加一个 GPU kernel),保证训练可复现;
- **handle 缓存**:推理解码时 gate 决策常常连续多步不变,把上一次的"回执单"直接传入,跳过数数和 CPU 同步,降低每步延迟。

### 6.8 汇总:为什么能行

| 挑战(§3.4 的"贵") | DeepEP 的做法 | 直接收益 |
|---|---|---|
| 通信极不规则、件数巨大 | 先互发货单 + 对格入座(§6.0①②) | 零协商搬运,支持到 EP2048 规模 |
| NVLink/RDMA 两速世界 | 对称内存直写(§6.1)+ 两级路由(§6.4) | 机内满速、跨机最少趟数 |
| CPU 介入与协议开销 | GIN 门铃直投 + 多 QP + 小件聚合(§6.2) | 低延迟、高吞吐 |
| 抢占算力 | 4 warp 统计 + TMA 搬运 + 解析式 SM 估算(§6.3) | 24 → 4-6 SM(README 声明) |
| kernel 间空转 | PDL 接力(§6.5) | 尾延迟降低 |
| 场景千差万别 | JIT 现场编译(§6.6) | 每种规模都是定制最优 |
| 需要可复现/低延迟推理 | 事后排序 + handle 缓存(§6.7) | 工程可用 |

其中带"README 声明"的为官方性能数据(代码无法直接验证);其余机制均可在 [主线详解](deep_ep_explained.md) 与 [实现导览](deep_ep_main_implementation.md) 中对应到具体代码行。

---

## 7. 适用条件与边界(诚实清单)

**硬件/软件门槛**(README Requirements):Hopper(SM90)及更新的 GPU;机内需要 NVLink,跨机需要 RDMA 网络(InfiniBand 全量实测,RoCE 理论兼容);NCCL ≥ 2.30.4、PyTorch ≥ 2.10、CUDA ≥ 12.3。

**V2 的已知取舍**(README Notes):显存缓冲区占用比 V1 大;低延迟模式不再支持 0-SM RDMA;Engram/PP/AGRS 仍是实验特性。

**它不做什么**:不做模型计算(GEMM 交给计算库);不做路由决策(gate 是模型的一部分);不替代 NCCL 的通用集合通信(all-reduce 等请直接用 NCCL)。

---

## 8. 继续学习:路线图与术语速查

### 8.1 建议阅读顺序

1. **本文**(全景与直觉)→
2. [deep_ep_explained.md](deep_ep_explained.md):dispatch/combine 每一阶段在 kernel 里如何发生(主线,配大量代码片段)→
3. [deep_ep_main_implementation.md](deep_ep_main_implementation.md):仓库目录、API 全表、构建与测试(当工具书查)→
4. [deep_ep_runtime_and_experimental.md](deep_ep_runtime_and_experimental.md):JIT 机制、通信计算重叠、Engram/PP/AGRS →
5. [deep_ep_transport.md](deep_ep_transport.md) 与 [nccl_gin_internals.md](nccl_gin_internals.md):下探到底层引擎(对称内存、QP、barrier、NCCL GIN 内部)→
6. [deep_ep_v2_5_changes.md](deep_ep_v2_5_changes.md):社区 V2.5 分支把这套架构推向何方。

### 8.2 术语速查表(按出现顺序)

| 术语 | 含义 |
|---|---|
| token | 模型处理的基本单元(约一个字/词),在本文语境=一条待投递的数据 |
| FFN / 专家 | 模型每层里的大计算块;MoE 把它复制 N 份,每份叫一个专家 |
| MoE / gate / top-k | 混合专家架构 / 打分选专家的路由器 / 每个 token 选几个专家 |
| EP | 专家并行:把专家切分到多张卡 |
| dispatch / combine | 把 token 发给专家所在卡 / 把结果寄回原卡加权合并 |
| all-to-all | 所有卡同时互发数据的通信模式 |
| NVLink / RDMA | 机内高速总线 / 跨机网卡直接读写对方内存 |
| SM / warp / kernel | GPU 工人小组 / 32 线程小队 / GPU 上并行执行的程序 |
| TMA | 专用搬运协处理器,搬运几乎不占算力 |
| stream | GPU 任务队列;DeepEP 分计算流与通信流 |
| QP / doorbell | 网卡发送通道 / 通知网卡"有新活"的信号 |
| GIN | NCCL 的 GPU-Initiated Networking:GPU 程序直接驱动网卡 |
| 对称内存 / window | 各卡布局完全相同的注册缓冲区,可用同一偏移互访 |
| channel | DeepEP 中"一条 warp 负责一条独立搬运通道"的抽象 |
| notify / 前缀和 | 通信前的计数交换 / 由计数算出每个收货格的起点 |
| expand / multiple reduction | 接收布局按 (token, expert) 展开 / 是否允许合并多轮归约 |
| PDL | 编程式启动依赖:后一个 kernel 自动接力前一个,消除启动空档 |
| JIT | 运行时按需编译 kernel(而非安装时预编译) |
| handle(EPHandle) | dispatch 返回的路由回执,combine 与缓存复用都要用它 |
| barrier(屏障) | 多卡/多执行流的会合点,保证互等 |
| scale-up / scale-out | 机内 NVLink 域 / 跨机 RDMA 域(DeepEP 的两级拓扑抽象) |
