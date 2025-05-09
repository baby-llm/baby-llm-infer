# 代码结构
```Engine.py

```
```Scheduler.py

```

# 背景知识

## 为啥Scheduler 和 Detokenizer 要用子进程和ZMQ ?

在 SGLang 的推理引擎（engine.py）中，之所以采用子进程（multiprocessing）而不是线程（threading）或其他方式，主要有以下几个原因：

Python GIL 限制
Python 的线程由于全局解释器锁（GIL），在同一时刻只能有一个线程执行 Python 字节码，这会极大限制多线程在 CPU 密集型任务（如深度学习推理）中的并行性能。而多进程则每个进程有独立的 Python 解释器和 GIL，可以实现真正的并行。

隔离性和稳定性
子进程之间内存空间完全隔离，一个进程崩溃不会影响主进程和其他子进程，有利于提升系统的健壮性和容错性。比如模型加载、推理等操作如果发生异常，可以只重启相关进程。

资源独占与管理
深度学习推理通常需要独占 GPU 资源。通过多进程可以让每个进程绑定不同的 GPU 或显存空间，便于资源分配和管理。线程则无法做到这一点。

第三方库兼容性
很多深度学习相关的底层库（如 PyTorch、CUDA 等）对多线程支持有限，容易出现死锁或资源竞争问题。多进程方式更符合这些库的最佳实践。

易于扩展和分布式部署
多进程架构天然适合横向扩展（如多机多卡），每个进程可以独立负责不同的任务（如 Tokenizer、Scheduler、Detokenizer），通过 IPC（如 ZMQ）通信，便于后续分布式部署和维护。

## DP和TP的区别是什么

数据并行 (Data Parallelism, DP) vs 张量并行 (Tensor Parallelism, TP)
这两种是常见的模型并行技术，用于在多个计算设备（通常是 GPU）上训练或运行大型深度学习模型。

张量并行 (Tensor Parallelism, TP):

核心思想: 将模型中的单个大层（例如 Transformer 中的 FFN 层或 Attention 层）的计算和参数切分到多个设备上。每个设备只持有和计算模型层的一部分（一个“分片”）。
工作方式:
权重矩阵被切分（例如按列或按行）。
输入激活（张量）也被相应切分或复制。
每个设备在其分片上执行计算。
通常需要在设备间进行通信（如 all-gather 或 reduce-scatter）来组合中间结果或最终输出。
优点: 可以运行单个层或整个模型都无法放入单个设备内存的超大模型。
缺点: 需要设备间有高带宽、低延迟的连接（如 NVLink），通信开销可能较大。实现相对复杂。
在此代码中: 当 dp_size == 1 时，run_scheduler_process 负责管理一个 TP rank（模型的一个分片）。多个这样的进程协同工作，共同完成整个模型的推理。
数据并行 (Data Parallelism, DP):

核心思想: 将整个模型完整地复制到多个设备上。每个设备处理输入数据的不同部分（不同的 mini-batch）。
工作方式:
输入数据被分成多个子批次。
每个设备使用其上的模型副本独立处理一个子批次。
（训练时）计算梯度，然后通过通信（如 all-reduce）聚合所有设备上的梯度，最后用聚合后的梯度更新每个设备上的模型副本。
（推理时）每个副本独立处理请求，可以显著提高总吞吐量。
优点: 实现相对简单，可以有效提高训练或推理的吞吐量。对设备间带宽要求通常低于 TP。
缺点: 每个设备都需要存储整个模型的副本，因此无法运行单设备内存放不下的模型。
在此代码中: 当 dp_size > 1 时，run_data_parallel_controller_process 启动。这个控制器负责协调多个模型副本（每个副本可能内部还使用了 TP）。它会将收到的请求分发给不同的模型副本进行处理，从而实现数据并行。
总结:

TP 切分模型层，DP 复制整个模型。
TP 解决模型过大放不下的问题，DP 解决处理速度不够快的问题。
它们可以结合使用：例如，使用 TP 将一个巨大的模型切分到 4 个 GPU 上，然后将这个 4-GPU 的 TP 单元作为一个整体，再进行 DP 复制到另外几组 4-GPU 上，以同时提高处理能力和吞吐量。此代码通过 dp_size 和 tp_size 参数以及相应的启动逻辑来支持这两种模式及其组合。

# SGlang如何管理kvcache
GitHub Copilot
好的，我们来分解一下 LLM 推理流程中 CPU、GPU、内存（RAM）和磁盘之间的数据流，并结合您提供的 KV 缓存管理代码。

假设场景： 用户输入一个提示（Prompt），LLM 生成后续文本。

1. 初始化阶段 (模型加载):

磁盘 -> CPU RAM: 程序启动时，LLM 的模型权重（通常很大，几十到几百 GB）从磁盘加载到 CPU 主内存（RAM）中。
CPU RAM -> GPU VRAM: 为了利用 GPU 进行高速并行计算，模型权重从 CPU RAM 复制到 GPU 的显存（VRAM）中。这个过程可能比较耗时。此时，模型准备就绪，在 GPU 上等待处理请求。
2. 推理阶段 - Prefill (处理输入 Prompt):

用户输入 (CPU RAM): 用户的提示文本首先存在于 CPU RAM 中。
Tokenization (CPU): CPU 将输入文本分词（Tokenize），转换成一系列 Token ID。
Embedding Lookup (GPU): Token ID 列表被发送到 GPU。GPU 使用存储在 VRAM 中的 Embedding 表将 Token ID 转换为 Embedding 向量。
Attention 计算 (GPU):
GPU 上的 Transformer 层并行处理所有输入 Token 的 Embedding 向量。
在每个 Attention 层中，为每个 Token 计算查询（Query）、键（Key）、值（Value）向量。
KV 缓存写入 (GPU): 计算出的 Key 和 Value 向量（即 KV 缓存）需要被存储起来，供后续 Token 生成时使用。
TokenToKVPoolAllocator.alloc(need_size): 在 GPU 上为这些新的 KV 对分配存储槽位（Indices）。它从 free_slots 张量（也在 GPU 上）中获取所需数量的空闲索引。
MHATokenToKVPool.set_kv_buffer(...): 将计算得到的 cache_k 和 cache_v 张量（在 GPU 上）根据 alloc 分配的 loc（槽位索引），写入到 GPU VRAM 中的 k_buffer 和 v_buffer 列表对应的张量中。这些 k_buffer 和 v_buffer 就是实际存储 KV 数据的物理显存区域。
后续层计算 (GPU): 输出的向量经过前馈网络（FFN）等其他层，最终计算出下一个 Token 的概率分布。
3. 推理阶段 - Decoding (生成新 Token):

Token 选择 (CPU/GPU): 根据上一步得到的概率分布，选择（Sample）出最可能的下一个 Token ID。这个选择过程可以在 GPU 上完成，但有时为了更复杂的采样策略，结果会被传回 CPU 进行处理。
循环开始 (生成第 N+1 个 Token):
Embedding Lookup (GPU): 新生成的 Token ID 被送到 GPU，查找其 Embedding 向量。
Attention 计算 (GPU):
GPU 计算新 Token 的 Query 向量。
KV 缓存读取 (GPU): 为了计算 Attention 分数，需要用到所有前面 Token（包括 Prompt 和之前已生成的 Token）的 Key 和 Value。
MHATokenToKVPool.get_key_buffer(layer_id) 和 get_value_buffer(layer_id): 从 GPU VRAM 的 k_buffer 和 v_buffer 中读取对应层的、与当前序列相关的所有历史 Key 和 Value 向量。这些数据直接在 GPU 内部被 Attention 机制使用。
KV 缓存写入 (GPU): 计算新 Token 的 Key 和 Value 向量。
TokenToKVPoolAllocator.alloc(1): 为这个新 Token 的 KV 对分配 1 个槽位。
MHATokenToKVPool.set_kv_buffer(...): 将新计算的 cache_k 和 cache_v 写入 GPU VRAM 中刚分配的槽位。
后续层计算 (GPU): 同 Prefill 阶段，计算出下一个（第 N+2 个）Token 的概率分布。
循环继续: 重复 Decoding 步骤，直到生成结束符或达到最大长度。
4. KV 缓存管理与层级缓存 (GPU VRAM <-> CPU RAM):

显存不足: 当生成的序列变长，GPU VRAM 中的 KV 缓存 (k_buffer, v_buffer) 可能被占满。
Eviction (GPU -> CPU): 为了给新的 KV 对腾出空间，需要将一些“旧”的或“不活跃”的 KV 缓存从 GPU VRAM 移出（Evict）。
选择要移出的 KV 对（基于某种策略，如 LRU）。
HostKVCache.alloc(): 在 CPU RAM 中预留空间（如果使用了 MHATokenToKVPoolHost）。
MHATokenToKVPool.get_flat_data(indices): 从 GPU VRAM 中收集要移出的 KV 数据。
MHATokenToKVPoolHost.transfer(indices, flat_data): 将收集到的数据 (flat_data) 从 GPU 复制到 CPU RAM 中由 HostKVCache 管理的 kv_buffer（这个 buffer 可能被 pin_memory=True 以加速传输）。
TokenToKVPoolAllocator.free(indices): 释放这些 KV 对在 GPU 上占用的槽位索引，使其可被重新分配。
Loading (CPU -> GPU): 如果后续计算需要用到已被移到 CPU RAM 的 KV 缓存（例如，某个被换出的请求又被激活）。
TokenToKVPoolAllocator.alloc(): 在 GPU 上重新分配槽位。
MHATokenToKVPoolHost.get_flat_data(indices): 从 CPU RAM 的 kv_buffer 中读取所需的 KV 数据。
MHATokenToKVPool.transfer(indices, flat_data): 将数据从 CPU RAM 复制回 GPU VRAM 中新分配的槽位。
同步: HostKVCache 中的状态管理（MemoryStateInt, synchronized 装饰器）确保在 CPU 和 GPU 之间传输数据时，缓存块的状态正确（例如，不能读取正在写入的块），保证数据一致性和线程安全。
5. 结束阶段:

Detokenization (CPU): 生成的 Token ID 序列从 GPU 传回 CPU，CPU 将其转换回人类可读的文本。
资源释放: 当一个请求处理完成，其占用的所有 KV 缓存槽位（无论在 GPU 还是 CPU）都需要被释放。
TokenToKVPoolAllocator.free(indices): 释放 GPU 上的槽位。
HostKVCache.free(indices): 释放 CPU RAM 中的槽位（如果数据被换出过）。
总结:

磁盘: 主要用于模型的持久化存储。
CPU RAM: 用于存储模型（临时）、处理输入输出文本、执行控制逻辑、以及作为 GPU VRAM 不足时的 KV 缓存扩展（Host Cache）。
GPU VRAM: 存储模型权重（主要部分）、存储活跃的 KV 缓存、执行主要的并行计算（Embedding、Attention、FFN）。
CPU: 负责数据预处理（Tokenization）、后处理（Detokenization）、流程控制、调度以及与 GPU 的数据交互。
GPU: 负责绝大部分的数值计算密集型任务。
这段代码实现的 TokenToKVPoolAllocator、MHATokenToKVPool 和 MHATokenToKVPoolHost 共同构成了一个层级 KV 缓存系统，优先使用高速的 GPU VRAM，当 VRAM 不足时，利用 CPU RAM 作为二级缓存，从而支持更长的序列或更多的并发请求，尽管访问 CPU RAM 会比直接访问 VRAM 慢。数据在 VRAM 和 RAM 之间的移动是性能优化的关键部分。

6. 在 sglang 中