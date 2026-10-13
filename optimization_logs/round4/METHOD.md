# 实现与归因边界

## 结构证书

原 Avalanche `MultiDatasetDataLoader._get_loader()` 每次迭代重新检查 FFCV；当前 fork 的 `has_ffcv_support()` 先经 `single_flat_dataset()` / `leaf_datasets()` 展开大量索引，最后才能判断叶子没有 `ffcv_info`。这不改变模型结果，却在服务线程产生长时间不可插入推理的区段。

新路径每个 epoch 遍历数据包装图：支持 AvalancheDataset、FlatData、标准 ConcatDataset/Subset，叶子必须是精确的 torchvision CIFAR10 类型，所有节点均无 `ffcv_info`。遇到未知类型或 FFCV 元信息则回退原 `_get_loader()`。验证成本取决于包装节点数，不是样本索引数；没有缓存输入或省略训练样本。

验证成功后使用原有 sampler 构造和 PyTorch loader 构造函数，保留随机数消费次序、shuffle、Replay 拼批和每 epoch 重建行为。仅修改该 loader 实例，不全局替换依赖库函数，不编辑 `.deps`。实现依赖当前 Avalanche fork 的内部接口，升级该依赖时需要重新验证。

## 合作式主机准备

`ServiceDataset` 只包装训练输入读取：每 8 次读取调用一个轻量检查；距上次检查超过 min(SLO/4, 8 ms) 且已有请求时，服务 FIFO 队列最多 32 个请求，再恢复输入准备。它不读取未来请求，不在训练 forward/backward 中切换模型。

执行是同线程串行交错，不是异步训练和推理并行。每个服务点最多一个推理批次，避免一口气排空队列导致输入准备一直得不到执行。推理保留第三轮的 RNG、train/eval 模式恢复；不改变 BN 统计或训练优化器。

插入的推理耗时已作为推理成本观测，从下一次训练耗时观测中扣除以免重复计费；实际总墙钟时间、请求等待时间和训练停顿均不扣除它。`service_seconds` 和 `controller_seconds` 存在包含关系，不能简单相加作为总开销。

限制：仅支持 `stream_fifo`、同步 `num_workers=0` 的本地执行路径。不能抢占一个数据读取、GPU kernel、GC 或外层管理操作；未知数据后端回退时也不提供新服务点。这个检查间隔不是严格时延上界。

## 怎样判断是否改进

1. baseline→certified：消除冗余后端检查的工程收益，不能声称是队列控制公式更优。
2. certified→cooperative：新增主机服务点的边际收益，必须同时检查时延和总耗时。
3. baseline→cooperative：组合结果，不能全部归功于任一子项。

减少停顿可能增加小批推理调用次数；因此允许出现尾延迟改善但吞吐量下降的结果，不把二者混为一谈。模型/请求轨迹哈希、完整请求数量与最终测试样本量必须相等。

指标沿用第三轮：总耗时包含初始化、warmup、全部训练、所有请求、最终完整测试及进程退出；goodput 使用最后响应时刻作为分母，不能将固定到达率下的 goodput 当作饱和容量。请求完成时的模型进度可能变化，在线准确率与最终准确率分别留档。

新增 `stream_preparation.json` 记录证书成功/回退次数、准备阶段服务批次/请求数、服务耗时与检查耗时。正式矩阵没有 cProfile 或 GC 钩子。没有关闭 GC，也没有给任何组减少最终测试或计时范围。
