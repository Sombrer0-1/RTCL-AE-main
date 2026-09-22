# 第二轮：边界调度与有界评估缓存

目的：在固定训练批量 16、相同训练数据和优化器的前提下，提高端到端有效训练吞吐量，保留每个模型版本的完整评估。第一轮的成功/负结果保持原样，本轮重新运行配对基线，不能直接混用不同轮次的耗时。

## 实现

- `fully_parallel`：原并行基线。
- `freshness_adaptive`：第一轮方案，保留多进程和模型快照传递。
- `freshness_cached`：为第一轮多进程方案加入相同的有界评估缓存，用作新种子消融，区分缓存与执行结构的贡献。
- `boundary_fixed`：单计算进程在每个训练 experience 完成后评估，推理批量固定为 16；仅用于首轮 pilot 消融。
- `boundary`：同一 CUDA 上下文中完成训练和评估，每个模型版本评估一次；推理批量由 P95 延迟反馈调整（沿用第一轮算法），没有模型快照传递、Manager 或独立调度进程。
- `boundary_cached`：在 `boundary` 上增加有容量上限的评估输入缓存。第一轮评估正常读取数据，并同时建立缓存，后续按当前批量重新切片。默认容量上限 128 MiB，CUDA 剩余空间不足时回退到原始数据加载。缓存准备成本计入总耗时。

缓存仅在确定性 SplitCIFAR10 测试变换路径启用；不能用来缓存真实在线到达的新样本，也不适用于随机测试增强。首次建缓存时会有当前 experience 的短暂拼接内存，不把 128 MiB 描述成整个进程的显存上限。

边界调度的约束：训练期间不提供持续推理，评估需要等待当前 experience 完成。因此它适合本项目的小规模周期性快照评估，不能宣称取代具有逐请求时限的实时在线调度。观察到的吞吐收益也包含执行结构与数据缓存的贡献。

## 训练等价性

边界评估恢复 Python、NumPy、CPU/CUDA RNG 和各模块训练模式，防止评估 DataLoader 的随机数消费改变下一阶段训练。每次实验记录最终完整模型 state_dict 的 SHA-256（包含 BatchNorm buffers）。配对种子的哈希必须相同；这比只比较四位小数准确率更严格。评估批量不同可能导致极少数浮点预测差异，不代表训练模型不同。

## 主对照结果：种子 11、22、33

| 方法 | 总耗时（秒） | 平均 QPS | 训练调用耗时（秒） | 完整评估次数 | 耗时 CV |
|---|---:|---:|---:|---:|---:|
| fully_parallel | 46.72 ± 4.29 | 1076.0 | 29.91 | 5.33 | 9.18% |
| freshness_adaptive | 44.02 ± 2.37 | 1138.1 | 28.07 | 10.00 | 5.38% |
| boundary | 43.47 ± 2.50 | 1152.8 | 25.56 | 10.00 | 5.76% |
| boundary_cached | 36.10 ± 0.81 | 1385.4 | 25.41 | 10.00 | 2.24% |

相对于本轮全并行基线，`boundary_cached` 的平均 QPS 提升 **28.75%**，平均总耗时降低 **22.72%**。三个配对加速比分别为 **1.272、1.414、1.196**。相对于同样完成 10 次评估的 `boundary`，缓存后的配对加速比为 **1.181、1.166、1.264**。这是重复固定评估集上的实测结果，不是对新到达在线样本的结论。

三种子的最终模型哈希逐一完全一致，最终准确率约 49.99%。缓存持久数据量为 **122,960,000 字节（117.26 MiB）**。主对照中 12 次运行均完成全部训练及最终评估，没有超时/报错。耗时 CV 只描述本次 3 个样本，不构成普遍稳定性保证。

图表见 [comparison.png](comparison.png) / [comparison.pdf](comparison.pdf)，逐次加速比见 [paired_speedups.json](paired_speedups.json)。

## 复跑

```bash
/home/zhuzetong/.conda/envs/aocl/bin/python optimization_logs/run_comparison.py --modes fully_parallel freshness_adaptive boundary boundary_cached --training-bs 16 --output optimization_logs/round2_repeat
/home/zhuzetong/.conda/envs/aocl/bin/python optimization_logs/round2/analyze.py
/home/zhuzetong/.conda/envs/aocl/bin/python -m unittest discover -s tests
```

环境沿用上一级 `environment.json`。本组日志首行包含实际命令，所有方案都启用了相同的最终模型哈希记录。`analyze.py` 汇总同目录结果；可复制到复跑目录后执行。`train_seconds` 仅累计每次训练调用耗时，`wall_sec` 包含导入、初始化、训练、所有评估、缓存准备和退出；QPS = 50,000 / wall_sec。3 个种子仅提供小规模可行性证据。

首轮 pilot 见 `../round2_pilot/`：单独边界调度没有加速，固定批量评估更慢，原始结果完整保留。

## 验证与代码版本

实现提交：`62f9e4c`。11 项测试覆盖 RNG/模块模式恢复、缓存重新分批、末批样本、容量回退、中途取消建缓存、子进程失败传播和超时清理。另用真实进程执行 `--max_runtime 0`，按预期返回超时错误（`timeout_check.log`），这不是一次失败的性能实验。Python 编译检查与 `git diff --check` 通过。

新种子消融：

```bash
/home/zhuzetong/.conda/envs/aocl/bin/python optimization_logs/run_comparison.py --modes freshness_cached boundary_cached --training-bs 16 --seeds 44 55 66 --output optimization_logs/round2_holdout_repeat
/home/zhuzetong/.conda/envs/aocl/bin/python optimization_logs/round2/analyze.py --directory optimization_logs/round2_holdout
```

这组在较晚时间执行，观察到其他 CPU 工作负载，记录见 `../round2_holdout/host_load.json`。不与主对照合并绝对耗时，也不将其作为隔离外部干扰后的因果证明。保留相邻配对结果，供复核相同缓存条件下的执行结构差异。


新种子消融最终结果见 [补充报告](../round2_holdout/README.md)。三对相同缓存加速比分别为 1.719、1.180、1.013；全部模型哈希相同，但负载未隔离，且边界方案跨种子的耗时 CV 更高，不能宣称普遍稳健性提升。

本轮共保存 **21 次完整 GPU 运行**（3 次 pilot、12 次主对照、6 次新种子消融），全部完成训练和最终全量评估。原默认模式保持不变；`boundary_cached` 是针对当前固定数据集验证流程的可选运行模式。
