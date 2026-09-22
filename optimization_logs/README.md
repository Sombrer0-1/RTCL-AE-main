# 调度算法改进实验

原始复现已独立归档：`origin/main` 提交 `2149f51`，标签 `repro-baseline-20260922`。本目录是后续探索，不能与仓库根目录的论文 AE 表混为一谈。

## 改进动机

原实现的 `latest_accuracy` / `latest_latency` 同时由训练 minibatch、训练 experience、评估 minibatch、完整评估周期覆盖。准确率来源和延迟尺度都不一致，UAM 的一次跳变不一定代表当前调度配置变好或变差。此外，`main.py` 对 AOCL 强制从训练批量 64 起步，不能把历史表相对批量 16 全并行基线的全部差异解释成纯调度收益。新方案使用独立的模型版本和完整评估周期反馈，训练批量保持可控。

## 结果（3 个配对种子，均值 ± 样本标准差）

| 方法 | 训练批量 | 总耗时 / s | QPS | 最终准确率 | 全量评估次数 | 耗时 CV |
|---|---:|---:|---:|---:|---:|---:|
| fully_parallel | 64 | 34.48 ± 0.80 | 1450.6 | 26.04% | 3.00 | 2.33% |
| adaptocl | 64 起步，随后动态改变 | 36.55 ± 4.66 | 1381.8 | 30.60% | 2.67 | 12.74% |
| freshness | 64 | 34.05 ± 1.08 | 1469.4 | 26.04% | 6.33 | 3.16% |
| freshness_adaptive | 64 | 34.47 ± 1.12 | 1451.7 | 26.04% | 7.00 | 3.26% |
| fully_parallel | 16 | 45.56 ± 1.19 | 1098.0 | 49.99% | 5.00 | 2.60% |
| freshness_adaptive | 16 | 46.95 ± 3.02 | 1067.9 | 49.99% | 10.00 | 6.43% |

结论：

1. **没有证明训练吞吐量或稳定性的全面提升。** 批量 64 时，单独 `freshness` 比全并行平均快约 1.3%，但差异小于本次运行间波动，不能称为显著加速。自适应推理变体平均耗时几乎与全并行相同，说明额外复杂度没有换来稳定训练吞吐收益。
2. **评估服务量明显增加。** 批量 64 时平均完整评估由 3 次增至 6.33 / 7 次，最终准确率基本保持。批量 16 时由 5 次增至 10 次（本组覆盖全部 10 个发布版本），平均耗时从 45.56 秒升到 46.95 秒，约增加 3.1%；最终准确率约 49.99%，逐种子差异不超过 0.02 个百分点。评估覆盖更多模型版本是实测收益，不应包装成训练吞吐提高。
3. **稳定性与时延存在取舍。** 原 AOCL 的耗时 CV 为 12.74%，版本感知方案约 3.16%，但原全并行更低（2.33%）。批量 16 自适应变体耗时 CV 也高于其全并行基线。大推理批量提高每轮处理效率，同时增加单批尾延迟；本次不能宣称尾延迟改善。
4. 原 AOCL 的最终平均精度高于固定训练批量 64 的方案，其动态训练批量改变了优化过程。批量 16 的新方案精度更高、总耗时也更长，属于另一运行点，不能据此宣称支配原 AOCL。

因此保留原默认路径；将新方案作为更重视模型更新评估覆盖率的可选实验模式。若目标是严格提高训练 QPS，这次尝试的证据不足，不把负结果筛掉，也不覆盖成功的历史复现。

所有 18 次 GPU 实验均完成 10 个训练 experience 和最终全测试集评估，没有超时或 traceback。6 项策略测试通过，另通过 Python 编译检查和 `git diff --check`。图见 [comparison.png](comparison.png) / [comparison.pdf](comparison.pdf)。

## 方法

- `fully_parallel`：原全并行基线。
- `adaptocl`：原 UAM 调度器，保留其动态训练批量和进程暂停行为。
- `freshness`：协作式评估准入。只消费有进展的模型版本，根据完整评估周期耗时的 EWMA 决定下次评估等待时间（限制在 1–5 秒），模型版本落后达到 2 时在最小间隔后提前触发；训练结束后排空最终模型。没有 SIGSTOP，也不改变训练批量。这里的版本滞后规则只约束评估准入，不能保证长时间评估过程中始终满足滞后上限。
- `freshness_adaptive`：上述方法加推理批量反馈控制。整轮评估后读取批延迟 P95；低于 5 ms 时倍增批量，高于 10 ms 时减半，中间区域保持，范围为初始推理批量至 256。训练批量和优化器不变。因为不消费逐训练批量准确率，去掉其重复 GPU 同步和 Manager 写入，评估也不再逐批写入旧 UAM 共享字段。10 ms 是软反馈目标，不是实时保证。

这是调度/反馈采集和推理批量的联合改进，不能把所有收益都归于准入策略。`freshness` 是分离后的准入对照。

## 实验口径

SplitCIFAR10、ResNet-20、replay、完整 10 个 experience / 50,000 训练样本，每次一轮训练。每次完整评估 10,000 测试样本。单张 RTX 5090，CPU 线程限制为 1，同一种子在 spawn 训练与评估进程内分别设置。CuDNN deterministic 开启，不保证所有 CUDA 算子逐位确定。逐次串行运行，组内轮换方法顺序。

历史复现使用 double buffer；本次配对实验统一使用带锁的文件模型快照，以固定模型传递方式。环境见 `environment.json`、`requirements-observed.txt`。当前实际使用 `/home/zhuzetong/.conda/envs/aocl/bin/python`，PyTorch **2.11.0+cu128**；基线归档说明中探测到的 2.10.0 是系统另一个环境，不是本次实验环境。

QPS = 50,000 / 总进程墙钟时间，包含导入、初始化、训练、评估、退出。准确率是**最终模型全测试集快照准确率**，不是严格的逐样本 prequential accuracy。批延迟覆盖数据传输、推理及结果同步，不包含 DataLoader 等待/队列延迟；`max_cycle_p99_ms` 是每轮 P99 的最大值，不是合并所有样本的 P99。评估批量变化时不能把批延迟直接解释成单样本延迟。

训练数据量相同但评估次数由调度决定，因此同时报告评估次数/样本量。更多评估代表更频繁的模型观测，不等价于更高最终精度。只有 3 个种子，均值/标准差和变异系数属于探索性证据，不足以证明广泛稳定性或统计显著性。

## 运行

```bash
/home/zhuzetong/.conda/envs/aocl/bin/python optimization_logs/run_comparison.py
/home/zhuzetong/.conda/envs/aocl/bin/python optimization_logs/summarize.py
/home/zhuzetong/.conda/envs/aocl/bin/python -m unittest discover -s tests
```

`run_comparison.py` 支持 `--modes`、`--seeds`、`--training-bs`、`--output`。每条原始日志首行记录实际命令；每次运行独立工作目录，外部 300 秒超时杀掉整组子进程，检测 traceback、训练 experience 数及最终评估，失败记录不用于 QPS 汇总。不要在两个进程中同时对同一 GPU 运行对照。

后续变体与小训练批量对照：

```bash
/home/zhuzetong/.conda/envs/aocl/bin/python optimization_logs/run_comparison.py --modes freshness_adaptive --output optimization_logs/adaptive64
/home/zhuzetong/.conda/envs/aocl/bin/python optimization_logs/run_comparison.py --modes fully_parallel freshness_adaptive --training-bs 16 --output optimization_logs/batch16
/home/zhuzetong/.conda/envs/aocl/bin/python optimization_logs/plot_results.py
```

`batch16` 组开始时补充了文件快照内的版本戳：读取模型与版本在同一锁内完成，避免只依赖独立共享计数器时，把更新中的模型标成旧版本。前两组的 `version_lower_bound` 是保守下界，后续文件模式为实际快照版本。最终模型在训练结束前被加载、结束后才完成评估的情况也通过 `final_snapshot` 认证，不强制做一次重复评估。Double buffer 仍继承原实现，本次没有验证该路径的新方案表现。
