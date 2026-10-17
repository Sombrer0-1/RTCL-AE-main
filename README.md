# AdaptOCL

Concurrent on-device online continual learning inference and retraining using real-time streaming data.

在本机 NVIDIA Thor 上使用 `main` 分支开发。计算环境为 Conda `rtcl`：Python 3.11、PyTorch 2.14.0+cu132、torchvision 0.29.0+cu132，默认物理 GPU 0。

```bash
conda activate rtcl
python main.py --help
python -m unittest discover -s tests -v
```

未激活环境时使用 `bash environment/run.sh main.py --help`。安装、依赖锁和 GPU 验证见 [本机环境](environment/README.md)。

## 运行

`main.py` 通过 `--global_scheduler_mode` 选择并发调度方法；`test_avalanche_lib.py` 为顺序 Avalanche 基线。详细方法、论文实验矩阵和表格字段见 [Artifact evaluation](docs/artifact-evaluation.md)。

AOCL 从训练 batch64 起步。`--uam_alpha` 为训练优先阶段的 experience 占比（默认 0.3），`--uam_delta_acc` 为 AA 阶段的准确率阈值（默认 0.4），`--latency_budget` 为延迟归一化预算（默认 5 秒）。动态配置的训练批量在 experience 边界应用，评估批量在评估周期边界应用。`--switch_penalty` 和 `--ekya_profiling_interval` 尚无对应实现，显式设置会报错。

新实验使用 `output/` 下的独立目录。矩阵运行器自动保存工作目录、日志和环境信息，例如：

```bash
python experiments/optimization/round4/run_matrix.py --phase pilot --gpu 0 --output output/thor-round4-pilot
```

直接运行入口时，输出写入当前工作目录：

```bash
mkdir -p output/thor-replay
cd output/thor-replay
python ../../main.py --benchmark split_cifar10 --algorithm replay --global_scheduler_mode fully_parallel --cuda 0
```

AE 驱动为 `scripts/ae/run_ae.sh`、`scripts/ae/run_fig123.sh` 和 `scripts/ae/fig5/run_fig5.sh`，默认输出到 `output/thor-ae/`。设置 `RTCL_OUTPUT_ROOT` 可更改位置；这些 Shell 驱动自动选择项目环境。

## 数据

激活 `rtcl` 后，运行 `bash download_benchmark.sh` 下载基准数据。Avalanche 默认数据位置由 `~/.avalanche/config.json` 决定；本地下载文件放入 `data/`。

软机器人原始图像需另行准备，预处理说明见 [scripts/soft_robot/README.md](scripts/soft_robot/README.md)。使用 `RTCL_SOFT_ROBOT_ROOT` 指定生成的数据目录，默认是 `~/.avalanche/data/soft_robot_data_raw`。

## 目录与实验记录

- `src/`：模型、调度器、工作进程及工具；`modified/`：Avalanche 补丁。
- `environment/`：本机环境；`scripts/`：实验配置和数据预处理；`tests/`：单元测试。
- `results/paper/`：AGX Orin 论文复现表格；[docs/](docs/artifact-evaluation.md)：复现说明与[论文](docs/rtss26-paper346.pdf)。
- [experiments/reproduction/](experiments/reproduction/README.md)：小规模基线复现记录。
- [experiments/optimization/](experiments/optimization/README.md)：各轮优化设计、运行/分析脚本和历史日志、CSV、JSON。
- `output/`、`data/`、`.deps/`：本机产物、数据与依赖，均不纳入 Git。

Orin 和 RTX 5090 记录保留原始实验环境；Thor 结果单独保存。开发与目录约定见 [AGENTS.md](AGENTS.md)。

当前优化实现以 R4 的结构证书与合作式 FIFO 为基础，保留 R4／AOCL 完整测试对照入口。R5 的预测攒批、耗时网络及 R6 的校准前瞻、RLS 已退出运行代码；[R5 实验](experiments/optimization/round5/THOR_RESULTS.md)和 [R6 实验](experiments/optimization/round6/README.md)保留方法、结果与失败教训。

[未来到达信息实验](experiments/optimization/arrival_oracle/QUOTA_RESULTS.md)在 R4 空队列分支加入可选训练启动控制。控制训练进度后的36次正式Thor对照中，8步/秒的整窗规划相对无未来信息的均匀节奏，将超时率从7.57%降至4.38%，按时正确率从17.48%升至18.19%，30秒时平均训练步数相同；9步/秒时收益缩小。另有独立成本估计诊断。[首轮纯避险实验](experiments/optimization/arrival_oracle/README.md)保留其推迟训练、降低相对R4按时正确率的结果；默认仍为空队列立即训练。

[持续学习质量实验](experiments/optimization/arrival_cl/README.md)用独立时钟执行器对齐训练与推理的类别变化，明确标签延迟，完成21次固定时间预算对照及6次整窗共同工作对照。同为456步时，未来整窗规划将超时率9.89%降至7.53%、按时正确率41.94%升至43.10%，最终模型完全一致，适应时间未改善；固定时间下的单步质量策略也未稳定改善学习质量。
