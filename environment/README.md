# NVIDIA Thor 本机环境

当前运行环境是独立的 Conda `rtcl`：Python 3.11、PyTorch 2.14.0+cu132、torchvision 0.29.0+cu132，使用 Thor 的物理 GPU 0。

Avalanche 保持论文所用提交 `eb075be393e1f458b2c352514ff6c17b5a2c0f4e`，安装为 `.deps/avalanche` 的 editable checkout。`install_avalanche.py` 自动复制分割、EndlessCL-Sim 和 SoftRobot 补丁，并将 SoftRobot 导出追加到上游初始化文件。论文算法及后续优化控制律保持原样。

## 使用

```bash
conda activate rtcl
python main.py --help
python -m unittest discover -s tests -v
python environment/check.py
```

未激活环境时也可从项目目录使用：

```bash
bash environment/run.sh main.py --help
bash environment/run.sh -m unittest discover -s tests -v
```

`check.py` 检查依赖、三个历史入口导入、真实 GPU Replay 训练，以及 FIFO、predictive 和 predictive_neural 的合作式准备路径。使用小规模合成 CIFAR10 数据，并核对最终模型哈希一致，不下载基准数据集、不执行论文性能矩阵。

## 重建

已有本机可用的 `orion` GPU 环境时，可复制其计算栈建立独立环境：

```bash
bash environment/setup.sh --source-env orion
```

也可以全新创建 Python 3.11 环境并从 PyTorch CUDA 13.2 官方索引安装计算栈：

```bash
bash environment/setup.sh
```

`RTCL_ENV_NAME` 可指定环境名称。关键计算依赖受 `constraints-thor.txt` 约束；完整已安装版本见 `requirements-thor.lock.txt`，Conda 包构建见 `conda-thor.lock.txt`。安装脚本会使用锁文件约束普通运行依赖。`verified-thor.json` 保存首次验证结果；重新安装时的实际环境验证记录写入 `output/environment.json`。

cuSPARSELt 0.8.1 官方 aarch64 wheel 的内部 `WHEEL` 文件标记为 `manylinux2014_sbsa`，会被 pip 的平台检查拒绝。`repair_wheel_metadata.py` 先验证动态库为小端 AArch64 ELF，再修正为 aarch64 标签并更新 RECORD 校验值；不修改 GPU 动态库。

## 新实验与历史记录

所有 Python 实验运行器支持 `--gpu`，默认物理 GPU 0；原程序的 `--cuda` 表示可见设备内的索引。运行器给每个输出目录保存 `execution_environment.json`，续跑时拒绝环境变化和缺少环境身份的历史结果。

新实验写入独立目录，例如：

```bash
python experiments/optimization/round5/run_matrix.py --phase pilot --gpu 0 --output output/thor-round5-pilot
python experiments/optimization/round5/run_matrix.py --phase main --gpu 0 --output output/thor-round5-main --dry-run
```

`scripts/ae/` 驱动默认写到 `output/thor-ae/`，可通过 `RTCL_OUTPUT_ROOT` 改变；软机器人数据路径通过 `RTCL_SOFT_ROBOT_ROOT` 指定。直接运行 `scripts/` 内的配置脚本前应激活 `rtcl`，`scripts/ae/` 驱动和 `run_scripts.sh` 会自动选择项目环境。`RTCL_PYTHON` 可显式覆盖 Shell 驱动的解释器。

实际 CIFAR/CORe50/EndlessCL-Sim 数据尚需下载，软机器人原始图像需另行准备。默认 Avalanche 数据位置来自用户的 `~/.avalanche/config.json`；环境检查使用合成数据不要求这些数据存在。

`experiments/reproduction/` 和 `experiments/optimization/` 中的旧环境清单、设计文档、结果和运行日志属于历史档案。新 Thor 计时结果单独保存。

平台安装参考：[NVIDIA Jetson PyTorch 安装说明](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)、[Thor CUDA 设置](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html)。
