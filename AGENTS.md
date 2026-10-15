# 开发约定

- 主力开发环境为本机 NVIDIA Thor（aarch64），直接在 `main` 分支开发。
- 项目根目录为 `/home/zhuzetong/research/cl/RTCL-AE-main`；脚本中用相对路径或从脚本位置计算根目录。
- 使用 Conda `rtcl`：Python 3.11、PyTorch 2.14.0+cu132、torchvision 0.29.0+cu132，默认物理 GPU 0。未激活环境时用 `bash environment/run.sh ...`。
- Avalanche 固定提交和补丁安装方式见 `environment/README.md`；修改项目补丁放在 `modified/`，通过 `environment/install_avalanche.py` 应用。

## 目录

| 路径 | 内容 |
| --- | --- |
| `main.py` | 并发训练与推理入口 |
| `test_avalanche_lib.py`、`test_avalanche_lib_mp.py`、`test_all_features.py` | 顺序基线、旧多进程入口与功能检查驱动 |
| `src/` | 配置、模型、调度器、工作进程和工具 |
| `modified/` | Avalanche 补丁 |
| `environment/` | 本机安装、依赖锁、环境验证与运行包装 |
| `scripts/` | 实验配置；`ae/` 为论文运行及汇总，`soft_robot/` 为数据预处理 |
| `tests/` | 单元测试 |
| `experiments/reproduction/` | 小规模基线复现记录 |
| `experiments/optimization/` | 各轮优化设计、运行/分析脚本和历史实验数据；CSV、JSON、日志按轮次及运行存放 |
| `results/paper/` | AGX Orin 论文复现表格 |
| `docs/` | 论文与详细 artifact evaluation 说明 |
| `output/` | 本机新实验输出，不纳入 Git |
| `data/`、`.deps/` | 本地数据与依赖，不纳入 Git |

## 数据与验证

- 新实验写入 `output/<实验名>/`，保存日志、CSV、逐请求 JSON 和环境记录。原程序向工作目录写文件，直接运行时先进入该输出目录，再用绝对路径启动入口；矩阵运行器自动设置工作目录。
- AE 驱动默认写入 `output/thor-ae/`，可设置 `RTCL_OUTPUT_ROOT`；软机器人数据根目录用 `RTCL_SOFT_ROBOT_ROOT` 指定。
- 历史 Orin/RTX 5090 结果保留其原始数值及环境信息；本机计时使用独立目录。清理数据前检查分析脚本的消费者，保留论文依据和实验失败证据，删除空表、临时产物及已被替代的检查数据。
- 常规检查：`bash environment/run.sh -m unittest discover -s tests -v`。调整运行环境、Avalanche 补丁或 GPU 路径时，再运行 `bash environment/run.sh environment/check.py`。
