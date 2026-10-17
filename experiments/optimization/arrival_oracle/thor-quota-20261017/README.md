# Thor训练配额与未来信息实验归档

解释与复现命令见[实验报告](../QUOTA_RESULTS.md)。所有GPU数值来自2026-10-17本机Thor GPU0 / MAXN，不能与历史机器或20步预热的首轮实验直接合并。

- `main/`：预先冻结的36次GPU矩阵。`results.csv`为运行器汇总，`audited_results.csv`为逐请求复核，`summary.csv`含均值及样本标准差，`paired_comparisons.csv`含逐种子差值、逐秒进度差与正确率分解。
- `recent/`：独立的6次近期计算成本诊断，规划函数取16条历史，其他控制仍取64条。
- `comparison/`：主矩阵和近期诊断的交叉核对、配对比较及合并绘图；不同版本分别统计。
- `pilot/`、`planner-pilot/`、`recent-pilot/`：seed19的2／1／1次pilot，不进入正式均值。
- `simulation-final/`：沿历史测量成本序列进行的确定性时间回放，仅有延迟与工作量。
- `window-planning-final/`：所有策略共享固定服务成本的模型参照，含成本输入、各窗口所选启动时刻、结果及网格敏感性。没有分类准确率模拟，也不代表连续时间全局最优解。
- `validation.json`：12种实际GPU执行配置的环境验证；`tests.txt`：65项单元测试记录。

各GPU阶段保留执行环境、功耗模式、预定矩阵及运行前后GPU信息。原始运行日志、请求、计算段、启动控制、窗口计划及对应源码位于项目根目录下`output/thor-arrival-quota-20261017/`。
