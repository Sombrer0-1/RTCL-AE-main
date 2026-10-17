# Thor：未来信息与持续学习质量

完整方法、指标与解释见[实验报告](../README.md)。

- `main/`：21次固定48秒预算的正式矩阵，三个配对种子。`summary.csv`包含均值和样本标准差；`paired_comparisons.csv`为逐种子差值及按时正确率分解；`boundary_learning.csv`为统一阶段末的模型测试表现。
- `work/`：6次修订后的整窗共同工作对照，每阶段114步。`window_statistics.csv`记录搜索可行性和耗时。
- `comparison/`：跨阶段配对，逐秒训练量、更新时间偏移及最终模型一致性检查。
- `pilot/`：预热及统一FIFO后的5次seed19 pilot；`pilot-cold/`：5次冷启动诊断，均不进入正式均值。
- `work-infeasible/`：初版逐步截止不合理的诊断，4次完整运行及预定矩阵保留，不并入修订对照均值；`diagnostic.json`说明原因。
- `tests.txt`、`validation.json`：75项单元测试及12种实际GPU执行配置验证。

各GPU阶段附环境、功耗模式、计划和运行前后GPU记录。逐请求、计算段、探针、窗口计划、最终模型及对应运行源码位于项目根目录下`output/thor-arrival-cl-20261017/`。两次未完成的实现检查也在该输出根目录保留，没有作为完整运行使用。
