# Soft robot 数据预处理

从项目根目录运行。原始图像命名为 `1.jpg` 至 `960.jpg`，放在 `data/soft_robot_data_raw/raw_images/`。可使用论文提供的[原始图像压缩包](https://drive.google.com/file/d/1fCZZF0BThM3Jk-D_ijoWlyFammKZVd8l/view)。

```bash
export RTCL_SOFT_ROBOT_ROOT="$PWD/data/soft_robot_data_raw"
rtcl_repo="$PWD"
mkdir -p "$RTCL_SOFT_ROBOT_ROOT/raw_images"
# 将原始图像解压至 raw_images 后执行：
cd "$RTCL_SOFT_ROBOT_ROOT"
python "$rtcl_repo/scripts/soft_robot/split_train_test.py"
python "$rtcl_repo/scripts/soft_robot/generate_ic.py" "$rtcl_repo/scripts/soft_robot/cases.txt" split_raw_datasets/train split_raw_datasets/test soft_robot_ic_dataset
python "$rtcl_repo/scripts/soft_robot/generate_il.py" "$rtcl_repo/scripts/soft_robot/cases.txt" split_raw_datasets/train split_raw_datasets/test soft_robot_il_dataset
```

`split_train_test.py` 从当前目录的 `raw_images/` 读取图像，按 80%/20% 分为训练与测试集。`cases.txt` 为图像标签；IC 按压力生成 12 个类别，IL 按光照生成 5 个类别。两个数据集均使用 `<数据集>/train/<类别>/` 和 `<数据集>/test/<类别>/` 结构。

训练时将 `RTCL_SOFT_ROBOT_ROOT` 指向包含 `soft_robot_ic_dataset/` 和 `soft_robot_il_dataset/` 的目录。
