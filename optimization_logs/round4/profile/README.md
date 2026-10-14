# 启动区段剖析

该运行只用于定位成本，启用了 cProfile 和 GC 回调，不能与正常运行墙钟或时延对比。最终模型哈希、请求轨迹与第三轮同设置 seed 101 一致。

`profile_fifo_seed101/stream_startup_profile.txt`：20 个 epoch 启动共约 3.670 秒，其中 has_ffcv_support 累计约 3.198 秒，核心路径为 single_flat_dataset → leaf_datasets → 索引遍历/列表构建。启动区段的 GC 约 1.269 秒，已包含于以上时间，不能相加。

因此优先测试不遍历样本索引的结构证书，再消融主机输入准备中的推理服务点。没有通过关闭 GC、改变训练工作量或缓存输入制造收益。
