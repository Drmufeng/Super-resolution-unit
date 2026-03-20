# 续训与 JSON 参数修改指南

## 常见修改

- 改轮次：`train.epochs`
- 改保存间隔：`train.save_every`
- 改保留数量：`train.keep_last`
- 改学习率：`train.lr_g/train.lr_d`

## 续训方式

- `init_checkpoint`：只加载权重，优化器重新开始
- `resume_checkpoint`：加载权重 + 优化器 + epoch（推荐）

建议二选一，避免混用。

## 单阶段续训（推荐流程）

1. 把配置里的 `train.epochs` 提高（例如 60 -> 100）
2. 设置 `train.resume_checkpoint` 为该阶段最新 checkpoint（或 best）
3. 保持 `save_every=20`
4. 运行对应单阶段脚本

示例（srcnn x2 微调续训）：

- 配置：`configs/finetune_x2_srgan.json`
- 运行：`python run_finetune_x2.py`
