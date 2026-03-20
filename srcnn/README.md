# srcnn（SRResNet/SRGAN 独立训练线）

`srcnn` 是 SRResNet 预训练 + SRGAN 微调训练线，和 `sr30_workspace` 的 EDSR 线不同。

## 使用前警告

- `run_full_from_scratch.py` 会清理 `checkpoints/`、`checkpoints_backup/`、`runs/`、`outputs/` 后再从头训练。
- 续训请使用单阶段脚本并在配置中设置 `resume_checkpoint`。

## 快速入口

- 单线一键：`run_one_click.bat`
- 从零全流程：`python run_full_from_scratch.py --only all`
- 仅 x2：`python run_full_from_scratch.py --only x2`
- 仅 x4：`python run_full_from_scratch.py --only x4`

## 默认训练节奏

- 预训练 100 轮
- 微调 60 轮
- 每 20 轮保存
- `*_best.pth` 按最佳指标覆盖

## 目录说明

- `configs/`：训练/评测配置
- `src/`：模型与数据集代码
- `data/`：训练数据与索引
- `test/`：Set14/B100/Urban100 测试集
- `checkpoints/`：模型权重
- `checkpoints_backup/`：权重备份
- `runs/`：训练日志
- `docs/`：文档
