# sr30_workspace（EDSR 训练线）

`sr30_workspace` 是独立算法线（EDSR，PSNR-first），和 `srcnn` 不同。

## 使用前警告

- `run_full_from_scratch.py` 是从零流程，会清理当前 `checkpoints/` 和 `runs/`。
- 如果你要续训，请不要用从零流程，改用单阶段脚本并设置 `resume_checkpoint`。

## 快速入口

- 单线一键（通过根目录脚本触发）：`run_one_click.bat`
- 从零全流程：`python run_full_from_scratch.py --only all`
- 仅 x2：`python run_full_from_scratch.py --only x2`
- 仅 x4：`python run_full_from_scratch.py --only x4`

## 训练策略（当前默认）

- x2：预训练 100 轮 + 微调 60 轮
- x4：预训练 100 轮 + 微调 60 轮
- 每 20 轮保存
- `edsr_x2_best.pth` / `edsr_x4_best.pth` 按最佳指标覆盖更新

## 目录说明

- `configs/`：训练和评测配置
- `data/`：训练数据与索引
- `test/`：Set14/B100/Urban100 测试集
- `checkpoints/`：训练输出权重
- `checkpoints_backup/`：权重镜像备份
- `runs/`：训练与验证日志
- `outputs/`：评测输出图像
- `docs/`：文档
