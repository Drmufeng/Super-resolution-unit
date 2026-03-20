# srcnn 文档导航

本目录对应 `srcnn` 独立算法线（SRResNet 预训练 + SRGAN 微调）。

## 先看这一条

- 从零流程会清理 checkpoint；若要续训，使用单阶段脚本 + `resume_checkpoint`。

## 快速开始

1. 推荐解释器：`<项目根目录>/.venv_shared/Scripts/python.exe`
2. 从零全流程：`python run_full_from_scratch.py --only all`
3. 单阶段续训：`python run_pretrain_x2.py` / `python run_finetune_x2.py` / `python run_pretrain_x4.py` / `python run_finetune_x4.py`

## 关键约定

- 预训练 100 轮，微调 60 轮
- 每 20 轮保存一次
- `*_best.pth` 为覆盖式最佳模型
- 训练期间会自动监控 Set14 指标

## 日志与产物

- 训练日志：`runs/*/train_metrics.csv`
- 验证日志：`runs/*/val_metrics.csv`
- checkpoint：`checkpoints/*/*.pth`
- 备份 checkpoint：`checkpoints_backup/*/*.pth`

## 详细文档

- `训练手册.md`
- `数据准备手册.md`
- `配置说明.md`
- `评测与出图手册.md`
- `续训与JSON参数修改指南.md`
- `故障排查手册.md`
- `交付目录规范.md`
- `交付验收清单.md`
