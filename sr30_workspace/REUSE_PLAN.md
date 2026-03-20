# 复用计划（简版）

## 可复用模块

- 模型定义：`psnr_models.py`
- 调度器：`schedulers.py`
- 训练框架：`train_psnr_edsr.py`
- 评测框架：`eval_psnr_edsr.py`

## 推荐复用方式

1. 复制 `configs/*.json` 新建实验配置
2. 只改 `data/model/train/loss/monitor` 参数
3. 通过 `run_stage_select.py` 进行阶段化续训

## 兼容原则

- 不与 `srcnn` 的 checkpoint 互相加载
- 保持 `sr30_workspace` 独立目录结构
