# Python 脚本功能总览

- `train_srcnn.py`: 主训练脚本（预训练/微调）
- `train_srresnet.py`: 预训练兼容入口
- `train_srgan.py`: 微调兼容入口
- `run_full_from_scratch.py`: 全流程训练入口
- `run_pretrain_x2.py/run_finetune_x2.py`: x2 单阶段入口
- `run_pretrain_x4.py/run_finetune_x4.py`: x4 单阶段入口
- `prepare_data_index.py`: 生成 `data/train_images.json`
- `eval.py`: 单配置评测与出图
- `validate_models.py`: 批量评测与汇总
