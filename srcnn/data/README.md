# data 目录说明（srcnn）

本目录用于存放训练数据和训练索引。

## 推荐目录

- `data/df2k/DIV2K_train_HR/`
- `data/df2k/DIV2K_valid_HR/`（可选）

## 训练索引文件

- `data/train_images.json`

生成方式：

```bash
python prepare_data_index.py
```

说明：

- 当前索引由 `prepare_data_index.py` 生成
- 索引路径使用相对路径，便于整体迁移
