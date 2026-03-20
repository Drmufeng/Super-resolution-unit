# Checkpoint 完整性审计报告（2026-03-15）

## 审计范围

- `<项目根目录>/srcnn/checkpoints`
- `<项目根目录>/sr30_workspace/checkpoints`

## 审计结果

- 总数：104
- 可加载：0
- 不可加载：104

## 处置动作

- 清理损坏 `.pth`
- 启用原子写入
- 启用 `checkpoints_backup` 镜像目录
- 重启全量训练流程
