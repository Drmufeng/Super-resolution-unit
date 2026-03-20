# Published Checkpoints (Git LFS)

This directory contains selected best checkpoints prepared for repository distribution via Git LFS.

## Files

| File | Scale | Pipeline | Size (bytes) | SHA256 |
|---|---|---|---:|---|
| `edsr_x2_best.pth` | x2 | EDSR (sr30) | 122454101 | `a63e0454ccbae88b8b3389221ec5655f0f46fea67eef263d8823c720095f5211` |
| `edsr_x4_best.pth` | x4 | EDSR (sr30) | 129540621 | `798bcd757faf2b56d25f9b9b33a14a6aa944e32e2cc1f1f52dfd14f399a8e95a` |
| `srgan_x2_best.pth` | x2 | SRGAN (srcnn line) | 299923643 | `b16bfe519f8c8534ac938cc6235fde652251334a44782dcabe33c94dbcbb268e` |
| `srgan_x4_best.pth` | x4 | SRGAN (srcnn line) | 301699995 | `9931d0ce3edeadcd708409e025864301f6a87f1ff933fbfef558c07bc9e94735` |

## Notes

- These files are copies of best checkpoints from local training outputs.
- Original training/output folders remain ignored by `.gitignore` to keep repo maintainable.
