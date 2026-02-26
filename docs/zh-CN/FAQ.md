# CortexNet 常见问题（FAQ）

## 1. 为什么我在 PyPI 页面看不到“刚发布”的变化？

通常有两种原因：

1. 你发布的是同一个版本号（PyPI 不允许覆盖同版本文件）
2. 页面缓存尚未刷新（一般几分钟内更新）

建议每次发布都递增版本号（如 `3.2.2 -> 3.2.3`）。

## 2. GitHub Actions 发布失败，最常见原因是什么？

1. `PYPI_API_TOKEN` 未配置或权限错误
2. 重复上传同版本文件
3. 发布前 `build/twine check` 未通过

## 3. CI 失败但本地能跑通怎么办？

优先检查：

1. Python 版本差异（CI 含 3.9/3.11/3.12）
2. lint 规则差异
3. 可选依赖在 CI 环境是否存在

## 4. 为什么推荐统一使用 `CortexNet` 类？

`CortexNet` 是对外稳定入口；历史命名（如 `CortexNetV3`）仅用于兼容旧代码。

## 5. 如何快速验证安装正确？

```bash
python -m cortexnet --version
python -m cortexnet --smoke-test
```

## 6. 为什么第一次 `from_pretrained` 很慢？

首次权重映射和设备初始化较重，属于预期行为。可用映射缓存和 lazy 预热优化。

## 7. 发布新版本的最小步骤是什么？

1. 更新版本号（`pyproject.toml`、`cortexnet/__init__.py`）
2. 更新 `CHANGELOG.md`
3. 运行 `make check`
4. 推送并触发 `publish.yml`
