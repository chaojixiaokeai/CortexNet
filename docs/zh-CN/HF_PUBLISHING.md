# Hugging Face 发布指南

## 1. 必要凭据

需要配置具备 **write 权限** 的 `HF_TOKEN`。

## 2. 本地发布（可选）

```bash
export HF_TOKEN=hf_xxx
python scripts/release/publish_hf_assets.py --namespace chaojixiaokeai
```

## 3. GitHub Actions 发布

在仓库 Secrets 配置：

- `HF_TOKEN`

然后手动触发：

- `.github/workflows/publish_hf.yml`

## 4. 发布结果

1. Model 仓库：`https://huggingface.co/<namespace>/CortexNet`
2. Space 仓库：`https://huggingface.co/spaces/<namespace>/CortexNet-Demo`
