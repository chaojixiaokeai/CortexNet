# Hugging Face Publishing Guide

## 1. Required Credential

Set `HF_TOKEN` with **write** permission.

## 2. Local Publish (optional)

```bash
export HF_TOKEN=hf_xxx
python scripts/release/publish_hf_assets.py --namespace chaojixiaokeai
```

## 3. GitHub Actions Publish

Configure repository secret:

- `HF_TOKEN`

Then run workflow:

- `.github/workflows/publish_hf.yml`

## 4. Outputs

1. Model repo: `https://huggingface.co/<namespace>/CortexNet`
2. Space repo: `https://huggingface.co/spaces/<namespace>/CortexNet-Demo`
