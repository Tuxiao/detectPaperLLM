# PyTorch + FastAPI 独立服务化方案 — 完成总结

## 目录结构

```
depoly-pytorch/
├── serve.py          # FastAPI 服务入口
├── inference.py      # DC 计算（自包含，来自 predict.py + discrepancy.py）
├── model_loader.py   # 模型加载（自包含，来自 lora.py）
├── reference.py      # 参考分布 KNN 概率估计（自包含，来自 reference_clustering.py）
├── requirements.txt  # 依赖：torch, transformers, peft, fastapi, uvicorn
├── start.sh          # 一键启动脚本
├── model-info.md     # 模型信息
└── GPU-machine.ini   # GPU 机器信息
```

> [!IMPORTANT]
> 零外部代码依赖 — 所有推理逻辑已拷贝为本地模块，可直接拷贝整个目录到任何机器部署。

## 测试结果

```
11 passed in 0.14s ✅
```

## GPU 机器部署

```bash
ssh -p 23 root@117.50.191.32
cd /root/detectPaperLLM/depoly-pytorch
pip install -r requirements.txt
bash start.sh
```

测试：
```bash
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test sentence."}'
```
