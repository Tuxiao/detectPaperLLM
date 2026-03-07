# PyTorch + FastAPI 独立服务化方案 — 完成总结

## 目录结构

```
    depoly-pytorch/
    ├── serve.py          # FastAPI 服务入口
    ├── requirements.txt  # 依赖：torch, transformers, peft, fastapi, uvicorn
    ├── start.sh          # 一键启动脚本
    ├── model-info.md     # 模型信息
    └── GPU-machine.ini   # GPU 机器信息
    ```

    > [!IMPORTANT]
    > 零外部代码依赖 — 推理核心逻辑直接复用了训练目录下的 `src/detectanyllm/training` 和 `src/detectanyllm/infer`。

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
