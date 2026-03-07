module.exports = {
  apps: [{
    name: 'detectPaperLLM',
    script: '/usr/local/miniconda3/envs/py312/bin/python',
    args: '-m uvicorn serve:app --host 0.0.0.0 --port 9000 --workers 1',
    cwd: '/root/detectPaperLLM/depoly-pytorch',
    interpreter: 'none',
    env: {
      MODEL_PATH: '/root/model_adapter/exp-qwen3p54b-g3-lora-high/checkpoint-1600',
      BASE_MODEL: '/root/model_base/Qwen3.5-4B',
      PYTHONUNBUFFERED: '1'
    },
    autorestart: true,
    watch: false,
    max_memory_restart: '20G',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    error_file: '/root/detectPaperLLM/depoly-pytorch/logs/error.log',
    out_file: '/root/detectPaperLLM/depoly-pytorch/logs/out.log',
    merge_logs: true
  }]
};
