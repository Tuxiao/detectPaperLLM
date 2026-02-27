# Migration Guide

This project moved from a single script to a package CLI.

## Old -> New command mapping

1. Prepare data:
`python ddl_finetune.py --stage prepare --human_file H --machine_file M --output_file O`
-> `detectanyllm prepare-pairs --human-file H --machine-file M --output-file O`

2. Train:
`python ddl_finetune.py --stage train --train_file T --base_model B --output_dir O`
-> `detectanyllm train --train-pairs-file T --model-name-or-path B --output-dir O`

3. Build reference stats (new):
`detectanyllm build-ref --model-path O --human-ref-file HR --machine-ref-file MR --ref-stats-file RS`

4. Infer:
`python ddl_finetune.py --stage infer --model_path O --input_file I --threshold TH`
-> `detectanyllm infer --model-path O --input-file I --output-file P --decision-mode threshold --threshold TH`

## Notes

1. Training input is now paired JSONL:
`{"human":"...","machine":"..."}`
2. Default inference mode is reference clustering (`--decision-mode pm`), not fixed-threshold.
3. DDL training uses online re-sampling with `q_phi=f_theta`.
4. For Qwen3-0.6B, use:
`scripts/run_qwen3_lora_example.sh` and `scripts/run_qwen3_ref_and_infer_example.sh`.
