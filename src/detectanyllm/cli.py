"""Command line interface for DetectAnyLLM."""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Sequence

import torch
from transformers import Trainer, TrainingArguments, set_seed

from detectanyllm.config import (
    DDLConfig,
    DataConfig,
    ModelConfig,
    ReferenceConfig,
    TrainRuntimeConfig,
)
from detectanyllm.data.collator import PairDataCollator
from detectanyllm.data.dataset import DDLPairDataset
from detectanyllm.data.io import prepare_pairs, split_jsonl_by_group_id, split_jsonl_random
from detectanyllm.infer.predict import build_reference_distributions, infer_file
from detectanyllm.infer.reference_clustering import load_reference_stats, save_reference_stats
from detectanyllm.training.live_metrics import LiveMetricsCallback, write_live_dashboard
from detectanyllm.training.test_eval_callback import PeriodicTestMetricsCallback
from detectanyllm.training.trainer import DDLTrainer

logger = logging.getLogger("detectanyllm")


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_prepare_pairs(args: argparse.Namespace) -> int:
    count = prepare_pairs(
        human_file=args.human_file,
        machine_file=args.machine_file,
        output_file=args.output_file,
        text_field=args.text_field,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    logger.info("Wrote %d paired samples to %s", count, args.output_file)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from detectanyllm.modeling.lora import build_lora_model

    model_config = ModelConfig(
        model_name_or_path=args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    train_pairs_file = args.train_pairs_file
    validation_pairs_file = args.validation_pairs_file
    test_pairs_file = args.test_pairs_file

    should_auto_split = (
        not validation_pairs_file
        and not test_pairs_file
        and (args.dev_ratio > 0 or args.test_ratio > 0)
    )
    if should_auto_split:
        split_dir = Path(args.output_dir) / "splits"
        if args.group_id_field:
            try:
                train_path, dev_path, test_path, split_info = split_jsonl_by_group_id(
                    input_file=args.train_pairs_file,
                    output_dir=split_dir,
                    group_id_field=args.group_id_field,
                    dev_ratio=args.dev_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.split_seed,
                )
                train_pairs_file = train_path.as_posix()
                validation_pairs_file = dev_path.as_posix()
                test_pairs_file = test_path.as_posix()
                logger.info(
                    "Group split by '%s' complete: train=%d rows (%d groups), dev=%d rows (%d groups), test=%d rows (%d groups).",
                    args.group_id_field,
                    len(split_info.train_rows),
                    len(split_info.train_groups),
                    len(split_info.dev_rows),
                    len(split_info.dev_groups),
                    len(split_info.test_rows),
                    len(split_info.test_groups),
                )
                logger.info(
                    "Split files: train=%s dev=%s test=%s",
                    train_pairs_file,
                    validation_pairs_file,
                    test_pairs_file,
                )
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Group split by '%s' failed (%s). Falling back to random shuffle split.",
                    args.group_id_field,
                    exc,
                )
                train_path, dev_path, test_path, split_info = split_jsonl_random(
                    input_file=args.train_pairs_file,
                    output_dir=split_dir,
                    dev_ratio=args.dev_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.split_seed,
                )
                train_pairs_file = train_path.as_posix()
                validation_pairs_file = dev_path.as_posix()
                test_pairs_file = test_path.as_posix()
                logger.info(
                    "Random split complete: train=%d rows, dev=%d rows, test=%d rows.",
                    len(split_info.train_rows),
                    len(split_info.dev_rows),
                    len(split_info.test_rows),
                )
                logger.info(
                    "Split files: train=%s dev=%s test=%s",
                    train_pairs_file,
                    validation_pairs_file,
                    test_pairs_file,
                )
        else:
            train_path, dev_path, test_path, split_info = split_jsonl_random(
                input_file=args.train_pairs_file,
                output_dir=split_dir,
                dev_ratio=args.dev_ratio,
                test_ratio=args.test_ratio,
                seed=args.split_seed,
            )
            train_pairs_file = train_path.as_posix()
            validation_pairs_file = dev_path.as_posix()
            test_pairs_file = test_path.as_posix()
            logger.info(
                "No group-id field configured; using random shuffle split: train=%d rows, dev=%d rows, test=%d rows.",
                len(split_info.train_rows),
                len(split_info.dev_rows),
                len(split_info.test_rows),
            )
            logger.info(
                "Split files: train=%s dev=%s test=%s",
                train_pairs_file,
                validation_pairs_file,
                test_pairs_file,
            )

    data_config = DataConfig(
        train_pairs_file=train_pairs_file,
        validation_pairs_file=validation_pairs_file,
        test_pairs_file=test_pairs_file,
        human_field=args.human_field,
        machine_field=args.machine_field,
        max_length=args.max_length,
    )
    ddl_config = DDLConfig(
        gamma=args.gamma,
        num_perturb_samples=args.num_perturb_samples,
        sigma_eps=args.sigma_eps,
    )
    runtime_config = TrainRuntimeConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=args.bf16,
        save_total_limit=args.save_total_limit,
    )

    runtime_config.ensure_output_dir()
    set_seed(runtime_config.seed)
    dashboard_path = write_live_dashboard(runtime_config.output_dir)

    bf16_enabled = runtime_config.bf16 and torch.cuda.is_available()
    if runtime_config.bf16 and not torch.cuda.is_available():
        logger.warning("bf16 requested but CUDA is unavailable; falling back to full precision.")

    model, tokenizer = build_lora_model(model_config=model_config, use_bf16=bf16_enabled)
    train_dataset = DDLPairDataset(
        data_file=data_config.train_pairs_file,
        tokenizer=tokenizer,
        max_length=data_config.max_length,
        human_field=data_config.human_field,
        machine_field=data_config.machine_field,
    )
    eval_dataset = None
    if data_config.validation_pairs_file:
        eval_dataset = DDLPairDataset(
            data_file=data_config.validation_pairs_file,
            tokenizer=tokenizer,
            max_length=data_config.max_length,
            human_field=data_config.human_field,
            machine_field=data_config.machine_field,
        )
    test_dataset = None
    if data_config.test_pairs_file:
        test_dataset = DDLPairDataset(
            data_file=data_config.test_pairs_file,
            tokenizer=tokenizer,
            max_length=data_config.max_length,
            human_field=data_config.human_field,
            machine_field=data_config.machine_field,
        )
        if eval_dataset is None:
            raise ValueError(
                "test_pairs_file requires a dev set (validation_pairs_file) for threshold tuning."
            )

    training_args_kwargs = {
        "output_dir": runtime_config.output_dir,
        "learning_rate": runtime_config.learning_rate,
        "num_train_epochs": runtime_config.num_train_epochs,
        "per_device_train_batch_size": runtime_config.per_device_train_batch_size,
        "per_device_eval_batch_size": runtime_config.per_device_train_batch_size,
        "gradient_accumulation_steps": runtime_config.gradient_accumulation_steps,
        "logging_steps": runtime_config.logging_steps,
        "save_steps": runtime_config.save_steps,
        "save_total_limit": runtime_config.save_total_limit,
        "remove_unused_columns": False,
        "report_to": "none",
        "lr_scheduler_type": runtime_config.lr_scheduler_type,
        "warmup_ratio": runtime_config.warmup_ratio,
        "bf16": bf16_enabled,
        "seed": runtime_config.seed,
    }
    eval_strategy = "steps" if eval_dataset is not None else "no"
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_args_params:
        training_args_kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in training_args_params:
        training_args_kwargs["eval_strategy"] = eval_strategy
    if eval_dataset is not None:
        training_args_kwargs["eval_steps"] = runtime_config.save_steps
    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": PairDataCollator(tokenizer),
        "gamma": ddl_config.gamma,
        "num_perturb_samples": ddl_config.num_perturb_samples,
        "sigma_eps": ddl_config.sigma_eps,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = DDLTrainer(**trainer_kwargs)
    live_metrics_cb = LiveMetricsCallback(runtime_config.output_dir)
    trainer.add_callback(live_metrics_cb)
    if test_dataset is not None and args.test_eval_steps > 0:
        trainer.add_callback(
            PeriodicTestMetricsCallback(
                trainer=trainer,
                dev_dataset=eval_dataset,
                test_dataset=test_dataset,
                eval_steps=args.test_eval_steps,
                threshold_objective=args.test_threshold_objective,
            )
        )
    elif args.test_eval_steps > 0 and test_dataset is None:
        logger.warning(
            "test_eval_steps=%d is set but no test dataset is configured; periodic test metrics are disabled.",
            args.test_eval_steps,
        )
    logger.info("Live dashboard: %s", dashboard_path)
    logger.info("Live metrics json: %s", live_metrics_cb.metrics_path)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(runtime_config.output_dir)
    tokenizer.save_pretrained(runtime_config.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if eval_dataset is not None and test_dataset is not None:
        test_metrics = trainer.evaluate_test_with_dev_threshold(
            dev_dataset=eval_dataset,
            test_dataset=test_dataset,
            threshold_objective=args.test_threshold_objective,
        )
        test_metrics["test_samples"] = len(test_dataset) * 2
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    logger.info("Training complete. Adapter saved to %s", runtime_config.output_dir)
    return 0


def cmd_build_ref(args: argparse.Namespace) -> int:
    from detectanyllm.modeling.lora import load_model_for_inference

    ddl_config = DDLConfig(
        gamma=args.gamma,
        num_perturb_samples=args.num_perturb_samples,
        sigma_eps=args.sigma_eps,
    )
    reference_config = ReferenceConfig(k_neighbors=args.k_neighbors)
    model, tokenizer, device = load_model_for_inference(
        model_path=args.model_path,
        base_model=args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_bf16=args.bf16,
    )

    meta = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "text_field": args.text_field,
        "max_length": args.max_length,
        "num_perturb_samples": ddl_config.num_perturb_samples,
        "sigma_eps": ddl_config.sigma_eps,
        "gamma": ddl_config.gamma,
        "k_neighbors": reference_config.k_neighbors,
    }

    stats = build_reference_distributions(
        model=model,
        tokenizer=tokenizer,
        human_ref_file=args.human_ref_file,
        machine_ref_file=args.machine_ref_file,
        text_field=args.text_field,
        max_length=args.max_length,
        num_perturb_samples=ddl_config.num_perturb_samples,
        sigma_eps=ddl_config.sigma_eps,
        device=device,
        meta=meta,
    )
    save_reference_stats(args.ref_stats_file, stats)
    logger.info(
        "Reference stats saved to %s (|D_h|=%d, |D_m|=%d)",
        args.ref_stats_file,
        len(stats["D_h"]),
        len(stats["D_m"]),
    )
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    from detectanyllm.modeling.lora import load_model_for_inference

    ddl_config = DDLConfig(
        gamma=args.gamma,
        num_perturb_samples=args.num_perturb_samples,
        sigma_eps=args.sigma_eps,
    )
    reference_config = ReferenceConfig(k_neighbors=args.k_neighbors)
    model, tokenizer, device = load_model_for_inference(
        model_path=args.model_path,
        base_model=args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_bf16=args.bf16,
    )

    ref_stats = None
    if args.ref_stats_file:
        ref_stats = load_reference_stats(args.ref_stats_file)

    predictions = infer_file(
        model=model,
        tokenizer=tokenizer,
        input_file=args.input_file,
        output_file=args.output_file,
        text_field=args.text_field,
        max_length=args.max_length,
        num_perturb_samples=ddl_config.num_perturb_samples,
        sigma_eps=ddl_config.sigma_eps,
        decision_mode=args.decision_mode,
        threshold=args.threshold,
        ref_stats=ref_stats,
        k_neighbors=reference_config.k_neighbors,
        device=device,
    )
    logger.info("Wrote %d predictions to %s", len(predictions), args.output_file)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DetectAnyLLM DDL toolkit")
    parser.add_argument("--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare-pairs", help="Prepare paired HWT-MGT dataset")
    prepare.add_argument("--human-file", required=True)
    prepare.add_argument("--machine-file", required=True)
    prepare.add_argument("--output-file", required=True)
    prepare.add_argument("--text-field", default="text")
    prepare.add_argument("--shuffle", action="store_true")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=cmd_prepare_pairs)

    train = subparsers.add_parser("train", help="Run DDL LoRA training")
    train.add_argument("--train-pairs-file", required=True)
    train.add_argument("--validation-pairs-file")
    train.add_argument("--test-pairs-file")
    train.add_argument("--group-id-field")
    train.add_argument("--dev-ratio", type=float, default=0.1)
    train.add_argument("--test-ratio", type=float, default=0.1)
    train.add_argument("--split-seed", type=int, default=42)
    train.add_argument("--model-name-or-path", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--human-field", default="human")
    train.add_argument("--machine-field", default="machine")
    train.add_argument("--max-length", type=int, default=512)
    train.add_argument("--gamma", type=float, default=100.0)
    train.add_argument("--num-perturb-samples", type=int, default=32)
    train.add_argument("--sigma-eps", type=float, default=1e-6)
    train.add_argument("--target-modules", default="q_proj,v_proj")
    train.add_argument("--lora-r", type=int, default=8)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--lora-dropout", type=float, default=0.1)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--num-train-epochs", type=float, default=5.0)
    train.add_argument("--per-device-train-batch-size", type=int, default=2)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--logging-steps", type=int, default=10)
    train.add_argument("--save-steps", type=int, default=200)
    train.add_argument("--save-total-limit", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--resume-from-checkpoint")
    train.add_argument("--test-eval-steps", type=int, default=0)
    train.add_argument("--test-threshold-objective", choices=["mcc", "f1"], default="mcc")
    train.add_argument("--trust-remote-code", action="store_true")
    train.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    train.set_defaults(func=cmd_train)

    build_ref = subparsers.add_parser("build-ref", help="Build reference discrepancy stats")
    build_ref.add_argument("--model-path", required=True)
    build_ref.add_argument("--base-model")
    build_ref.add_argument("--human-ref-file", required=True)
    build_ref.add_argument("--machine-ref-file", required=True)
    build_ref.add_argument("--ref-stats-file", required=True)
    build_ref.add_argument("--text-field", default="text")
    build_ref.add_argument("--max-length", type=int, default=512)
    build_ref.add_argument("--gamma", type=float, default=100.0)
    build_ref.add_argument("--num-perturb-samples", type=int, default=32)
    build_ref.add_argument("--sigma-eps", type=float, default=1e-6)
    build_ref.add_argument("--k-neighbors", type=int, default=100)
    build_ref.add_argument("--trust-remote-code", action="store_true")
    build_ref.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    build_ref.set_defaults(func=cmd_build_ref)

    infer = subparsers.add_parser("infer", help="Run inference with d_c and p_m outputs")
    infer.add_argument("--model-path", required=True)
    infer.add_argument("--base-model")
    infer.add_argument("--input-file", required=True)
    infer.add_argument("--output-file", required=True)
    infer.add_argument("--ref-stats-file")
    infer.add_argument("--text-field", default="text")
    infer.add_argument("--max-length", type=int, default=512)
    infer.add_argument("--decision-mode", choices=["pm", "threshold"], default="pm")
    infer.add_argument("--threshold", type=float, default=50.0)
    infer.add_argument("--gamma", type=float, default=100.0)
    infer.add_argument("--num-perturb-samples", type=int, default=32)
    infer.add_argument("--sigma-eps", type=float, default=1e-6)
    infer.add_argument("--k-neighbors", type=int, default=100)
    infer.add_argument("--trust-remote-code", action="store_true")
    infer.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    infer.set_defaults(func=cmd_infer)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
