from __future__ import annotations

import torch
from transformers import TrainingArguments

from detectanyllm.training.discrepancy import compute_dc, compute_dc_from_logits
from detectanyllm.training.trainer import DDLTrainer


class _DeterministicCategorical:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def sample(self, sample_shape: tuple[int, ...]) -> torch.Tensor:
        k = sample_shape[0]
        b, t, _ = self.logits.shape
        return torch.zeros((k, b, t), dtype=torch.long, device=self.logits.device)


def test_dc_matches_manual_formula_with_fixed_resamples(monkeypatch):
    monkeypatch.setattr(
        "detectanyllm.training.discrepancy.torch.distributions.Categorical",
        _DeterministicCategorical,
    )

    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [1.0, 2.0, -2.0],
                [0.5, -1.0, 2.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)

    dc = compute_dc_from_logits(
        logits=logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_perturb_samples=4,
        sigma_eps=1e-6,
    )

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()
    log_probs = torch.log_softmax(shift_logits, dim=-1)

    original = (log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1) * shift_mask).sum(dim=-1)
    sampled = (log_probs[..., 0] * shift_mask).sum(dim=-1)
    expected = (original - sampled) / 1e-6
    torch.testing.assert_close(dc, expected)


class _DummyModel(torch.nn.Module):
    def __init__(self, human_logits: torch.Tensor, machine_logits: torch.Tensor) -> None:
        super().__init__()
        self.human_logits = human_logits
        self.machine_logits = machine_logits
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache=False):  # noqa: D401
        del attention_mask, use_cache
        device = self.dummy.device
        input_ids = input_ids.to(device)
        bsz = input_ids.shape[0]
        human = self.human_logits.to(device).unsqueeze(0).expand(bsz, -1, -1)
        machine = self.machine_logits.to(device).unsqueeze(0).expand(bsz, -1, -1)
        is_machine = (input_ids[:, :1] == 1).to(human.dtype).unsqueeze(-1)
        logits = human * (1 - is_machine) + machine * is_machine + (self.dummy * 0)
        return type("Output", (), {"logits": logits})


def test_ddl_trainer_compute_loss_matches_eq10(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "detectanyllm.training.discrepancy.torch.distributions.Categorical",
        _DeterministicCategorical,
    )

    human_logits = torch.tensor(
        [
            [2.0, 0.0, -1.0],
            [1.0, 2.0, -2.0],
            [0.5, -1.0, 2.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    machine_logits = torch.tensor(
        [
            [0.0, 2.0, -1.0],
            [-1.0, 3.0, 0.0],
            [0.2, 1.0, -2.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model = _DummyModel(human_logits=human_logits, machine_logits=machine_logits)

    trainer = DDLTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=tmp_path.as_posix(),
            per_device_train_batch_size=1,
            report_to="none",
            logging_steps=1,
        ),
        gamma=100.0,
        num_perturb_samples=4,
        sigma_eps=1e-6,
    )

    inputs = {
        "human_input_ids": torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
        "human_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "machine_input_ids": torch.tensor([[1, 2, 1, 0]], dtype=torch.long),
        "machine_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
    }

    loss = trainer.compute_loss(model, inputs)
    d_h = compute_dc(
        model,
        inputs["human_input_ids"],
        inputs["human_attention_mask"],
        num_perturb_samples=4,
        sigma_eps=1e-6,
    )
    d_m = compute_dc(
        model,
        inputs["machine_input_ids"],
        inputs["machine_attention_mask"],
        num_perturb_samples=4,
        sigma_eps=1e-6,
    )
    expected = (torch.abs(d_h) + torch.abs(100.0 - d_m)).mean()
    torch.testing.assert_close(loss, expected)
