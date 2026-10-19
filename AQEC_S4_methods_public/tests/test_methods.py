"""Independent CPU checks for the standalone public methods.

Run with ``python -m unittest discover -s tests -v`` from the repository root.
The tests use only synthetic tensors and temporary files.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

from aqec_s4_methods import (
    Episode,
    FOMAMLTrainer,
    ModelConfig,
    QuantumS4Model,
    SyntheticComplexMSELoss,
    TaskBatch,
    finetune,
    load_checkpoint,
    make_synthetic_episodes,
    parameter_groups,
    read_task_batch,
    save_checkpoint,
)

torch.set_num_threads(1)


class ScalarComplexModel(nn.Module):
    """One real parameter with a complex output and a known derivative."""

    def __init__(self, initial: float = 0.7, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(initial, dtype=dtype))

    def forward(self, inputs, measurements=None):
        value = self.weight * inputs[:, 0, 0:1]
        return torch.complex(value, 0.5 * value)


def scalar_batch(xs: list[float], ys: list[complex]) -> TaskBatch:
    return TaskBatch(
        torch.tensor(xs, dtype=torch.float64).reshape(-1, 1, 1),
        torch.zeros(len(xs), 1, dtype=torch.float64),
        torch.tensor(ys, dtype=torch.complex128).reshape(-1, 1),
    )


def analytic_gradient(weight: float, xs: list[float], ys: list[complex]) -> float:
    """Derivative of mean |(1 + 0.5j) * w * x - y|^2, without autograd."""
    return 2.0 * sum(
        x * (weight * x - target.real)
        + 0.5 * x * (0.5 * weight * x - target.imag)
        for x, target in zip(xs, ys)
    ) / len(xs)


class PublicMethodsTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_fomaml_query_gradient_matches_independent_first_order_calculation(self):
        initial, inner_lr = 0.7, 0.08
        definitions = [
            ([-1.0, 0.5, 2.0], [0.2 + 0.1j, -0.3 + 0.4j, 0.5 - 0.2j],
             [0.3, 1.5], [0.4 - 0.2j, -0.1 + 0.6j]),
            ([0.4, 1.2], [-0.2 + 0.3j, 0.6 + 0.1j],
             [-0.8, 0.6, 1.1], [0.1 + 0.2j, -0.4 + 0.3j, 0.8 - 0.1j]),
        ]
        episodes, expected_gradients, second_order_gradients = [], [], []
        for support_x, support_y, query_x, query_y in definitions:
            adapted = initial - inner_lr * analytic_gradient(initial, support_x, support_y)
            query_gradient = analytic_gradient(adapted, query_x, query_y)
            expected_gradients.append(query_gradient)
            support_hessian = 2.5 * sum(x * x for x in support_x) / len(support_x)
            second_order_gradients.append(query_gradient * (1.0 - inner_lr * support_hessian))
            episodes.append(Episode(scalar_batch(support_x, support_y), scalar_batch(query_x, query_y)))
        expected = sum(expected_gradients) / len(expected_gradients)
        second_order = sum(second_order_gradients) / len(second_order_gradients)
        self.assertGreater(abs(expected - second_order), 1e-3)

        model = ScalarComplexModel(initial)
        trainer = FOMAMLTrainer(
            model, SyntheticComplexMSELoss(), inner_lr=inner_lr, meta_lr=0.01,
            inner_steps=1, adaptive_inner=False, weight_decay=0.0, max_grad_norm=1e12,
        )
        metrics = trainer.train_step(episodes)

        self.assertIsNotNone(model.weight.grad)
        self.assertAlmostEqual(float(model.weight.grad), expected, places=11)
        self.assertNotEqual(float(model.weight.detach()), initial)
        self.assertLess((float(model.weight.detach()) - initial) * expected, 0.0)
        self.assertEqual(metrics["valid_episodes"], len(episodes))
        self.assertEqual(metrics["gradient_tensors"], 1)

    def test_transfer_shuffle_keeps_metadata_aligned_and_uses_tail(self):
        count, epochs = 7, 2
        sample_ids = torch.arange(count, dtype=torch.float64)
        batch = TaskBatch(
            sample_ids[:, None, None].expand(-1, 2, 1).clone(),
            sample_ids[:, None].expand(-1, 2).clone(),
            torch.complex(sample_ids, torch.zeros_like(sample_ids))[:, None],
            [{"sample_id": index} for index in range(count)],
        )
        observed_ids: list[int] = []
        observed_sizes: list[int] = []

        def checking_loss(predictions, targets, measurements, metadata):
            ids = targets[:, 0].real.to(torch.long).tolist()
            self.assertEqual(ids, [record["sample_id"] for record in metadata])
            self.assertEqual(ids, measurements[:, 0].to(torch.long).tolist())
            observed_ids.extend(ids)
            observed_sizes.append(len(ids))
            return (predictions - targets).abs().square().mean(), 0.0

        history = finetune(
            ScalarComplexModel(), batch, checking_loss,
            epochs=epochs, batch_size=3, lr=0.01, seed=42,
        )
        self.assertEqual(observed_sizes, [3, 3, 1] * epochs)
        self.assertEqual(Counter(observed_ids), Counter({index: epochs for index in range(count)}))
        self.assertNotEqual(observed_ids[:count], list(range(count)))
        self.assertEqual(len(history), epochs)
        self.assertTrue(all(math.isfinite(value) for value in history))

    def test_transfer_freezes_fusion_while_updating_head(self):
        config = ModelConfig(input_dim=2, d_model=8, d_state=4, n_layers=1, output_dim=2, dropout=0.0)
        model = QuantumS4Model(**config.to_dict())
        batch = make_synthetic_episodes(
            n_tasks=1, support_size=5, query_size=2, seq_len=4,
            input_dim=2, output_dim=2, seed=6,
        )[0].support
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        # Simulate gradients left by a preceding meta-learning step. Frozen
        # gradients must not dominate clipping during subsequent fine-tuning.
        for name, parameter in model.named_parameters():
            if "measurement_fusion" in name:
                parameter.grad = torch.full_like(parameter, 1e12)
        groups = parameter_groups(model, 0.01, freeze_measurement_fusion=True)
        self.assertTrue(groups)
        self.assertTrue(all(group["params"] for group in groups))
        rates = {id(parameter): group["lr"] for group in groups for parameter in group["params"]}
        for name, parameter in model.named_parameters():
            if "measurement_fusion" in name:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
                self.assertNotIn(id(parameter), rates)
            elif "input_projection" in name:
                self.assertAlmostEqual(rates[id(parameter)], 0.01 * 0.05)
            elif "s4_layers" in name:
                self.assertAlmostEqual(rates[id(parameter)], 0.01 * 0.10)
            elif "control_head" in name:
                self.assertAlmostEqual(rates[id(parameter)], 0.01)

        finetune(model, batch, SyntheticComplexMSELoss(), epochs=2, batch_size=3, lr=0.01, seed=3)
        after = dict(model.named_parameters())
        for name, previous in before.items():
            if "measurement_fusion" in name:
                self.assertTrue(torch.equal(previous, after[name].detach()), name)
        self.assertTrue(any(
            not torch.equal(previous, after[name].detach())
            for name, previous in before.items() if "control_head" in name
        ))
        self.assertFalse(model.training)

    def test_s4_finite_forward_gradients_and_exact_checkpoint_roundtrip(self):
        # An odd model width also covers the positional-encoding cosine edge.
        config = ModelConfig(input_dim=3, d_model=7, d_state=4, n_layers=1, output_dim=2, dropout=0.0)
        model = QuantumS4Model(**config.to_dict())
        batch = make_synthetic_episodes(
            n_tasks=1, support_size=3, query_size=2, seq_len=5,
            input_dim=3, output_dim=2, seed=11,
        )[0].support
        output = model(batch.inputs, batch.measurements)
        self.assertEqual(tuple(output.shape), (3, 2))
        self.assertTrue(output.is_complex())
        self.assertTrue(bool(torch.isfinite(output).all()))
        loss, _ = SyntheticComplexMSELoss()(output, batch.targets)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

        model.eval()
        with torch.no_grad():
            expected = model(batch.inputs, batch.measurements)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_checkpoint.pth"
            save_checkpoint(path, model, config, metadata={"sample_kind": "synthetic"})
            restored, restored_config, metadata = load_checkpoint(path)
            self.assertEqual(restored_config, config)
            self.assertEqual(metadata, {"sample_kind": "synthetic"})
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor.cpu(), restored.state_dict()[name]), name)
            with torch.no_grad():
                actual = restored(batch.inputs, batch.measurements)
            self.assertTrue(torch.equal(expected, actual))

    def test_synthetic_domain_shift_keeps_a_shared_learnable_teacher(self):
        settings = dict(n_tasks=2, support_size=32, query_size=16, seq_len=5, input_dim=2, output_dim=2, seed=123)
        source = make_synthetic_episodes(**settings)
        shifted = make_synthetic_episodes(**settings, domain_shift=0.35)
        repeated = make_synthetic_episodes(**settings)

        def features(batch):
            values = torch.cat((
                batch.inputs.mean(dim=1), batch.inputs[:, -1],
                batch.measurements.mean(dim=1, keepdim=True),
                torch.ones(len(batch), 1),
            ), dim=1)
            return values.to(torch.float64)

        def latent_targets(batch):
            values = torch.cat((batch.targets.real, batch.targets.imag), dim=1).to(torch.float64)
            return torch.atanh(2.0 * values)

        for original, target, duplicate in zip(source, shifted, repeated):
            self.assertTrue(torch.equal(original.support.targets, duplicate.support.targets))
            torch.testing.assert_close(
                target.support.inputs - original.support.inputs,
                torch.full_like(original.support.inputs, 0.35), atol=2e-7, rtol=0,
            )
            # Fit only source support data, then predict held-out source and shifted
            # queries. Independent random labels or a new target teacher fail this.
            fitted_teacher = torch.linalg.lstsq(
                features(original.support), latent_targets(original.support),
            ).solution
            for query in (original.query, target.query):
                predicted = 0.5 * torch.tanh(features(query) @ fitted_teacher)
                expected = torch.cat((query.targets.real, query.targets.imag), dim=1).to(torch.float64)
                torch.testing.assert_close(predicted, expected, atol=2e-6, rtol=2e-5)

    def test_json_reader_validates_schema_and_does_not_invent_metadata(self):
        task = {
            "input_data": [[1.0, 2.0], [3.0, 4.0]],
            "measurement_sequence": [0.25, 0.5],
            "target_output": [{"real": 0.3, "imag": -0.2}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"

            def write(payload):
                path.write_text(json.dumps(payload), encoding="utf-8")

            write([task])
            batch = read_task_batch(path, max_seq_len=3, output_dim=2)
            self.assertEqual(tuple(batch.inputs.shape), (1, 3, 2))
            self.assertEqual(tuple(batch.targets.shape), (1, 2))
            self.assertTrue(torch.equal(batch.inputs[0, -1], torch.zeros(2)))
            self.assertEqual(batch.targets[0, 1], 0j)
            self.assertIsNone(batch.metadata)
            annotated = deepcopy(task)
            annotated["sample_metadata"] = {"sample_id": 19}
            write({"tasks": [annotated]})
            annotated_batch = read_task_batch(path, max_seq_len=1, output_dim=1)
            self.assertEqual(annotated_batch.metadata, [{"sample_id": 19}])
            self.assertEqual(tuple(annotated_batch.inputs.shape), (1, 1, 2))

            malformed = []
            for field, bad_value in (
                ("input_data", [[1.0, 2.0], [3.0]]),
                ("input_data", [[True, 2.0], [3.0, 4.0]]),
                ("measurement_sequence", [0.25]),
                ("measurement_sequence", [float("nan"), 0.5]),
                ("target_output", [{"real": 0.3}]),
                ("sample_metadata", []),
            ):
                invalid = deepcopy(task)
                invalid[field] = bad_value
                malformed.append(invalid)
            for invalid in malformed:
                with self.subTest(invalid=invalid):
                    write([invalid])
                    with self.assertRaises(ValueError):
                        read_task_batch(path)
            write({"tasks": []})
            with self.assertRaises(ValueError):
                read_task_batch(path)


if __name__ == "__main__":
    unittest.main()
