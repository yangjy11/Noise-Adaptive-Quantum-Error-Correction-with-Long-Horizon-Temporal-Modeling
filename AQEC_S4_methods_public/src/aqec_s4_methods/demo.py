"""Standalone synthetic demonstration; results are not physical benchmarks."""

import argparse
from pathlib import Path

import torch

from .checkpoints import load_checkpoint, save_checkpoint
from .config import ModelConfig
from .data import make_synthetic_episodes
from .losses import SyntheticComplexMSELoss
from .meta import FOMAMLTrainer
from .model import QuantumS4Model
from .transfer import finetune
from .utils import set_seed, write_json


def evaluate(model, batch, loss_fn, device):
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        prediction = model(batch.inputs, batch.measurements)
        loss, _ = loss_fn(prediction, batch.targets, batch.measurements, batch.metadata)
    return float(loss.item())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/synthetic_demo"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meta-epochs", type=int, default=8)
    parser.add_argument("--transfer-epochs", type=int, default=10)
    parser.add_argument("--tasks", type=int, default=6)
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if min(args.meta_epochs, args.transfer_epochs, args.tasks, args.inner_steps, args.threads) < 1:
        parser.error("epochs, tasks, inner steps, and threads must all be positive")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    config = ModelConfig(d_model=16, d_state=64, n_layers=1, output_dim=4, dropout=0.0)
    common = dict(n_tasks=args.tasks + 2, support_size=8, query_size=8, seq_len=12,
                  input_dim=config.input_dim, output_dim=config.output_dim, seed=args.seed)
    source = make_synthetic_episodes(**common, domain_shift=0.0)
    target = make_synthetic_episodes(**common, domain_shift=0.25)
    loss_fn = SyntheticComplexMSELoss()
    model = QuantumS4Model(**config.to_dict()).to(args.device)
    source_before = evaluate(model, source[-1].query, loss_fn, args.device)
    trainer = FOMAMLTrainer(model, loss_fn, inner_lr=0.03, meta_lr=0.003,
                            inner_steps=args.inner_steps, adaptive_inner=True, device=args.device)
    meta_history = trainer.fit(source[:args.tasks], epochs=args.meta_epochs,
                               episodes_per_epoch=min(3, args.tasks), seed=args.seed)
    source_after = evaluate(model, source[-1].query, loss_fn, args.device)
    source_path = save_checkpoint(args.output_dir / "source_model.pt", model, config,
                                 metadata={"task": "synthetic", "stage": "meta", "seed": args.seed})
    restored, restored_config, _ = load_checkpoint(source_path, device=args.device)
    probe = source[-1].query.to(args.device)
    model.eval()
    with torch.no_grad():
        roundtrip_error = float((model(probe.inputs, probe.measurements)
                                 - restored(probe.inputs, probe.measurements)).abs().max().item())
    target_before = evaluate(restored, target[-1].query, loss_fn, args.device)
    transfer_history = finetune(restored, target[-1].support, loss_fn,
                                epochs=args.transfer_epochs, batch_size=4, lr=0.01,
                                freeze_measurement_fusion=True, device=args.device, seed=args.seed)
    target_after = evaluate(restored, target[-1].query, loss_fn, args.device)
    save_checkpoint(args.output_dir / "transferred_model.pt", restored, restored_config,
                    metadata={"task": "synthetic", "stage": "transfer", "seed": args.seed})
    summary = {
        "scope": "Synthetic sequence regression; no physical performance claims",
        "seed": args.seed, "device": args.device, "model_config": config.to_dict(),
        "source_heldout_query_loss_before": source_before,
        "source_heldout_query_loss_after": source_after,
        "target_query_loss_before": target_before,
        "target_query_loss_after": target_after,
        "checkpoint_roundtrip_max_abs_error": roundtrip_error,
        "meta_history": meta_history, "transfer_history": transfer_history,
    }
    write_json(args.output_dir / "training_summary.json", summary)
    print(f"Source held-out query MSE: {source_before:.6f} -> {source_after:.6f}")
    print(f"Target query MSE:          {target_before:.6f} -> {target_after:.6f}")
    print(f"Checkpoint round-trip error: {roundtrip_error:.3g}")
    print(f"Saved demo artifacts to {args.output_dir.resolve()}")
    return summary


if __name__ == "__main__":
    main()
