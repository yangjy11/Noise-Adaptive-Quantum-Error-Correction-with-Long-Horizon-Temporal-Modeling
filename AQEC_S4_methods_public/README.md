# AQEC-S4 learning methods

A standalone extraction of the AQEC-S4 sequence network, first-order meta-learning
(FOMAML), and groupwise transfer fine-tuning. The network follows the supplied
research implementation, including its measurement fusion, FFT-based temporal
layer, and complex output head. It is not an upstream S4 reference implementation.

The package contains no physical state-space construction, environment simulator,
feedback execution, real training datasets, or pretrained weights. The included
example learns synthetic sequence-regression tasks. Its losses are software
demonstrations, not quantum error-correction performance measurements or a
reproduction of the paper's numerical results.

## Quick start

Python 3.10+ is required. Install PyTorch for your platform and NumPy, or install
the package and its declared dependencies:

```bash
python -m pip install -e .
python examples/train_synthetic.py
```

If PyTorch and NumPy are already installed, the example also works directly from
this source checkout without installation:

```bash
python examples/train_synthetic.py --meta-epochs 8 --transfer-epochs 10
```

After installation, `aqec-s4-demo` and `python -m aqec_s4_methods.demo` run the same
example. It defaults to CPU and one Torch thread; use `--device cuda` when desired.
For a short smoke run:

```bash
python examples/train_synthetic.py --meta-epochs 2 --transfer-epochs 2 --output-dir artifacts/smoke
```

The example trains on source episodes, saves and reloads the model, then fine-tunes
on the support split of a shifted target task and evaluates its untouched query
split. Source query evaluation uses an episode excluded from meta-training.
Source and target share a synthetic teacher; only the input distribution shifts.
Outputs are `source_model.pt`, `transferred_model.pt`, and
`training_summary.json` in `artifacts/synthetic_demo/`. These generated artifacts
are ignored by Git and are not supplied in the release archive.

## Modules

| Module | Public content |
| --- | --- |
| `model.py` | `hippo_initialization`, `S4Layer`, `S4Block`, `QuantumS4Model` |
| `meta.py` | Adaptive inner learning rate, detached support adaptation, averaged first-order query-gradient update |
| `transfer.py` | Optional measurement-fusion freezing, grouped AdamW fine-tuning, cosine scheduling |
| `losses.py` | Extracted `FeedbackGRAPELoss` and independent `SyntheticComplexMSELoss` |
| `data.py` | `TaskBatch`, `Episode`, synthetic tasks, validated external JSON adapter |
| `checkpoints.py` | Configuration and tensor state-dictionary save/load |
| `config.py` | Serializable `ModelConfig` |

## Training API

```python
from aqec_s4_methods import (
    ModelConfig, QuantumS4Model, FOMAMLTrainer, SyntheticComplexMSELoss,
    make_synthetic_episodes, finetune, save_checkpoint, load_checkpoint,
)
from aqec_s4_methods.utils import set_seed

set_seed(42)
config = ModelConfig(d_model=16, n_layers=1, output_dim=4, dropout=0.0)
model = QuantumS4Model(**config.to_dict())
episodes = make_synthetic_episodes(seed=42, output_dim=4)
loss_fn = SyntheticComplexMSELoss()
trainer = FOMAMLTrainer(model, loss_fn, inner_steps=2)
history = trainer.fit(episodes, epochs=3, seed=42)
save_checkpoint("artifacts/source.pt", model, config)

target_model, target_config, _ = load_checkpoint("artifacts/source.pt")
shifted = make_synthetic_episodes(seed=42, output_dim=4, domain_shift=0.25)
transfer_history = finetune(target_model, shifted[0].support, loss_fn,
                            epochs=3, batch_size=4, lr=0.01)
```

`TaskBatch` holds `inputs: [B,L,input_dim]`, `measurements: [B,L]`, complex
`targets: [B,output_dim]`, and optional per-sample `metadata`. An `Episode` has
separate `support` and `query` batches. `TaskBatch.select()` indexes the tensors
and metadata together. External input, measurement, and target generation remains
the caller's responsibility.

`read_task_batch(path, max_seq_len=50, output_dim=20)` accepts a list of records,
or an object with a `tasks` list. Each record has `input_data`,
`measurement_sequence`, and `target_output` (a list of `{real, imag}` values).
Optional `sample_metadata` is passed through, not synthesized by this library.
The loader only packs existing records; it does not create train/query splits.

Every loss callback uses this interface:

```python
loss, score = loss_fn(predictions, targets, measurements, metadata)
```

`loss` must be a differentiable real scalar tensor. `score` is a finite scalar
diagnostic, not necessarily a physical fidelity. The synthetic example uses
complex MSE and an overlap diagnostic.

The extracted `FeedbackGRAPELoss` preserves the original algebra. Its leakage
and energy penalties are proxies computed from predicted outputs, not an
environment rollout. The compatibility flag `use_grape_gradient` does not enable
a separate physical GRAPE implementation. This loss selects batch settings from
the first metadata record: use homogeneous batches when these settings matter,
or supply your own per-sample loss. Its scalar metadata fields do not require
the private environment implementation.

The source meta-learning initializer overrode the standalone loss defaults. To
retain those objective settings, construct the callback explicitly:

```python
from aqec_s4_methods import FeedbackGRAPELoss

meta_loss = FeedbackGRAPELoss(
    lambda_control=0.0005, lambda_smooth=0.00005, lambda_feedback=0.005,
    lambda_leakage=0.008, lambda_energy=0.004,
)
# The source transfer script used FeedbackGRAPELoss() with constructor defaults.
```

This does not recreate the omitted data-generation or evaluation protocol.

Transfer learning-rate multipliers are 0.05 for input projection, 0.10 for the
temporal backbone, 1.00 for the output head, and 0.20 for other parameters.
`freeze_measurement_fusion=False` preserves existing trainability flags; it does
not unfreeze parameters previously frozen by a caller. Checkpoints save model
weights and configuration, not optimizer state or an exact training-resume state.

## Validation and provenance

```bash
python -m unittest discover -s tests -v
```

The tests check an independently calculated first-order meta-gradient, actual
parameter updates, shuffled metadata alignment, frozen-layer behavior, model
gradients, checkpoint round trips, and synthetic/JSON data interfaces. See
[CHANGES.md](CHANGES.md) for differences from the original training scripts and
[VALIDATION.md](VALIDATION.md) for the verified environment and run results.

Random seeds are explicit; constructors no longer reset global seeds. The
original minimum effective temporal state size of 64 is retained. All examples
and tests are independent of the private project.

## Citation and licensing

Fill in the actual paper citation, authors, and repository URL when publishing.
See [LICENSE_NOTICE.md](LICENSE_NOTICE.md): this extraction does not assign a
software license on behalf of the copyright holders.
