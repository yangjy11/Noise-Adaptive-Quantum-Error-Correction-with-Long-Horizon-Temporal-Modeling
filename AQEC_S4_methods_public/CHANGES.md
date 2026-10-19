# Extraction and behavior changes

This release extracts learning code from the author-supplied research package.
It is a reusable methods library, not a snapshot of the full experimental system.

## Preserved

- Network structure and state-dictionary parameter names from the original
  `hippo_initialization`, `S4Layer`, `S4Block`, and `QuantumS4Model`.
- All mathematical terms in the original `FeedbackGRAPELoss`.
- Loss/gradient-dependent inner learning-rate multipliers and step decay.
- First-order support adaptation and explicit query-gradient averaging into the
  initialization; outer AdamW weight decay and gradient clipping.
- Transfer parameter-group ratios, optional fusion freezing, AdamW weight decay,
  cosine schedule, and gradient clipping.

## Refactored or corrected

- Model constructors no longer reset Torch's global seed. Call `set_seed()` at
  application startup. Odd-width positional encoding now trims the cosine slice.
- `FOMAMLTrainer` supports first-order learning only. It does not expose the
  original unverified higher-order option. Unused momentum settings are omitted.
- Loss and preconstructed tasks are explicit inputs. No automatic environment
  reconstruction, domain-specific default metadata, or private data is loaded.
- Inner steps are explicit and uniform across supplied tasks. The previous
  domain-dependent schedule and adaptive early-stop heuristic are not silently
  applied. Callers can select the desired inner-step count.
- `fit` performs one update per epoch on reproducibly sampled episodes. The
  original fixed-period outer schedule and training-score-based early stopping
  and best-weight selection are not included.
- Shuffled transfer tensors and metadata use the same indices. The final short
  minibatch is retained rather than dropped.
- Frozen fusion parameters have stale gradients cleared. Transfer gradient
  clipping includes only the optimizer's trainable parameters.
- Invalid inputs and nonfinite losses/gradients raise errors instead of silently
  skipping episodes. Unused parameters do not receive an accidental outer update.
- Checkpoints use a documented tensor/configuration format and restricted
  `weights_only=True` loading. Optimizer state is not stored.
- An independent synthetic teacher, demo, tests, and documentation were added.

## Research scope

The network naming follows the research implementation. No claim is made that
the temporal layer is interchangeable with all upstream S4 variants. The loss
name is retained for compatibility; its proxy penalties and compatibility flags
are described in its docstrings and README.

These extraction changes can change optimization trajectories. Reproducing a
paper result requires the corresponding data, experimental configuration,
training protocol, and evaluation system, which this release does not provide.
