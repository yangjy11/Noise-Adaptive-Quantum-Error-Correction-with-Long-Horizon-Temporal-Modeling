# Validation record

Verified on 2026-10-19 using Windows, Python 3.13.3, PyTorch 2.7.0+cu118,
NumPy 2.2.5, and CPU execution with one Torch thread. Other supported dependency
versions and accelerator execution have not been exercised in this extraction.

## End-to-end source demonstration

```bash
python examples/train_synthetic.py
```

Default seed 42, eight meta updates and ten transfer epochs completed. The model
checkpoint was saved, loaded with strict state-dictionary checking, and used for
target fine-tuning. Source and target evaluation query samples were held out
from their respective optimization stages.

| Synthetic MSE | Before | After |
| --- | ---: | ---: |
| Source held-out query | 0.333488 | 0.233666 |
| Target query | 0.247795 | 0.192352 |

Maximum output difference immediately after save/reload was **0**. These values
are a software demonstration on synthetic tasks and do not measure physical
error-correction performance. Different environments or hyperparameters may
produce different optimization trajectories.

## Extracted objective and distribution build

- An additional two-episode FOMAML step using `FeedbackGRAPELoss` completed with
  finite loss 0.427459 and query gradients for 38 parameter tensors. Direct
  fine-tuning using that same objective also completed with finite loss.
- Importing and training did not load the original project's private modules.
- The source network and objective were statically compared with the supplied
  implementation; mathematical extraction differences are listed in CHANGES.md.
- A Python wheel was built without dependency downloads. Importing the library
  from the extracted wheel and running a short demo succeeded, with zero
  checkpoint round-trip output difference.

## Regression tests

`python -m unittest discover -s tests -v` completed: **6 tests passed**
(14.520 seconds, after interpreter startup). The checks cover:

1. First-order meta-gradients against an independent analytic calculation over
   two tasks, including a distinction from the second-order gradient.
2. External JSON schema validation, padding, and optional metadata.
3. Finite S4 forward/backward values and exact checkpoint output round trips.
4. A shared, learnable synthetic teacher across the source/target input shift.
5. Frozen fusion weights, cleared stale gradients, and updated output-head weights.
6. Shuffled sample/metadata alignment and inclusion of the short final batch.

## Release scope

The release archive uses an explicit source-file allowlist. It excludes generated
checkpoints, caches, build output, real datasets, private project history, and
the simulation environment. Source and documentation are scanned for excluded
state-construction markers, private imports, machine paths, and credential-like
strings. `MANIFEST.sha256` lists the released file hashes, excluding itself.
