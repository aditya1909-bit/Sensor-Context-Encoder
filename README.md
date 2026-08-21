# Sensor Context Encoder

This repository compares a compact inertial-sensor classifier with a model that inserts one learned, continuous sensor embedding directly into a frozen language model. It uses only the nine windowed inertial signals from UCI HAR; the supplied 561 engineered features are never loaded.

See [`RESULTS.md`](RESULTS.md) for the completed comparison, ablations, verification checklist, and reproduction commands.

## Setup

The reference environment is Python 3.11 on an Apple M3 Pro. Create an isolated environment so the pinned scientific packages do not conflict with system packages:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

SmolLM2 is downloaded from Hugging Face during context-model construction. The model ID and immutable revision are saved in each context checkpoint.

## Reproduce the experiment

Download and verify the official nested UCI archive:

```bash
python download_data.py --data-dir data
```

Train the two models separately with the same seed, splits, encoder architecture, effective batch size, optimizer, and validation criterion:

```bash
python train.py --model direct --device mps --seed 42 --output-dir runs/direct
python train.py --model context --device mps --seed 42 --output-dir runs/context
```

The lean context model is the primary path: its selected checkpoint achieved the strongest validation and test performance. The six MPS-safe representation objectives remain available as a research ablation without changing the inference architecture:

```bash
python train.py \
  --model context \
  --with-context-objectives \
  --device mps \
  --seed 42 \
  --output-dir runs/context-objectives
```

Use `runs/context/best.pt` as the context checkpoint in the primary comparison. See [`CONTEXT_OBJECTIVES.md`](CONTEXT_OBJECTIVES.md) for the six objectives, source papers, fixed weights, expected benefits, and risks.

When objectives are enabled, their weights ramp linearly over the first three epochs, allowing the classifier to establish a supervised decision surface before the representation constraints reach full strength. Use `--objective-warmup-epochs N` to change that schedule; it is stored in each resumable checkpoint.

If the 16-example context batch exceeds MPS memory, preserve effective batch 128 with:

```bash
python train.py --model context --device mps --seed 42 --batch-size 8 --accumulation-steps 16 --output-dir runs/context
```

The same fallback retains the lean primary context model; add `--with-context-objectives` only for the objective ablation. Confirm that the normal Terminal session exposes MPS before starting:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

The command must print `True`. The Codex execution sandbox may not expose MPS even on an Apple-silicon host; run the full experiment from the normal macOS Terminal.

## Checkpoints, ETA, and resuming

Every run writes two atomic checkpoints:

- `latest.pt` contains trainable weights, optimizer and scheduler state, Python/NumPy/PyTorch/MPS RNG state, the deterministic epoch shuffle position, partial-epoch metrics, early-stopping state, and elapsed time. Context runs update it after every optimizer step by default; with accumulation 8, at most seven completed microbatches are replayed after an abrupt stop.
- `best.pt` contains the same resumable state at the best validation macro-F1.

The live progress bar shows the current epoch, batch rate, loss, and ETA to all configured epochs. Early stopping can make the actual run shorter. Machine-readable progress is atomically updated at `runs/context/progress.json`; inspect it at any time with:

```bash
cat runs/context/progress.json
```

Resume without repeating the original settings:

```bash
python train.py --resume runs/context/latest.pt --device mps
```

The saved model, data path, seed, context-objective setting, batch sizes, optimizer settings, epoch limit, and patience are restored automatically. Use `--checkpoint-every-steps N` to change checkpoint frequency; the defaults are every optimizer step for context training and every 25 optimizer steps for the fast direct baseline. Checkpoint replacement is atomic, so interruption cannot leave a partially written `latest.pt`.

Evaluate all three required conditions only after both validation-selected checkpoints exist:

```bash
python evaluate.py \
  --direct-checkpoint runs/direct/best.pt \
  --context-checkpoint runs/context/best.pt \
  --device mps \
  --shuffle-seed 43 \
  --output-dir results
```

This writes `results/results.json`, `results/results.csv`, and `results/results.md`. Copy the measured table and recommendation into `TECHNICAL_NOTE.md`; do not report smoke-run output.

## Data and comparison controls

- Input shape is `[128, 9]` in this order: total acceleration x/y/z, body acceleration x/y/z, and body gyroscope x/y/z.
- Official test subjects are 2, 4, 9, 10, 12, 13, 18, 20, and 24.
- Validation subjects are 1, 3, 15, 25, and 27, selected once from the official training subjects with seed 42. The resulting counts are 5,551 train, 1,801 validation, and 2,947 test.
- Channel normalization is fitted on the 5,551 training windows only.
- Both models use the same 1.09M-parameter Conv1D sensor encoder and select the best checkpoint by validation macro-F1.
- The context model adds a 256→512→960 projector and a 960→6 head, for 1.72M trainable parameters. All SmolLM2 parameters, including its embedding table, remain frozen.
- The optional context-objective ablation adds no parameters or inference operations. It combines physics-aware SO(3) views, VISReg, supervised contrastive structure, frozen text-prototype alignment, relational semantic distillation, and frozen token-manifold matching.
- The projected `[960]` rows are cached and deranged across the entire test set with seed 43. No row has a fixed position, no values within a row are rearranged, and no retraining occurs.

The continuation recommendation requires both:

```text
context macro-F1 >= direct macro-F1 - 0.05
context macro-F1 - shuffled macro-F1 >= 0.20
```

## Checks and smoke runs

Run the fast suite, which uses a tiny frozen stand-in for the language model:

```bash
python -m pytest -q
```

Run the pinned real-backbone gradient check once before full context training:

```bash
RUN_REAL_MODEL_TESTS=1 python -m pytest -q -m integration tests/test_real_backbone.py
```

Short pipeline checks are available but their metrics are not reportable:

```bash
python train.py --model direct --device mps --epochs 1 --max-batches 2 --output-dir runs/smoke-direct
python train.py --model context --device mps --epochs 1 --max-batches 2 --output-dir runs/smoke-context
python train.py --model context --with-context-objectives --device mps --epochs 1 --max-batches 2 --output-dir runs/smoke-context-objectives
```

The backbone is kept in evaluation mode and is not wrapped in `no_grad` during context training. This allows gradients to reach the projected sensor embedding while startup checks reject any backbone gradient or missing trainable-module gradient.

## Outputs and reproducibility

Checkpoints contain trainable weights, full resumable training state, normalization statistics, split metadata, seed, epoch history, dependency/runtime versions, parameter counts, and the pinned model revision. Context checkpoints intentionally omit the frozen language-model weights. Data, model caches, checkpoints, and generated result files are ignored by Git.

MPS runs use explicit seeds but are not guaranteed to be bit-identical across macOS or PyTorch releases. Use the pinned environment and retain the checkpoint metadata for a reproducible experimental run.
