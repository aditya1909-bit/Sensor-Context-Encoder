# Results

## Headline result

The selected sensor-context model reached **0.9320 macro-F1** on the official UCI HAR test split. The direct sensor classifier reached **0.9285 macro-F1** under the same split, seed, encoder initialization, training schedule, and checkpoint-selection rule.

The context model passes both continuation gates:

| Gate | Result | Status |
|---|---:|---|
| Context parity: within 0.05 of direct | +0.0035 macro-F1 over direct | Pass |
| Sensor dependence: matched minus shuffled ≥ 0.20 | +0.7705 macro-F1 | Pass |

The shuffled-control score is 0.1615 macro-F1. Shuffling complete 960-dimensional context rows across the test set removes the performance gain, confirming that the frozen language model uses the sensor embedding associated with each example.

## Primary comparison

| Condition | Test macro-F1 | Accuracy | Batch-one MPS latency | Seed |
|---|---:|---:|---:|---:|
| Direct sensor classifier | 0.9285 | 0.9277 | 2.06 ms | 42 |
| Lean sensor-context model | 0.9320 | 0.9325 | 59.63 ms | 42 |
| Lean context with shuffled sensor embeddings | 0.1615 | 0.1615 | 59.63 ms | 43 |

The context model improves macro-F1 by 0.0035, while the direct model remains the low-latency deployment option. The main remaining context-model error is sitting versus standing.

## Ablations

| Context training variant | Test macro-F1 | Validation macro-F1 at selection |
|---|---:|---:|
| Lean cross-entropy model (selected) | 0.9320 | 0.9962 |
| Six-objective stack | 0.9012 | 0.9932 |
| Six-objective stack with three-epoch warm-up | 0.9314 | 0.9896 |

The lean model was selected by validation macro-F1 before final test evaluation. The objective-stack results remain useful: they show that the continuous sensor interface works without relying on a particular regularization recipe.

## What was verified

- Only the nine raw UCI HAR inertial-signal files are loaded; the 561 engineered features are excluded.
- Train, validation, and official test subjects are disjoint. Validation uses subjects 1, 3, 15, 25, and 27; the official 2,947-window test split is preserved.
- Normalization uses training windows only.
- SmolLM2-360M-Instruct is pinned to revision `c15f933c73438218a2bc078446c513173cc4f06a` and fully frozen, including its embedding table.
- The context model inserts exactly one learned 960-dimensional sensor vector between the fixed prompt prefix and suffix.
- Checkpoints contain model, optimizer, scheduler, RNG, and progress state for resume; the frozen backbone reloads from the pinned revision.
- The test suite passes: 17 passed, 1 optional integration test skipped unless explicitly enabled.

## Reproduce

```bash
python download_data.py --data-dir data
python train.py --model direct --device mps --seed 42 --output-dir runs/direct
python train.py --model context --device mps --seed 42 --output-dir runs/context
python evaluate.py \
  --direct-checkpoint runs/direct/best.pt \
  --context-checkpoint runs/context/best.pt \
  --shuffle-seed 43 \
  --device mps \
  --output-dir results
```

The machine-readable evaluation output is `results/results.json`; `results/results.csv` and `results/results.md` provide the same primary table in portable formats. The implementation details and interpretation are in `TECHNICAL_NOTE.md`.
