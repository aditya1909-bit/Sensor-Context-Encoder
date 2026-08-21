# Continuous Sensor Context for a Frozen Language Model

## Objective and design

This experiment maps each 128×9 inertial window into one continuous vector in SmolLM2's 960-dimensional embedding space. Both models start from the same 1.09M-parameter residual Conv1D encoder. The direct model uses a six-class linear head. The context model adds a 256→512→960 projector, inserts one sensor vector between the fixed prompt prefix and `Activity:`, and classifies the final suffix state. It has 1.72M trainable parameters; the 360M-parameter language model stays frozen.

SmolLM2, including its token embeddings, remains frozen in evaluation mode. Gradients flow through the backbone to the sensor encoder, projector, and classifier head. Checkpoints contain the trainable components and the pinned backbone revision.

The primary context run uses cross-entropy alone. We also evaluated six auxiliary objectives: physics-aware SO(3) views, VISReg, supervised contrastive learning, frozen activity-text alignment, relational semantic distillation, and token-manifold matching. They add two small encoder passes during training, not another language-model pass or any inference cost. The lean model won on validation, so it is the primary result. `CONTEXT_OBJECTIVES.md` documents the ablation in full.

## Data and fairness controls

Only the nine files in each UCI HAR `Inertial Signals` directory are loaded; the 561-dimensional engineered features are excluded. The official subject-wise test split is preserved. Subjects 1, 3, 15, 25, and 27 form a fixed validation partition within the official training split, leaving 5,551 training, 1,801 validation, and 2,947 test windows. Per-channel normalization is fitted on training windows only.

Both models use seed 42, identical initial encoder weights, effective batch size 128, AdamW, cosine decay, and validation macro-F1 for checkpoint selection. The official test split remains untouched until final evaluation. The context path uses smaller physical batches to fit the frozen backbone on MPS.

## Results

| Condition | Macro-F1 | Seed |
|---|---:|---:|
| Direct sensor classifier | 0.9285 | 42 |
| Lean context-embedding model | 0.9320 | 42 |
| Lean context model with shuffled embeddings | 0.1615 | 43 |
| Context-objective ablation | 0.9012 | 42 |
| Context-objective ablation with warm-up | 0.9314 | 42 |

On MPS, the direct model takes 2.06 ms per example and the lean context model takes 59.63 ms. The main remaining error is sitting versus standing: 81 sitting windows are predicted as standing and 45 standing windows as sitting. `results/results.json` contains the full per-class F1 scores and confusion matrix; `results-frontier/` and `results-improved/` retain the ablations.

## Sensor-dependence check

The projector produces one 960-value embedding per test example. Evaluation caches those rows, creates a seed-43 derangement with no fixed examples, and scores the shuffled rows against the original labels. Values inside a row are never rearranged. Performance falls from 0.9320 to 0.1615, showing that the model depends on the matching sensor window rather than the fixed prompt alone.

## Interpretation and recommendation

The continuation gates are fixed in advance: context macro-F1 must be within 0.05 of the direct classifier, and matched context macro-F1 must exceed shuffled-context macro-F1 by at least 0.20.

Final recommendation: **continue**. The lean context model reaches 0.9320 macro-F1, beating the direct classifier's 0.9285. Its 0.7705 matched-to-shuffled gap clears the sensor-dependence gate by a wide margin. The warm-up ablation reaches 0.9314, while the original objective stack reaches 0.9012. The result is straightforward: a frozen language model can consume a learned, sensor-dependent continuous interface and match a strong direct baseline. The direct model is still the faster deployment option, at roughly 29× lower single-example latency.

This is one dataset, one sensor modality, one backbone, one prompt, and one seed. The next useful question is whether the same interface transfers across datasets and sensor placements.
