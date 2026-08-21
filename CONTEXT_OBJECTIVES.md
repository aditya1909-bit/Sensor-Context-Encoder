# Context Training Objectives

The optional context-objective ablation combines six auxiliary objectives while preserving the required architecture, one continuous sensor embedding, frozen language model, parameter limit, subject splits, and test protocol. The direct classifier remains an uncontaminated engineering baseline. These methods are research-inspired hypotheses, not guaranteed improvements; the matched-versus-shuffled test and predefined continuation gates remain decisive. The default context run uses the cleaner cross-entropy objective because it achieved the best validation score in the completed comparison.

## 1. Physics-aware SO(3) multi-view learning

Each training window produces two stochastic views. One proper 3D rotation is applied consistently to the total-acceleration, body-acceleration, and gyroscope xyz triads, followed by mild triad scaling, Gaussian jitter, and a contiguous 16-step temporal mask. A shared rotation preserves vector magnitudes and physical relationships while discouraging dependence on one phone orientation. The masks adapt masked time-series representation learning ideas such as [Ti-MAE](https://arxiv.org/abs/2301.08871) without adding a reconstruction decoder or inference cost.

Trade-off: UCI HAR uses a standardized waist placement, so large rotations would erase useful orientation cues. The rotation is limited to ±15°.

## 2. VISReg sliced-Wasserstein regularization

[VISReg](https://arxiv.org/abs/2606.02572) is a 2026 successor to covariance-only representation regularization. The two augmented sensor representations are trained for invariance while scale and centering terms prevent collapse. Sixty-four random one-dimensional projections are sorted and matched to Gaussian quantiles, controlling the full projected distribution shape with an MPS-safe `O(NDK)` operation.

Trade-off: physical context batches contain 16 examples, so empirical quantiles are noisy. VISReg therefore uses a light 0.02 weight alongside the supervised loss.

## 3. Supervised contrastive sensor geometry

The [supervised contrastive](https://arxiv.org/abs/2004.11362) objective pulls same-activity sensor representations together and separates other activities. Recent work such as [RankSCL](https://arxiv.org/abs/2401.18057) applies class-aware contrastive structure specifically to time series. Anchors without another same-class example in the current batch are excluded safely.

Trade-off: some random batches contain few positive pairs. Cross-entropy remains the main objective and the contrastive term uses weight 0.10.

## 4. Frozen text-prototype alignment

The frozen SmolLM2 embedding table creates one prototype for each activity phrase. A temperature-scaled contrastive loss aligns the projected sensor vector with the correct prototype before it enters the language model. This directly follows the time-series-to-LLM motivation of [TEST](https://arxiv.org/abs/2308.08241): learn time-series embeddings that inhabit a language model-compatible space while keeping the LLM frozen.

Trade-off: an input-embedding average is a compact semantic target rather than a full sentence representation. It remains auxiliary to the frozen-LLM classification loss.

## 5. Relational semantic-geometry distillation

Pointwise prototype alignment does not preserve relationships among classes. The relational objective matches pairwise cosine similarities between projected sensor contexts to similarities among their frozen activity-text prototypes. This is a cross-modal adaptation of [relational knowledge distillation](https://arxiv.org/abs/1904.05068): transfer geometry rather than exact features.

Trade-off: lexical and inertial similarity do not always agree—for example, walking upstairs and downstairs may be close linguistically but distinct in the sensor data. The smooth-L1 term therefore uses a 0.05 weight.

## 6. Frozen token-manifold matching

A project-specific distribution regularizer matches the projected contexts' per-dimension mean, standard deviation, and average vector norm to statistics of the frozen SmolLM2 vocabulary embeddings. This tests a stronger version of “continuous token compatibility”: contexts should not merely have dimension 960, but should enter the backbone at a plausible embedding scale and location.

Trade-off: vocabulary embeddings are a useful prior, not a sensor-specific one. The term uses a 0.01 weight and its raw value is logged independently.

## Compute and interpretation

All six methods use native PyTorch operations and add no trainable parameters or inference work. SmolLM2 still runs once per batch; the two augmented views are concatenated so the small encoder handles them together. Their combined loss ramps to full weight across the first three epochs. Checkpoint history records every raw auxiliary loss and evaluation records the enabled methods. Isolating the contribution of individual objectives would require dedicated ablations.
