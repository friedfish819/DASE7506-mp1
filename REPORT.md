Improving a Small GPT with Rotary Position Embeddings and Ensembling

DASE7506 MP1 — Small Language Model Challenge

**Final Test BPB: 1.5929**(baseline: 2.1013, relative improvement: −24.2%)

Abstract

We improve a 1.05M-parameter GPT trained from scratch on WikiText-2 through two mechanisms: replacing learned absolute position embeddings with rotary position embeddings (RoPE), and ensembling three models trained with different random seeds. We also report several negative results. At equal training budget (1,200 steps, 9,830,400 processed targets), RoPE alone reduces validation BPB from 2.0711 to 1.9223, while RMSNorm alone does not help. Increasing training to 4,800 steps and searching the peak learning rate further reduces validation BPB to 1.6467 for a single model. A 3-model ensemble reaches 1.5929 test BPB at 1.587 GiB peak RAM and 12.12 MiB of inference assets. EMA, cosine annealing to zero, and SwiGLU did not improve quality at this scale. The binding constraint is evaluation time: the ensemble scores in \~46 s on an idle CPU, close to the 5× baseline limit of \~51 s.

1\. Introduction

The baseline is a four-block GPT (width 128, four heads, context 256, 1,088,256 parameters) trained from random initialization on WikiText-2 with a fixed BPE-2048 tokenizer. It reaches approximately 2.10 test bits per byte (BPB).

We investigate two questions:

1\. Can a modern positional encoding improve a small GPT at equal training budget?

2\. How much can ensembling and training-budget tuning add on top?

We report the initial baseline, a same-budget comparison, an ablation of the key mechanism, several negative results, and the trade-offs between prediction quality and computational cost.

2\. Method

2.1 Baseline architecture

4 Transformer blocks, width 128, 4 attention heads, context 256. Pre-LayerNorm, GELU MLP with 4× expansion. Learned absolute position embeddings. Tied input/output embeddings. AdamW, peak LR 1e-3, 100-step warmup, cosine decay to 10% of peak. 1,200 steps × batch 32 × 256 targets = 9,830,400 processed targets.

2.2 Rotary position embeddings (RoPE)

RoPE replaces the learned absolute position embedding with a rotation applied to query and key vectors inside each attention head. For a head dimension d, each pair of dimensions (2i, 2i+1) is rotated by an angle proportional to the token position:

Properties: Relative position is encoded through rotation, not addition. No position parameters: 32,768 fewer parameters. Causality is preserved: the rotation depends only on position, not on future tokens. The model becomes stateless between windows, satisfying the evaluation contract. We precompute cos/sin tables once per model and slice them to the input length.

2.3 Ensembling

We train three models with identical configuration but seeds 17, 42, 123, and average their log-probabilities at inference:

This is a geometric mean of the per-model probabilities. Members share the same architecture and training recipe; diversity comes from random initialization and data ordering. The ensemble is evaluation-only: no additional training is performed.

2.4 Interface compliance

Both mechanisms preserve the required interface: build_model(config) returns a model with context=256. forward(ids) returns unnormalized logits [batch, time, 2048]. predict_log_probs(ids) returns finite, normalized natural-log probabilities. Predictions at position t use only the observed prefix through t. Temporary state is reset between independent windows. All five contract tests in tests/test_contract.py pass.

3\. Experimental setup

Data: supplied WikiText-2 raw splits, BPE-2048 tokenizer, 256-token causal windows. Protocol: 7506-mp1-wt2-v2. Evaluation: FP32, CPU, 4 threads. Training: AdamW (weight decay 0.1), gradient clipping at 1.0, batch size 32, 100-step warmup, cosine decay to 10% of peak unless stated. Development: all model selection used the validation split. The test split was evaluated only for the final frozen configuration and for intermediate reference points noted explicitly. Fairness: comparisons at equal processed-target count unless stated.

4\. Same-budget comparison

All four configurations were trained for 1,200 steps (9,830,400 processed targets) with the same optimizer and schedule.

| Configuration | Norm      | Position | Val BPB |
|---------------|-----------|----------|---------|
| Baseline      | LayerNorm | Learned  | 2.0711  |
| RMSNorm only  | RMSNorm   | Learned  | 2.0777  |
| RoPE only     | LayerNorm | RoPE     | 1.9223  |
| RMSNorm+RoPE  | RMSNorm   | RoPE     | 1.9219  |

Findings: RoPE alone improves validation BPB by −0.1488. RMSNorm alone slightly degrades BPB (+0.0066), within noise but not a gain. Adding RMSNorm to RoPE changes BPB by −0.0004, i.e. no further benefit. So RoPE is the effective mechanism at this scale. This table doubles as the same-budget comparison and the RoPE ablation: removing RoPE from the winning configuration costs +0.149 val BPB.

5\. Negative results

We tested three additional mechanisms on top of the RoPE configuration. None helped.

5.1 EMA of model weights

| Schedule      | EMA decay | Val BPB |
|---------------|-----------|---------|
| Cosine to 10% | 0 (off)   | 1.9223  |
| Cosine to 10% | 0.995     | 1.9356  |
| Cosine to 0   | 0.995     | 1.9541  |

The training loss curve of the EMA run matches the non-EMA run step-for-step, so the difference is entirely due to the averaged weights. At 1,200 steps the model is still improving, and EMA with a \~200-step effective window pulls in earlier, worse weights.

5.2 Cosine annealing to zero

Annealing the learning rate to exactly zero leaves the last tens of steps with near-zero updates. At this short budget the model has not converged, so this discards useful training signal. Cosine to 10% of peak is better.

5.3 SwiGLU MLP

| MLP                  | Steps | Val BPB |
|----------------------|-------|---------|
| GELU 4×              | 4,800 | 1.6930  |
| SwiGLU (8/3 × width) | 4,800 | 1.6976  |

Hidden size was chosen as 8/3×width to roughly match the parameter count of the GELU 4× MLP (1,052,416 vs 1,055,488). The difference is within seed noise (±0.005). SwiGLU's typical benefits appear in larger models or longer training.

Interpretation: at this model scale and training budget, the simple baseline components (LayerNorm, GELU, final LR floor at 10%) are already appropriate.

6\. Training budget

We trained the RoPE-only model (LayerNorm, GELU) at increasing step counts, all at LR 0.001.

| Steps | Processed targets | Val BPB |
|-------|-------------------|---------|
| 1,200 | 9,830,400         | 1.9223  |
| 2,400 | 19,660,800        | 1.7786  |
| 3,600 | 29,491,200        | 1.7206  |
| 4,800 | 39,321,600        | 1.6930  |

Gains per doubling:

| Interval      | Val BPB gain |
|---------------|--------------|
| 1,200 → 2,400 | -0.1437      |
| 2,400 → 3,600 | -0.0580      |
| 3,600 → 4,800 | -0.0276      |

Diminishing returns: each doubling roughly halves the previous gain. The model is not fully saturated at 4,800 steps, but the marginal return is small relative to training cost.

Validation–test gap is stable at \~+0.027 across step counts, indicating no overfitting to validation.

7\. Learning-rate search

We searched the peak learning rate at 1,200 steps first, then confirmed the best candidates at 4,800 steps.

7.1 1,200 steps

| LR     | Val BPB |
|--------|---------|
| 0.0005 | 2.0594  |
| 0.001  | 1.9223  |
| 0.002  | 1.8363  |
| 0.003  | 1.8051  |
| 0.004  | 1.7930  |
| 0.005  | 1.7847  |
| 0.006  | 1.7870  |
| 0.008  | 1.7817  |
| 0.01   | 1.7827  |
| 0.012  | 1.7846  |

Peak at 1,200 steps is around 0.005 to 0.01, with differences below 0.005.

7.2 4,800 steps

| LR    | Val BPB |
|-------|---------|
| 0.001 | 1.6930  |
| 0.003 | 1.6769  |
| 0.005 | 1.6467  |
| 0.008 | 1.6776  |

The optimal LR decreases with longer training: 0.008 is best at 1,200 steps but worse than 0.005 at 4,800 steps. This is the expected behaviour when the schedule is stretched and the model approaches convergence. LR 0.005 is the best confirmed value at 4,800 steps.

Impact: switching from 0.001 to 0.005 at 4,800 steps improves single-model validation BPB by −0.0463.

8\. Ensembling

We trained three members at LR 0.005, 4,800 steps, seeds 17/42/123.

8.1 Single members

| Member | Seed | Val BPB | Train Time |
|--------|------|---------|------------|
| 1      | 17   | 1.6467  | \~3,000s   |
| 2      | 42   | 1.6659  | 2,653s     |
| 3      | 123  | 1.6713  | 4,577s     |

Training times vary because of background load on the test machine, but the checkpoints themselves are unaffected.

8.2 Ensemble size

| Members | LR    | Val BPB | Test eval time |
|---------|-------|---------|----------------|
| 1       | 0.001 | 1.6930  | 12.9s          |
| 2       | 0.001 | 1.6418  | 23.2s          |
| 3       | 0.001 | 1.6238  | 36.5s          |
| 3       | 0.005 | 1.5714  | \~46s          |

Findings: Ensembling gives large, consistent gains: 1→2 members is −0.0512 val BPB, 2→3 is −0.0180 val BPB. The LR improvement composes with ensembling: the 0.005 ensemble improves on the 0.001 ensemble by another −0.0524 val BPB. Gains diminish with ensemble size, and evaluation time grows linearly. At 4 members the estimated scoring time (\~60 s) would exceed the 5× budget, so 3 is the practical maximum.

8.3 Ensemble diversity

Members differ only by random seed; they share architecture, optimizer, schedule and step count. Even so, the ensemble gain is substantial, confirming that seed-level diversity is a reliable source of improvement at this scale.

9\. Final configuration

| Item                         | Value                                                            |
|------------------------------|------------------------------------------------------------------|
| Architecture                 | LayerNorm + RoPE GPT, width 128, depth 4, heads 4, context 256   |
| MLP                          | GELU 4×                                                          |
| Position                     | RoPE, base 10000                                                 |
| Parameters per member        | 1,055,488                                                        |
| Training                     | AdamW, peak LR 5e-3, 100-step warmup, cosine to 10%, 4,800 steps |
| Processed targets per member | 39,321,600                                                       |
| Members                      | 3(seeds 17, 42, 123)                                             |
| Ensemble                     | Arithmetic mean of log-probabilities                             |
| Test BPB                     | 1.5929                                                           |
| Validation BPB               | 1.5714                                                           |
| Baseline Test BPB            | 2.1013                                                           |
| Relative improvement         | −24.2%                                                           |

10\. Resources and cost

10.1 Evaluation limits (final frozen predictor)

| Limit               | Measured      | Budget               | Status       |
|---------------------|---------------|----------------------|--------------|
| CPU scoring time    | \~46 s (idle) | ≤51 s (5× baseline)  | Pass (tight) |
| Peak evaluation RAM | 1.587 GiB     | ≤4 GiB               | Pass         |
| Inference assets    | 12.12 MiB     | ≤64 MiB uncompressed | Pass         |

Evaluation time was measured at 46.2 s in a dedicated measure_ram.py run, and 50.8–60.1 s under varying background load. The budget is 5× the baseline scoring time, approximately 51 s. Evaluation should be run on an idle machine.

10.2 Training cost

| Phase                            | Runs | Per run     | Total   |
|----------------------------------|------|-------------|---------|
| LR search (1,200 steps)          | 8    | \~530–570s  | \~75min |
| Step-count study (1,200–4,800)   | 4    | 560–3,770s  | \~2.5h  |
| Final members (4,800 steps)      | 3    | 2650–4,580s | \~3h    |
| Ablations (RMSNorm, SwiGLU, EMA) | 5    | 530–2,320s  | \~2h    |

Total training and search: approximately 10 hours of CPU time on 4 threads. Training duration is unrestricted by the guide; only evaluation is budgeted.

10.3 Trade-off

The 3-member ensemble costs roughly 3× the single-model scoring time for a −0.127 test BPB improvement over the best single model. This is the most favourable quality/cost trade in our experiments, but it consumes almost the entire evaluation-time budget. Any further increase in evaluation cost would violate the constraint.

11\. Critical analysis

11.1 Why RoPE helps

RoPE encodes relative position directly in the attention operation. For a small model on natural text, this is a better inductive bias than learned absolute positions: it removes 32,768 parameters, provides consistent position information across the context window, and does not require the model to memorise a position table. The improvement is large and consistent across budgets (−0.149 val BPB at 1,200 steps).

11.2 Why RMSNorm, SwiGLU, EMA and cosine-to-zero did not help

RMSNorm is functionally close to LayerNorm; at this scale the mean-centering removed by RMSNorm is not a bottleneck. Its main benefit here would be speed. SwiGLU typically helps at larger width or longer training, where the gating can specialise. At width 128 with 4,800 steps, it is within noise of GELU. EMA and cosine-to-zero both reduce the effective number of late-training updates. At a budget where the model is still improving, this hurts. They are more appropriate when training is long enough to overfit or to oscillate.

These negative results are useful: they show the baseline recipe is already well-tuned for this scale, and that gains come from the positional encoding and from ensembling rather than from adding complexity.

11.3 What the comparisons establish

The same-budget table isolates RoPE as the effective mechanism (−0.149). The step-count table quantifies the value of additional training and shows clear diminishing returns. The LR table shows the optimal peak LR depends on training length, a standard but often overlooked interaction. The ensemble table shows seed-level ensembling is a reliable, if costly, gain.

11.4 Limitations

The ensemble's evaluation time is close to the budget; the result is sensitive to machine load at scoring time. We did not test larger models or longer training, both of which are allowed by the guide but would require more compute. Only three seeds were used; more members would likely help further but exceed the evaluation budget. Weighted ensembling and member diversity (different LR or step counts) were not explored; both could give small additional gains.

12\. Reproducibility

12.1 Final score

Download the checkpoint, then:

python evaluate.py --checkpoint runs/ensemble-lr005/[checkpoint.pt](https://checkpoint.pt/) --device cpu --precision fp32 --split test

Expected: bpb = 1.5929.

CheckpointSHA256: 5a1dadb4716a92f31470994b9c16563a5b3d2d57b21d3d587fc9d740589a06cb

12.2 Training from scratch

python train.py --implementation student --config configs/rope_only.json --device cpu --threads 4 --seed 17 --run-dir runs/member-17 --steps 4800 --lr 0.005

python train.py --implementation student --config configs/rope_only.json --device cpu --threads 4 --seed 42 --run-dir runs/member-42 --steps 4800 --lr 0.005

python train.py --implementation student --config configs/rope_only.json --device cpu --threads 4 --seed 123 --run-dir runs/member-123 --steps 4800 --lr 0.005

Edit pack_ensemble.py MEMBERS to point at the three run directories, then:

python pack_ensemble.py

python evaluate.py --checkpoint runs/ensemble-lr005/[checkpoint.pt](https://checkpoint.pt/) --device cpu --precision fp32 --split test

12.3 Contract tests

python -m unittest discover -s tests -v

Expected: 5 tests, all pass.

13\. AI assistance disclosure

AI assistance was used to draft and review the RoPE, RMSNorm and SwiGLU implementations, the ensemble packing script, and the experimental plan. All generated code was reviewed, tested against tests/test_contract.py, and modified where necessary. The author understands and can explain every component. No external training text, pretrained weights, test-set tuning, cached answers, future-token access, cross-window state, or evaluation-time network access was used. All model and hyperparameter selection used the validation split only.

14\. Data attribution

WikiText-2 was introduced by Merity et al. in Pointer Sentinel Mixture Models (arXiv:1609.07843). Text by Wikipedia contributors, licensed under CC BY-SA 3.0 and the GNU Free Documentation License. The supplied splits preserve revision b08601e04326c79dfdd32d625aee71d232d685c3 and are encoded as UTF-8 with a training-fitted BPE-2048 tokenizer.
