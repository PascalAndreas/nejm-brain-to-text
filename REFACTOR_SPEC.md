# Neural Encoder + CTC Baseline — Implementation Blueprint

This doc is the authoritative spec for refactoring the model side of the pipeline. It captures **all decisions** we just locked in so Claude (and future-you) can implement without re-reading threads.

---

## 0) Scope & Goals

* Replace the legacy training stack with a clean, modular **Neural Encoder → CTC head** that plugs into the existing **/decoding** module (Flashlight-CTC + KenLM).
* Keep dataset and decoding interfaces stable; move all model-specific logic behind a **single contract**.
* Fix the padding/length class of bugs permanently.
* Support day-specific adaptation **safely** (FiLM with identity tether).
* Enable future swaps (GRU ↔ Conformer), ensembling, and LLM N-best rescoring.

Non-goals for Phase 1: WFST changes, phone-LM, diffusion-like refinement, replacing KenLM, backward-compat with legacy trainer.

---

## 1) Data Interface (what the dataloader returns)

The current collate (zero-pads using `pad_sequence`) yields:

* `input_features`: **float32** `[B, T, F]` — binned neural features
* `n_time_steps`: **int** `[B]` — true lengths (frames) before padding
* `seq_class_ids` (optional): **int64** `[B, Lp]` — target phone IDs (padded)
* `phone_seq_lens` (optional): **int** `[B]` — true label lengths
* `day_indices`: **int** `[B]`
* `ids/meta`: session/block/trial/corpus; `utt_id` string per example

We will keep the dataset as-is and adapt it to a common **Batch**.

### Batch adapter (shape contract)

Wrap loader outputs into a light struct/dict with keys:

* `x: float32 [B, T, F]`
* `x_lens: int [B]` *(pre-pad true lengths)*
* `y: int64 [B, Lp] | None`
* `y_lens: int [B] | None`
* `day_id: int [B]`
* `utt_id: list[str]`
* `meta: dict`

> Padding value for inputs is **0.0**; labels padded with **0** are fine since we always pass `y_lens` into CTC (padded tail ignored).

---

## 2) Model Abstractions (contracts)

### Emissions (model output)

* `log_probs: float32 [B, T_out, V]` — **log-softmax** over phones (includes CTC blank index)
* `out_lens: int [B]` — true emission lengths after all time-reducing ops
* `aux: dict` — optional features for logging/losses

### NeuralEncoder (base API)

* `forward(batch: Batch) -> Emissions`
* `time_reduction() -> int` (overall stride; 1 if none)
* `supports_packed() -> bool` (for RNN impls)
* `eval_calibrate(dev_loader) -> None` (optional global temperature fitting; stores `T`)
* `export_config() -> dict` (architecture & hyperparams for reproducibility)

### Registry / Factory

```
ENCODERS = {
  "gru_v1": GRUCTC(...),
  "conformer_v1": ConformerCTC(...),
  ...
}

build_encoder(name: str, cfg: dict) -> NeuralEncoder
```

> Pipeline never touches internals; it only calls **build → forward/emit → decode**.

---

## 3) Model Composition (blocks)

```
NeuralEncoder =
  PreNet(
    GaussianSmoother(),           # MVP: fixed Gaussian (kernel=100, std=2) to match legacy
    DayAdapter(),                 # FiLM(γ,β) identity‑init; optional group‑wise (8 blocks)
    Nonlinearity(softsign)        # to match legacy behavior
    # (Later: TypeAwareNormalize, LearnedDepthwiseConv, etc.)
  )
  └─ EncoderBackbone()            # GRU (uni; causal) for MVP; Conformer later
  └─ ProjectionHead()             # Linear(d_hidden → V)
  └─ Calibrator()                 # Temperature T on logits at EVAL; T=1 at TRAIN
  └─ LogSoftmax()                 # produce log_probs for CTC & decoding
```

### PreNet details (MVP)

* **GaussianSmoother**: deterministic; applies along time per feature. Keep params identical to current baseline (kernel size = 100 frames; std = 2). Expose in config.
* **DayAdapter (FiLM)**:

  * `y = γ_d ⊙ x + β_d`
  * Identity init: `γ_d = 1`, `β_d = 0`
  * Regularization: L2 tether on `(γ_d−1, β_d−0)`; optional L1 for sparsity
  * Option: *group-wise FiLM* (one `(γ,β)` per the 8 predefined blocks of 64 chans)
* **Softsign** activation after FiLM.

> Later variants: Type-aware normalization (sqrt for thresholds, log1p for power, then robust per-day z-score), learned depthwise 1D conv as smoother.

### EncoderBackbone (GRU‑CTC MVP)

* Uni-directional GRU stack (e.g., 5 layers × 768 hidden) with dropout & optional Zoneout/variational dropout.
* For RNN: **pack\_padded\_sequence(..., enforce\_sorted=False)** to completely skip padded time.

### ProjectionHead

* Linear to `V` phones (CTC blank included). (Keep it consistent; idx[BLANK]=0, idx[SIL]=40.)

### Calibrator

**temperature scaling**: apply `logits / T` before `log_softmax` at EVAL. Fit `T` once on a mixed-day dev set (minimize NLL).
* Keep `T=1` during TRAIN.

### LogSoftmax

* Always output **log\_probs** from the encoder.
* This makes training, ensembling, and decoding consistent.
* We will have to adapt the decoder so that it expects emissions. Currently, the decoder is doing log_softmax.
* Although this redundancy does not effect results as log_softmax(log_softmax(x))=log_softmax(x)

---

## 4) Time Length Handling (invariants)

We will not create a separate class; instead, enforce **two helpers + invariants**:

1. **`compute_output_lengths(lengths_in, ops_cfg) -> lengths_out`**

   * Pure function. Each time-reducing op (e.g., patch concat, strides) contributes its mapping; compose them.
   * Use it in PreNet (if patching) and Backbone; store final `out_lens` in `Emissions`.

2. **`mask_logits_(logits, out_lens)`**

   * After forward, set all positions beyond each sample’s `out_lens[i]` to **−1e9**, then compute `log_softmax`.
   * This is a safety net even if packing/masking was correct.

**Backbone-specific**

* RNN path: use `pack_padded_sequence` and `pad_packed_sequence` (batch\_first) so paddings never reach the GRU.
* Transformer/Conv path (future): carry a boolean time mask and zero-out attention keys/values or conv outputs beyond `lengths`.

**Do not** approximate lengths (no “trim to 80%”). Always propagate true lengths from dataset → after ops.

---

## 5) Training (Lightning or custom)

### LightningModule responsibilities

* `training_step(batch)`:

  * `em = encoder.forward(batch)` → `log_probs, out_lens`
  * `loss_main = CTC(log_probs, y, out_lens, y_lens)`
  * (optional) `loss_aux = λ * CTC(log_probs_mid, out_lens_mid, ...)` — see §6
  * log losses/metrics
* `validation_step(batch)`:

  * Greedy CTC collapse → PER; optionally decode a fixed small slice for CER/WER sanity
* `configure_optimizers()`:

  * AdamW + cosine; warmup; grad-clip; AMP
* Callbacks for checkpoints, LR monitor; W\&B logger recommended.

Lightning is a wrapper; it does **not** constrain the block modularity.

### If staying custom

Mirror the same responsibilities. Keep AMP, grad-clip, seeding, and checkpointing consistent.

---

## 6) Mid-layer CTC (deep supervision)

* Attach a small projection head to a **mid layer** (e.g., GRU layer 3 of 5; Conformer layer 6 of 12).
* Compute a second CTC loss against the same targets using the correct `out_lens_mid` from `compute_output_lengths`.
* Total loss: `L = L_top + λ_aux * L_mid` with `λ_aux ∈ [0.2, 0.3]`.
* Benefits: better gradient flow, encourages monotonic/discriminative intermediates, reduces brittleness of final layer.

---

## 7) Ensembling Strategy

* **Primary**: ensemble at **emissions** level — average **log-probs** (or probs) across models per frame → run a **single** CTC+KenLM decode.
* **Secondary (later)**: union of N-best lists → LLM or tiny in-domain LM rescoring.

---

## 8) Decoder Interface (unchanged for now)

* Decoder consumes `log_probs [B,T_out,V]` + `out_lens [B]`.
* Never argmax‑gate blanks. Use full posteriors.
* Keep **N-best** (50–100) available for future LLM rescoring.
* Tuning: low-ish `lm_weight`, grid `word_score` for spacing; continue using your grid/Bayes search.

---

## 9) Day Adapters (safe default)

* **FiLM(γ,β)** per feature or **per block** (8 blocks of 64 channels).
* Identity‑init; add **L2 tether** on deviations; (optional) L1.
* Log ‖(γ−1,β−0)‖ per day in W\&B; alert on outliers.

---

## 10) Smoothing & Normalization (MVP and beyond)

* **MVP**: GaussianSmoother → FiLM → softsign (PreNet).
* **Later**: Type-aware transforms (threshold: sqrt/Anscombe; power: log1p) + robust per‑day z-score, then learned depthwise conv as smoother.

---

## 11) Emissions Caching for Decoder Tuning

* `pipeline/emit.py`: forward the split, dump `{utt_id: (log_probs, out_lens)}` to disk (NPZ/PT).
* `/decoding` can then load caches to sweep `(lm_weight, word_score, beams)` or run Bayesian search **without re-forwarding**.
* This is also how you ensemble multiple encoders cheaply.

---

## 12) Invariants & Unit Tests (must pass)

1. **Pad invariance**: For a batch, compare (A) forward with true lengths & no pad vs. (B) forward with padding + packing/masking. Logits must match up to `T_out` (tol \~1e-5).
2. **Length mapping**: Given known input lengths and configured PreNet/Backbone ops, `compute_output_lengths` equals actual `T_out` returned by the model.
3. **Decoder no-pads**: After `mask_logits_`, decoding a sample yields identical output whether batched with shorter/longer neighbors or alone.

---

## 13) Project Layout

```
models/
  __init__.py          # registry + build_encoder()
  base.py              # Batch/Emissions/NeuralEncoder contracts
  blocks/
    prenet.py          # GaussianSmoother, DayAdapter(FiLM), Softsign, (later: TypeAwareNormalize, DepthwiseConv)
    rnn.py             # GRU stack (causal)
    conformer.py       # (future)
    head.py            # Linear projection; AuxHead for mid-CTC
    calibrator.py      # Temperature scaling (eval-only)
    utils.py           # compute_output_lengths, mask_logits_
  gru_ctc.py           # compose blocks into GRU‑CTC
  conformer_ctc.py     # compose blocks into Conformer‑CTC (later)

pipeline/
  emit.py              # forward → cached emissions
  generate_submissions.py

decoding/              # existing router (flashlight_ctc, greedy, …)
dataset/               # existing dataset + collate (unchanged)
```

---

## 14) Config (single YAML under /pipeline)

```
dataset:
  path: data/...
  batch_size: 32
  num_workers: 8

model:
  name: gru_v1
  vocab_size: 41
  prenet:
    smoother: {type: gaussian, kernel: 100, std: 2}
    day_adapter: {type: film, grouping: blocks8, l2_tether: 1e-4, l1: 0.0}
    nonlinearity: softsign
  backbone:
    type: gru
    hidden_size: 768
    num_layers: 5
    dropout: 0.2
  head: {type: linear}
  calibrator: {temperature: null}  # set after eval_calibrate

training:
  optimizer: {type: adamw, lr: 3e-4, weight_decay: 1e-2}
  schedule: {type: cosine, warmup_steps: 5000}
  amp: true
  grad_clip: 1.0
  batch_frames: 60000
  time_mask: {num: 2, width: 40}
  channel_mask: {p: 0.1}
  channel_dropout: 0.05
  aux_ctc: {enabled: true, layer: 3, weight: 0.25}

decoding:
  impl: flashlight_ctc
  tokens: artifacts/tokens.txt
  lexicon: artifacts/lexicon.txt
  kenlm: artifacts/filtered.bin
  lm_weight: 2.5
  word_score: -0.2
  beam_size: 100
  beam_size_token: 10
  beam_threshold: 20.0
  nbest: 50
```

---

## 15) Migration Checklist (MVP)

* [ ] Implement `models/base.py` contracts and registry.
* [ ] Implement PreNet(MVP): GaussianSmoother → FiLM → Softsign.
* [ ] Implement GRU backbone with packing; ProjectionHead; Calibrator (T fit optional later).
* [ ] Implement `compute_output_lengths` + `mask_logits_` and use them.
* [ ] Write the 3 unit tests; run on CI.
* [ ] Add `pipeline/emit.py` to cache emissions.
* [ ] Keep current `/decoding` and grid/Bayes search; ensure it reads `out_lens` exactly.
* [ ] (Optional) Add mid-layer CTC (aux head) once MVP is stable.

---

## 16) Guardrails & Known Pitfalls

* **Never** approximate lengths or trim by ratio; only use true `out_lens`.
* Always **reset hidden state** between sequences at inference.
* Ensure `log_probs` beyond `out_lens` are **−∞** after masking, not zeros.
* Keep **full posteriors** for the beam; no argmax gating of blanks.
* Day adapters: monitor the norm of deviations; cap if runaway days appear.

---

## 17) Future Extensions (post-MVP)

* Swap Gaussian with learned depthwise conv; introduce TypeAwareNormalize.
* Conformer‑CTC encoder with relative position encodings and stochastic depth.
* Adversarial day‑invariance (DANN) on trunk; keep FiLM for amplitude.
* Emissions‑level ensembling; N‑best LLM rescoring.
* Confusion‑matrix posterior denoising; tiny in‑domain 3‑gram for rerank.

---

## 18) Rationale for Two Key Choices

* **Emissions as log-probs**: unify training, ensembling, decoding; avoids subtle double-softmax bugs and lets you sum log-probs across models cleanly.
* **PreNet owns smoothing & adapters**: dataset stays raw; swapping preprocessing is a local change; reproducibility lives in model config, not data I/O.

---

This is the single source of truth—keep it updated as you iterate.
