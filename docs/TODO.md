# Burgers RF — TODO

Actionable research plan. See `RESEARCH_LOG.md` for the numerical results and
reasoning behind each item; this file tracks priority and status only.

## Current priority

**1. Investigate why the fixed-RF PDE training objective permits only a
small negative local amplitude, despite larger solution-error improvement
being possible.**

`RESEARCH_LOG.md` Section 23 (fixed-RF sweep): the PDE-training objective's
own optimum is `d*≈-0.01`, while solution-error optima prefer `d*≈-0.66` to
`-1.11` depending on the metric. Section 24 (two-stage training) confirmed
Adam actually reaches `d≈-0.0105` when the RF is frozen — matching the
fixed-RF prediction almost exactly — but this barely improves the solution
(canonical shock error `0.8390488 → 0.8348522`). This is now the central
open question (Section 25).

**2. A controlled `WEIGHT_DECAY` diagnostic is a plausible simple next
test.**

Inspect/discuss the exact mechanism before implementing anything. Vanilla
`Adam`'s weight decay (not decoupled, unlike `AdamW`) adds
`weight_decay*param` directly to the gradient, in proportion to the
parameter's own current magnitude — plausible as one contributor to the
small amplitude above. **Do not assume this is the cause** — it is one
candidate among several (`RESEARCH_LOG.md` Section 25, `IDEAS.md` item 17).

## Next

**3. Investigate the structure/conditioning of the PDE residual near the
shock**, as a possible explanation for why the training objective tolerates
only a small local-correction magnitude.

**4. Consider two-stage training with a richer local representation** (e.g.
two Gaussians, trainable center/width) if the diagnostics above justify it.

## Later

**5. Two shifted Gaussians (G1+G2) post-hoc diagnostic, and (if justified)
use that evidence to design the next PDE-training experiment.**

Motivated by the two-lobe residual (`RESEARCH_LOG.md` Section 15/19): a
single shifted Gaussian only corrects one lobe. No longer the immediate
next experiment — Sections 21–25 found a more urgent prior question (Tasks
1–2 above): even a single, correctly-signed local Gaussian only weakly
activates under the current training objective, so adding a second Gaussian
does not obviously address that magnitude problem on its own. Not yet
tested, post-hoc or otherwise.

**Do not treat the Section 17 single-Gaussian optimum (`c=0.014,
sigma=0.007646`) as parameters to train directly** — those are the optimum
for a single post-hoc Gaussian fit against the fixed PDE-trained residual,
not an established correct local basis.

**6. Trainable Gaussian parameters (center, width).**

Later, after the fixed-parameter diagnostics above.

**7. Other local bases** (derivative-of-Gaussian, tanh transition,
compactly supported bases, possibly a small learned local model).

**8. Separate implementation task: fix `nn.Parameter(...).to(device)`
registration in `model.py`.**

Root cause documented in `RESEARCH_LOG.md` Section 16 — on CUDA,
`.to(device)` silently downgrades `Wx/Wt/bx/bt` from `Parameter` to plain
`Tensor`, so they are absent from `state_dict()` on GPU-trained checkpoints.
Does not affect training correctness (these tensors have
`requires_grad=False` and were used correctly throughout), only checkpoint
completeness — not urgent, but should not be forgotten.

**9. Increase RF feature count as a separate RF-only diagnostic.**

Investigate whether the Gibbs-like oscillation/overshoot near the shock
(visible after the Section 9 plotting fix) is a feature-count/conditioning
effect, independent of the local-feature question.

## After a promising model

Multi-seed validation (not before — single-seed results throughout
`RESEARCH_LOG.md` have not yet been checked for seed sensitivity).

## Completed (do not repeat without a new reason)

- Width sweep, centered Gaussian, sigma in {.02–.50} — best .20, worst .02.
- Multi-Gaussian widths {.05,.10,.20} — no better than best single width.
- Longer training (10,000 epochs) at sigma=.02 — no rescue.
- Collocation redistribution to 720 shock points — no rescue, global/smooth
  worsened.
- Additive shock collocation refinement (+643 points) — no rescue.
- Very-small-sigma sweep {.0001–.3125} — negligible activation at narrow
  widths.
- Resolution-controlled small-sigma experiment (150 points in 3·sigma) —
  no activation, including at 10,000 epochs.
- Implementation/gradient/conditioning audit — no implementation bug found.
- IC-weight experiment (IC_WEIGHT in {1000,100,10,0}) — no activation;
  removing IC weight actively harmful.
- Direct spatial-fit diagnostic (centered Gaussian, t=1, dense-LS RF
  baseline) — centered narrow Gaussian gives almost no benefit.
- IC-compatible local feature — no activation; IC competition was not the
  primary cause.
- Explicit shock-region loss weighting (alpha up to 20) — no activation;
  worsens accuracy at high alpha.
- Simple temporal local feature (`linear_t`, single scalar `d`) — no
  activation; temporal RF complexity was not the primary cause.
- Fixed/shifted Gaussian direct diagnostic — centered Gaussian conclusion
  refined: a shifted, optimally-scaled Gaussian recovers ~20–27% of the
  *direct-fit* residual, but this has not yet been shown for the
  PDE-trained residual.
- Analyzed the actual PDE-trained RF baseline and residual
  (`RESEARCH_LOG.md` Section 15) — residual has a two-lobe, sign-changing
  structure around the shock (positive left, negative right at `t=1`); a
  centered, even Gaussian cannot represent this by itself.
- Diagnosed and reconstructed the missing `Wx/Wt/bx/bt` checkpoint issue
  (Section 16) — training itself was unaffected; a complete checkpoint now
  exists (`checkpoint_complete.pt`); the underlying `model.py` fix is
  tracked separately above (Task 5).
- Single-Gaussian parameter search/correction on the fixed PDE-trained RF at
  `t=1` (Sections 17–19) — large L2 improvement (shock ≈47.8%) and
  stronger residual alignment than the earlier direct-fit result, but
  visual inspection shows the correction is structurally incomplete: it
  corrects only the right-hand residual lobe, leaving the left-hand lobe
  uncorrected. One Gaussian is not sufficient to reconstruct the full shock
  transition.
- Joint PDE training with the shifted/wider Gaussian, IC-compatible form
  (Section 21) — activated much more strongly than the old centered
  Gaussian (coefficient norm ≈3.54x larger, ratio_shock ≈46x larger) but
  worsened the solution; the learned local contribution had the wrong sign
  relative to the post-hoc-useful direction.
- `linear_t` with the shifted Gaussian (Section 22) — the single-scalar
  local feature still learned a positive `d`, ruling out temporal-expansion
  flexibility as the explanation for the wrong sign.
- Fixed-RF one-dimensional `d` sweep (Section 23) — the PDE training
  objective and solution error agree on direction (negative `d`) but
  disagree sharply on magnitude; the `t=1` shock-error optimum (`d≈-0.66`)
  matches the independent post-hoc oracle almost exactly.
- Two-stage / frozen-RF training (Section 24) — directly confirmed RF/local
  co-adaptation as a cause of the earlier joint-training sign reversal;
  with RF frozen, `d` moved negative as the fixed-RF objective predicted
  and modestly improved the solution, though the improvement remains small
  (Section 25).

## Workflow

```
hypothesis -> smallest controlled change -> inspect -> correctness check ->
experiment -> save results -> interpret -> update RESEARCH_LOG -> review diff
-> commit
```
