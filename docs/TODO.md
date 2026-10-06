# Burgers RF — TODO

Actionable research plan. See `RESEARCH_LOG.md` for the numerical results and
reasoning behind each item; this file tracks priority and status only.

## Current priority

**1. Two-Gaussian post-hoc diagnostic on the fixed PDE-trained RF.**

No PDE retraining yet. `RESEARCH_LOG.md` Section 15 confirmed the
PDE-trained RF residual has a two-lobe, sign-changing structure around the
shock (positive on the left, negative on the right at `t=1`), and Section
19 showed that a single shifted Gaussian only corrects one (right-hand)
lobe, leaving the other largely untouched. Test whether a pair

```
u_local(x,t) = d1(t) G1(x;c1,sigma1) + d2(t) G2(x;c2,sigma2)
```

(centers/widths searched, amplitudes analytically optimal, same closed-form
approach as Sections 13/17) can represent substantially more of the fixed
residual than one Gaussian — same post-hoc, non-training diagnostic style
as Section 17.

**Do not treat the Section 17 single-Gaussian optimum (`c=0.014,
sigma=0.007646`) as parameters to train directly** — those are the optimum
for a single post-hoc Gaussian fit against the fixed PDE-trained residual,
not yet established as the correct local basis to train with.

## Next

**2. Use the two-Gaussian spatial-basis evidence to design the next
PDE-training experiment.**

Only after Task 1 establishes whether/how much of the residual a
two-Gaussian basis explains, and with centers/widths justified by that
evidence (not assumed from the single-Gaussian search alone). Change
exactly one factor at a time relative to the current baseline; keep RF
realization, collocation set, optimizer, initialization, and training
duration controlled, matching the established controlled-experiment
convention used throughout `RESEARCH_LOG.md`.

**3. Trainable Gaussian parameters (center, width).**

Later, after the fixed-parameter diagnostics above.

**4. Two-stage RF + local training.**

Stage 1: train the RF term alone. Stage 2: freeze or partially freeze the
RF term and train the local correction against the resulting residual.

**5. Separate implementation task: fix `nn.Parameter(...).to(device)`
registration in `model.py`.**

Root cause documented in `RESEARCH_LOG.md` Section 16 — on CUDA,
`.to(device)` silently downgrades `Wx/Wt/bx/bt` from `Parameter` to plain
`Tensor`, so they are absent from `state_dict()` on GPU-trained checkpoints.
Does not affect training correctness (these tensors have
`requires_grad=False` and were used correctly throughout), only checkpoint
completeness — not urgent, but should not be forgotten.

**6. Increase RF feature count as a separate RF-only diagnostic.**

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

## Workflow

```
hypothesis -> smallest controlled change -> inspect -> correctness check ->
experiment -> save results -> interpret -> update RESEARCH_LOG -> review diff
-> commit
```
