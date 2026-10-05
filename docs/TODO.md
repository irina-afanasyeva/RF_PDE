# Burgers RF — TODO

Actionable research plan. See `RESEARCH_LOG.md` for the numerical results and
reasoning behind each item; this file tracks priority and status only.

## Current priority

**1. Analyze the actual PDE-trained RF residual.**

Everything in Sections 9 and 13 of `RESEARCH_LOG.md` (direct spatial-fit
diagnostics, including the shifted/scaled Gaussian result) used the
*direct least-squares-fitted* RF approximation at `t=1`, **not** the actual
PDE-trained RF solution. This is the critical caveat carried over from the
09/30 meeting and must be resolved before designing any new local feature.

Obtain/save the actual RF-only prediction from a real Burgers PDE-training
run, and study

```
e_RF(x,t) = u_true(x,t) - u_RF(x,t)
```

For several times, visualize:
- u_true
- u_RF
- the residual e_RF
- a zoomed view near the shock

Questions to answer:
- Is the residual symmetric or antisymmetric around the shock?
- Does it have positive/negative lobes?
- Does the residual's effective center move with time?
- Does its width change with time?
- Does the shape look like one Gaussian, two Gaussians, a derivative-of-
  Gaussian, or something else?

**This is diagnostic only — do not modify training yet.**

## Next

**2. Fit Gaussian corrections to the PDE-trained RF residual.**

For selected times, search center and sigma and analytically compute the
optimal amplitude `d*` (same closed-form approach as Section 13, applied to
the PDE-trained residual instead of the direct-fit residual). Record
`c*(t)`, `sigma*(t)`, `d*(t)`.

**3. Controlled shifted-center PDE experiment.**

Only after Tasks 1–2 justify it. Change exactly one factor: local-feature
`center=0` vs. the diagnosed `c*`. Keep RF realization, width, collocation
set, optimizer, initialization, and training duration controlled, matching
the established controlled-experiment convention used throughout
`RESEARCH_LOG.md`.

**4. Multiple shifted Gaussians.**

Only if the residual structure from Task 1 justifies it (e.g. two
opposite-signed lobes).

**5. Trainable Gaussian parameters (center, width).**

Later, after the fixed-parameter diagnostics above.

**6. Two-stage RF + local training.**

Stage 1: train the RF term alone. Stage 2: freeze or partially freeze the
RF term and train the local correction against the resulting residual.

**7. Increase RF feature count as a separate RF-only diagnostic.**

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

## Workflow

```
hypothesis -> smallest controlled change -> inspect -> correctness check ->
experiment -> save results -> interpret -> update RESEARCH_LOG -> review diff
-> commit
```
