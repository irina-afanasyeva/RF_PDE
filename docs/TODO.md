# Burgers RF — TODO

Actionable research plan. See `RESEARCH_LOG.md` for the numerical results and
reasoning behind each item; this file tracks priority and status only.

## Current priority

**Following the 2026-10-07 advisor meeting** (`docs/meetings/2026-10-07.md`;
`RESEARCH_LOG.md` Section 30), the immediate execution order is A–E below.
The still-open `d1`/`d2` asymmetry question from before the meeting
(previously item 1) is **not dropped** — it is carried forward as the
scientific motivation for item C and the diagnostics listed under it.

**A. RF basis-size baseline study: N = 500, 1000, 2000.**

Advisor-specified values. Train the RF-only baseline at each `N` (all
other settings controlled/unchanged), reporting both global relative L2
and shock-region relative L2 for each. This fulfills and concretizes the
earlier "increase RF feature count" diagnostic (previously item 6).

**B. Plot global and shock relative L2 error versus N.**

One error-vs-`N` plot (both metrics on it), using the results from A, to
determine whether increasing RF capacity continues to reduce error or
whether it plateaus.

**C. Design the `K>=8` PDE-training experiment in light of the
capacity-study result (A/B).**

Extend Gaussian enrichment to `K>=8` for both the post-hoc/oracle
representation diagnostics and the PDE-training diagnostics (advisor
direction), informed by whichever RF feature count A/B motivates. **The
exact N-by-K combinations to run are TBD** — not yet decided, and this
should not be expanded into a full N×K matrix without further
confirmation. This also continues the still-open question of why the
frozen-RF PDE objective favors `d2` far more than `d1`
(`RESEARCH_LOG.md` Section 29) — a richer `K>=8` basis gives another angle
on that question, not a replacement for it.

**Possible future diagnostics for the `d1`/`d2` asymmetry question**
(`RESEARCH_LOG.md` Section 29 — not yet run, remain open questions, not
conclusions):
- analyze the PDE-loss contributions/gradients separately for `G1` versus
  `G2`, to see whether the objective's local gradient structure itself is
  asymmetric between the two Gaussians;
- compare the PDE-training objective's optimum in the full `(d1,d2)` plane
  against the solution-error optimum (extending Section 23's 1-D sweep to
  2-D), to see whether the objective's preferred point is itself
  asymmetric or whether optimization dynamics are responsible;
- only later return to joint RF+local training with a richer basis.

## Next

**D. Trainable Gaussian geometry: train center `c`, width `sigma`, and
amplitude `d` jointly through the PDE objective** (generalizes to
`{c_k, sigma_k, d_k}` for multiple Gaussians).

Advisor direction (2026-10-07 meeting); supersedes/extends the earlier
"trainable center and width" item (previously item 3) by also training the
amplitude. Recorded as a future experiment, not yet attempted — only after
the amplitude/capacity questions above (A–C) are further understood.

**E. Multiscale/two-population RF experiment.**

New hypothesis from the 2026-10-07 meeting (`RESEARCH_LOG.md` Section 30):
two RF populations at different frequency scales,
```
u_RF = sum_k a_k cos(w_k^(1) x + b_k^(1)) + sum_k beta_k cos(w_k^(2) x + b_k^(2))
```
with `w_k^(1) ~ N(0,sigma_1^2)`, `w_k^(2) ~ N(0,sigma_2^2)`. Motivation:
one population for smoother structure, the other for additional shock
resolution. **Not yet tested; candidate `N1,N2` values from the
handwritten notes are TBD / need advisor confirmation before
implementation** — do not invent specific values.

**F. Investigate the structure/conditioning of the PDE residual near the
shock**, as a possible explanation for the amplitude/asymmetry questions
above.

## Later

**G. Other local bases** (derivative-of-Gaussian, tanh transition,
compactly supported bases, possibly a small learned local model).

**H. Separate implementation task: fix `nn.Parameter(...).to(device)`
registration in `model.py`.**

Root cause documented in `RESEARCH_LOG.md` Section 16 — on CUDA,
`.to(device)` silently downgrades `Wx/Wt/bx/bt` from `Parameter` to plain
`Tensor`, so they are absent from `state_dict()` on GPU-trained checkpoints.
Does not affect training correctness (these tensors have
`requires_grad=False` and were used correctly throughout), only checkpoint
completeness — not urgent, but should not be forgotten.

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
- PDE-trained-RF post-hoc two-Gaussian representation diagnostic (Section
  26) — a jointly searched, well-conditioned K=2 pair explains ~99.8% of
  the `t=1` shock residual (shock error `0.7597 → 0.0359`, ≈95.3%
  improvement); representation capacity is not the bottleneck.
- Greedy K-Gaussian representation-capacity study, K=1..8, with
  independent dense-grid validation (Section 26) — confirms very high,
  genuine representation capacity; K=8 is explicitly NOT claimed optimal,
  and the clean K=2 pair remains preferred for trainability experiments.
- Frozen-RF two-Gaussian amplitude-only PDE training (Section 27) — both
  `d1,d2` start in the correct (oracle-matching) direction from zero, but
  converge to only ~5.6% of their useful post-hoc amplitudes, with the RF
  completely frozen throughout (ruling out co-adaptation as the
  explanation for this specific gap).
- Weight-decay sensitivity sweep, `weight_decay in {1.0,0.1,0.01,0.0}`
  (Section 28) — weight decay materially suppresses the amplitudes
  (especially `d2`) and the effect saturates by `weight_decay≈0.01`, but
  even at `weight_decay=0` both amplitudes remain far from oracle, with a
  pronounced, unexplained asymmetry between `d1` (~3.5% of oracle) and
  `d2` (~33% of oracle). Weight decay is a real but partial contributor,
  not a full explanation.

## Workflow

```
hypothesis -> smallest controlled change -> inspect -> correctness check ->
experiment -> save results -> interpret -> update RESEARCH_LOG -> review diff
-> commit
```
