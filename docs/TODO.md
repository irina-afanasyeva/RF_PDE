# Burgers RF — TODO

Actionable research plan. See `RESEARCH_LOG.md` for the numerical results and
reasoning behind each item; this file tracks priority and status only.

## Current priority — Phase A: improve/understand u_RF

**Following the 2026-10-14 advisor meeting** (`docs/meetings/2026-10-14.md`;
`RESEARCH_LOG.md` Section 35), the research priority has shifted: the
immediate problem is the quality of the baseline RF approximation `u_RF`
itself, not further local-correction complexity. Return to local shock
enrichment (the former Phase, items now moved to "Later / deprioritized"
below) only once a satisfactory smooth-region `u_RF` is established.

**A. Compare the current RF implementation/setup against the advisor's
better RF solution**, once shared, to identify what differs.

**B. Multi-seed RF study.** The basis-size sweep (completed, see below) used
only seed=42. Run multiple seeds (advisor mentioned e.g. 20, 21, ... as
examples) to separate "poor particular random realization" from a
systematic capacity/training issue.

**C. Systematic Nx/Nt convergence study**, investigating Nx and Nt
jointly rather than Nx alone (the completed sweep fixed Nt=10), possibly
with a controlled Nx/Nt ratio, over a substantially broader RF-size range.
**The exact Nx/Nt grid is explicitly TBD** — not decided at the meeting; do
not invent specific values.

**D. Smaller-learning-rate diagnostic**, as a controlled test of `u_RF`
quality — kept isolated from other architecture changes in the same run,
not combined with them.

**E. Systematic multiscale RF study.** Extends the two completed
diagnostics (below): select sigma1/sigma2 in a principled way — ultimately
informed by PDE structure rather than chosen arbitrarily — and
investigate the N1/N2 allocation between populations and the effect of
increasing total N. **Do not treat the completed std=75 result as having
found the correct or optimal scales** — exact future values/grid TBD.

**F. Joint space-time RF formulation, as an idea to test (not an
established improvement):**
```
phi_k(x,t) = cos(wx_k*x + wt_k*t + b_k)   (joint, one feature per k)
```
versus the current separable product form
```
phi_ij(x,t) = cos(wx_i*x+bx_i) * cos(wt_j*t+bt_j)
```
**A fair comparison budget between the separable `Nx*Nt`-coefficient
representation and a joint `N`-feature representation is explicitly TBD**
— do not invent the comparison protocol yet.

## Next

**G. Investigate the structure/conditioning of the PDE residual near the
shock**, as a possible contributor to both the Phase-A `u_RF` quality
question and the still-open local-amplitude questions below.

## Later / deprioritized (Phase B — resume after a satisfactory u_RF baseline)

**H. Trainable Gaussian geometry: train center `c`, width `sigma`, and
amplitude `d` jointly through the PDE objective** (generalizes to
`{c_k, sigma_k, d_k}` for multiple Gaussians).

**Explicitly deprioritized/postponed per the 2026-10-14 advisor
direction** (`RESEARCH_LOG.md` Section 35, item 6) — not abandoned, but
not the next experiment. Revisit only after Phase A establishes a good
`u_RF` baseline.

**I. The still-open `d1`/`d2` amplitude-asymmetry question**
(`RESEARCH_LOG.md` Section 29) and further K-Gaussian PDE-training work
(Section 33 found K=8 activates only weakly, similar to K=2) — both
carried forward as Phase B work, not dropped:
- analyze the PDE-loss contributions/gradients separately per Gaussian;
- compare the PDE-training objective's optimum in the full `(d1,...,dK)`
  space against the solution-error optimum;
- only later return to joint RF+local training with a richer basis.

**J. Other local bases** (derivative-of-Gaussian, tanh transition,
compactly supported bases, possibly a small learned local model).

**K. Separate implementation task: fix `nn.Parameter(...).to(device)`
registration in `model.py`.**

Root cause documented in `RESEARCH_LOG.md` Section 16 — on CUDA,
`.to(device)` silently downgrades `Wx/Wt/bx/bt` from `Parameter` to plain
`Tensor`, so they are absent from `state_dict()` on GPU-trained checkpoints.
Does not affect training correctness (these tensors have
`requires_grad=False` and were used correctly throughout), only checkpoint
completeness — not urgent, but should not be forgotten. (A related,
narrower device-handling issue specific to post-hoc diagnostic scripts —
not `model.py` itself — was found and fixed separately; see
`test_rf_nx2000_long_training.py`'s and
`test_rf_multiscale_two_population.py`'s commit history.)

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
- RF basis-size sweep, Nx in {500,1000,2000}, Nt=10, seed=42 (Section 31)
  — solution error (global/smooth/shock) *increased* with Nx under the
  fixed canonical training budget, despite the sampled PDE loss
  decreasing; single-seed/fixed-Nt limitations motivate Phase A items B/C
  above.
- Error-vs-Nx plot (same sweep, Section 31) — generated alongside the
  above.
- Nx=1000 longer-training diagnostic, 1000->10000 epochs (Section 32) —
  only modest improvement; still does not match canonical Nx=500 at 1000
  epochs; weakens but does not rule out "just needs more epochs."
- Frozen-RF K=8 Gaussian PDE-training experiment (Section 33) — K=8 has
  strong post-hoc representational capacity (consistent with Section 26)
  but PDE training activates it only weakly; RF/local co-adaptation
  excluded as the explanation (RF completely frozen); notably performed
  *worse* at t=1 shock error than the earlier K=2 frozen-RF result despite
  much stronger post-hoc capacity.
- Two-population multiscale RF, aggressive split: 250@std25 + 250@std312.5
  (Section 34.1) — substantially worsened solution accuracy and PDE loss
  relative to canonical Nx=500.
- Two-population multiscale RF, moderate split: 250@std25 + 250@std75
  (Section 34.2) — first controlled positive multiscale result: same
  total Nx=500, better frequency allocation improved global, smooth,
  shock, and PDE metrics simultaneously. Not claimed optimal.

## Workflow

```
hypothesis -> smallest controlled change -> inspect -> correctness check ->
experiment -> save results -> interpret -> update RESEARCH_LOG -> review diff
-> commit
```
