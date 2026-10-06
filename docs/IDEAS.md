# Burgers RF — Ideas

Research hypotheses and candidate directions that are **not yet** committed
experiments. See `RESEARCH_LOG.md` for what has actually been run and
`TODO.md` for what's next in priority order.

## Candidate directions

1. **Residual-driven local basis design.** The local feature should be
   designed to approximate `u_true - u_RF` (the actual PDE-trained
   residual), not necessarily to resemble `u_true` or the physical shock
   shape itself. This reframing follows directly from `TODO.md` Task 1.

2. **Shifted Gaussian features.** Motivated by the Section 13 finding in
   `RESEARCH_LOG.md`: allow `center != 0`.

3. **Two shifted Gaussians, opposite-signed (now the primary next
   direction).** `RESEARCH_LOG.md` Section 15 confirms the PDE-trained RF
   residual has a two-lobe, sign-changing structure around the shock
   (positive on the left, negative on the right at `t=1`), and Section 19
   shows a single shifted Gaussian corrects only one (right-hand) lobe.
   Natural next candidate, consistent with the Sept-30 advisor discussion:

       u_local(x,t) = d1(t) G1(x;c1,sigma1) + d2(t) G2(x;c2,sigma2)

   with potentially opposite-signed amplitudes. The immediate question is
   **not yet** how to train `d1, d2` — first determine post-hoc (no PDE
   retraining) whether this two-Gaussian spatial basis can explain
   substantially more of the fixed PDE-trained RF residual than one
   Gaussian can (see `TODO.md` Task 1). Kept as the primary direction here
   because it follows directly from both the advisor's suggestion and the
   observed residual data, not just from speculation about the direct-fit
   residual's shape.

4. **Derivative-of-Gaussian feature — secondary interpretation.** A pair of
   oppositely-signed shifted Gaussians (item 3) can itself resemble an odd,
   derivative-like localized correction, consistent with the confirmed
   two-lobe residual shape. Keep this as a secondary future interpretation
   of the item-3 result, not a separate experiment to run first — item 3
   remains the primary direction since it follows directly from the
   advisor's Sept-30 suggestion and the observed data.

5. **Moving local feature**, `center(t) = s(t)`, potentially also
   `sigma(t)` — raised at the 08/26 meeting, not yet tested in a controlled
   experiment.

6. **Trainable center and width** (as opposed to fixed, swept values).

7. **Two-stage / residual training** (train RF, then freeze and fit the
   local term against the resulting residual) — see `TODO.md` Task 6.

8. **Error-dependent local activation.** An "error-dependent" or
   "reciprocal" idea appears in handwritten notes from the 09/30 meeting.
   **Its exact mathematical formulation is uncertain** and must remain
   marked as such until clarified with the advisor — do not implement a
   guessed version of this idea.

9. **Adaptive shock weighting / softmax-like strategy.** Lower priority:
   fixed regional shock-loss weighting already failed (`RESEARCH_LOG.md`
   Section 11), so a softmax/adaptive variant is not an obvious next step
   without a reason to expect it would behave differently.

10. **Increasing RF capacity** (more spatial/temporal random features), to
    address the Gibbs-like oscillation near the shock noted after the
    Section 9 plotting fix — see `TODO.md` Task 7.

11. **Alternative local bases**: tanh transition, derivative-of-Gaussian,
    Gaussian pairs, compactly supported bases, possibly a small learned
    local model.

12. **Important conceptual distinction** (not yet fully resolved): the
    local basis should represent the *RF residual*, not necessarily match
    the physical shock shape — these are different design targets and
    earlier experiments (physical-width matching, Section 1) implicitly
    conflated them.

13. **Conditioning-aware local features** — design choices that avoid the
    `1/sigma^2` derivative blowup identified in `RESEARCH_LOG.md` Section 7
    for very narrow widths.

14. **RF-space-orthogonalized local features**: `G_perp = G - P_RF(G)`,
    i.e. explicitly remove whatever part of the Gaussian is already
    representable by the RF span before using it as a correction.

15. **Multi-seed robustness** — see `TODO.md`, after a promising model is
    found.

## Ideas already tested — evidence against (do not re-attempt without new reasoning)

- **More epochs alone** fixes narrow-sigma non-activation — tested up to
  10,000 epochs at sigma=.02 and at resolved sigma=.0025; no rescue.
- **Simply adding more shock-region collocation points** (redistribution or
  additive) activates the local feature — tested at 720/3600 and 719/4243
  shock-point configurations; no rescue, global/smooth often worsened.
- **Fixed shock-region loss weighting** activates the local feature — tested
  up to alpha=20 (30% effective objective share); no activation, accuracy
  worsened at high alpha.
- **IC penalty is the sole explanation** for near-zero local coefficients —
  tested both by reducing/removing `IC_BC_WEIGHT` and by making the local
  feature structurally IC-compatible; neither activated the feature.
- **Temporal RF complexity is the sole explanation** — tested by collapsing
  the 10-dimensional temporal random-feature expansion to a single linear
  scalar (`d*t`); no activation.
- **Narrow Gaussians are redundant with the RF span** (so there's nothing
  for them to contribute) — only true at sigma=.20 (`residual_g ~ 0`); at
  sigma=.0025 the Gaussian is ~88.5% outside the RF span, yet still gave
  almost no benefit in the *centered* direct-fit test.
- **Blanket claim that Gaussian functions cannot represent the missing
  correction** — refuted by the shifted/scaled direct-fit diagnostic
  (`RESEARCH_LOG.md` Section 13): a shifted, optimally-scaled Gaussian
  recovers a substantial fraction of the *direct-fit* residual. The
  correct, narrower claim is that a *centered, unit-amplitude* Gaussian is a
  poor match — not that the Gaussian family is unsuitable in general.
