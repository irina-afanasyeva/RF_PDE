# Burgers RF — Ideas

Research hypotheses and candidate directions that are **not yet** committed
experiments. See `RESEARCH_LOG.md` for what has actually been run and
`TODO.md` for what's next in priority order.

## Candidate directions

1. **Residual-driven local basis design.** The local feature should be
   designed to approximate `u_true - u_RF` (the actual PDE-trained
   residual), not necessarily to resemble `u_true` or the physical shock
   shape itself. This reframing follows directly from the residual analysis
   in `RESEARCH_LOG.md` Section 15.

2. **Shifted Gaussian features.** Motivated by the Section 13 finding in
   `RESEARCH_LOG.md`: allow `center != 0`.

3. **Two shifted Gaussians, opposite-signed — representation confirmed;
   training amplitude now the open question.** `RESEARCH_LOG.md` Section
   15 confirmed the PDE-trained RF residual has a two-lobe, sign-changing
   structure around the shock, and Section 19 showed a single shifted
   Gaussian corrects only one lobe. This item is now **tested**: Section
   26's jointly searched, well-conditioned K=2 pair (consistent with the
   Sept-30 advisor discussion of multiple shifted-center Gaussians)
   explains ~99.8% of the `t=1` shock residual (≈95.3% shock-error
   improvement over RF-only) — representation is confirmed strong, not
   merely hypothesized. Section 27 then trained

       u_local(x,t) = t * [d1*G1(x;c1,sigma1) + d2*G2(x;c2,sigma2)]

   against the frozen RF from `d1=d2=0`: both amplitudes start in the
   correct direction but converge to only ~5.6% of their useful oracle
   values, with the RF completely frozen (ruling out co-adaptation as the
   explanation here). Section 28's weight-decay sweep shows this gap is
   only partially regularization-driven, and reveals a new, unexplained
   asymmetry between `d1` and `d2` (see `RESEARCH_LOG.md` Section 29 /
   `TODO.md` current priority).

4. **Derivative-of-Gaussian feature — secondary interpretation.** A pair of
   oppositely-signed shifted Gaussians (item 3) can itself resemble an odd,
   derivative-like localized correction, consistent with the confirmed
   two-lobe residual shape. Keep this as a secondary future interpretation
   of item 3, not a separate experiment to prioritize on its own.

5. **Moving local feature**, `center(t) = s(t)`, potentially also
   `sigma(t)` — raised at the 08/26 meeting, not yet tested in a controlled
   experiment.

6. **Trainable Gaussian geometry: center, width, AND amplitude — advisor
   direction (2026-10-07 meeting, `docs/meetings/2026-10-07.md`).** Extends
   this earlier "trainable center and width" idea to also train the
   amplitude `d` jointly through the PDE objective: `{c, sigma, d}` for one
   Gaussian, generalizing to `{c_k, sigma_k, d_k}` for K Gaussians. Not yet
   attempted; see `TODO.md` item D.

7. **Two-stage / residual training — experimentally supported, worth
   retaining as a diagnostic strategy.** Train the RF term first (already
   done, canonical checkpoint), then freeze it and train the local term
   against the resulting residual. `RESEARCH_LOG.md` Section 24 ran exactly
   this for the `linear_t` shifted Gaussian: it confirmed RF/local
   co-adaptation as a real effect (joint training learns the wrong sign;
   frozen-RF training learns the sign the fixed-RF objective itself
   predicts) and modestly improved the solution. The remaining limitation
   is magnitude, not sign (Section 25) — this strategy diagnoses
   co-adaptation well but has not by itself produced a large accuracy
   improvement.

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
    Section 9 plotting fix — see `TODO.md`.

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

16. **Conceptual distinctions clarified by `RESEARCH_LOG.md` Sections
    21–25.** Several previously conflated questions turned out to be
    separate and must be evaluated independently for any local feature:
    - *activation* — does the local coefficient move away from zero at all?
    - *sign/direction* — does it move in the direction that would actually
      reduce error?
    - *amplitude* — does it move far enough to matter, given the training
      objective's own optimum?
    - *representation capacity* — can the chosen local basis (e.g. one
      Gaussian) represent the needed correction shape at all?
    - *co-adaptation* — does joint optimization of the RF and local terms
      together change the outcome relative to training the local term
      against a fixed RF?
    A local feature can activate strongly with the wrong sign (Section 21),
    or activate weakly but with the correct sign (Section 24) — these are
    different failure modes requiring different fixes.

17. **`WEIGHT_DECAY=1` suppressing the local amplitude — now tested; a
    real but partial contributor, not a full explanation.** `RESEARCH_LOG.md`
    Section 28 ran the controlled `weight_decay in {1.0,0.1,0.01,0.0}`
    sweep this hypothesis called for. Result: reducing weight decay does
    measurably grow both amplitudes and improve PDE loss/solution error,
    and the effect saturates by `weight_decay≈0.01` — so weight decay is a
    real contributor. However, even at `weight_decay=0` both amplitudes
    remain far from their post-hoc oracle values, and the two amplitudes
    are affected asymmetrically (`d2` reaches ~33% of oracle, `d1` only
    ~3.5%). **Do not treat weight decay as the cause of the amplitude
    problem** — it explains only part of it, and does not explain the
    `d1`/`d2` asymmetry at all.

18. **Conceptual framework, extended after Sections 26–28.** The
    distinctions in item 16 (activation / sign-direction / amplitude /
    representation capacity / co-adaptation) are now joined by two more,
    needed to describe today's results:
    - *regularization* — does an explicit regularizer (e.g.
      `WEIGHT_DECAY`) suppress the learned amplitude independently of the
      PDE/IC/BC objective itself? (Section 28: yes, partially.)
    - *objective mismatch* — does the PDE-residual training objective have
      a genuinely different preferred correction (in sign, magnitude, or
      balance between multiple local terms) than the one that minimizes
      solution error, even when representation, direction, co-adaptation,
      and regularization are all controlled for? (Section 27/28: with
      representation (strong), direction (correct), co-adaptation (RF
      frozen, ruled out), and regularization (tested, partial and
      asymmetric) progressively isolated, the remaining `d1`/`d2`
      amplitude asymmetry currently points toward an objective mismatch —
      i.e., the PDE/IC/BC objective itself may simply prefer a different,
      weaker, and asymmetric correction between the two Gaussians than the
      `t=1` solution-error oracle does.)

    **This is the current leading hypothesis, not an established
    conclusion** — Section 29's updated central question (why the
    objective favors `d2` far more than `d1`) has not yet been directly
    tested (e.g. by comparing per-Gaussian PDE-loss gradients, or mapping
    the PDE objective's optimum in the full `(d1,d2)` plane). Do not
    overstate causality beyond what has actually been tested.

19. **Multiscale / two-population RF basis — new hypothesis from the
    2026-10-07 meeting, not yet tested.**
    ```
    u_RF = sum_{k=1}^{N1} a_k cos(w_k^(1) x + b_k^(1))
         + sum_{k=1}^{N2} beta_k cos(w_k^(2) x + b_k^(2))

    w_k^(1) ~ N(0, sigma_1^2),  w_k^(2) ~ N(0, sigma_2^2),  N = N1 + N2
    ```
    Motivation: one RF population represents smoother spatial structure
    while a second, different frequency scale provides additional
    resolution around the sharp shock. **This is a hypothesis, not an
    established conclusion.** Candidate `N1,N2` values from the advisor's
    handwritten notes are TBD / need confirmation before implementation —
    do not invent specific values. See `TODO.md` item E.

20. **Interaction between global RF capacity (N) and local Gaussian
    enrichment (K) — raised at the 2026-10-07 meeting.** The RF
    basis-size study (`TODO.md` items A/B) and the `K>=8` Gaussian-
    enrichment study (`TODO.md` item C) are not fully independent:
    increasing `N` could change the RF-only residual's shape/magnitude
    near the shock, which would in turn change how much
    representation/training benefit a K-Gaussian local correction
    provides. The exact N-by-K experimental design is explicitly **TBD**
    (`TODO.md` item C) — this item records the conceptual interaction, not
    a committed experimental matrix.

21. **Proposed diagnostic axes for the amplitude/asymmetry question,
    extending item 18 after the 2026-10-07 meeting.** A fully controlled
    future diagnostic should aim to separate FOUR axes, not just the two
    (representation, regularization) already tested:
    - *representation capacity* — already strongly established (Section
      26);
    - *PDE-objective preference* — what the PDE/IC/BC objective's own
      optimum actually is in the full `(d1,...,dK)` space, independent of
      how it is reached by an optimizer;
    - *regularization* — already partially tested (Section 28);
    - *optimization dynamics* — whether Adam (or its hyperparameters)
      actually reaches the PDE objective's own optimum, or stalls/diverges
      from it for other reasons.
    Section 29's asymmetry finding is consistent with a PDE-objective-
    preference explanation, but has not yet been separated from an
    optimization-dynamics explanation. **This is a design sketch, not a
    committed experiment.**

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
