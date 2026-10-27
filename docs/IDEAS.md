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
   direction (2026-10-07 meeting, `docs/meetings/2026-10-07.md`); now
   DEPRIORITIZED per the 2026-10-14 meeting.** Extends the earlier
   "trainable center and width" idea to also train the amplitude `d`
   jointly through the PDE objective: `{c, sigma, d}` for one Gaussian,
   generalizing to `{c_k, sigma_k, d_k}` for K Gaussians. Still a valid
   future idea, retained here, but explicitly postponed until Phase A (RF
   baseline quality) is resolved — see `RESEARCH_LOG.md` Section 35 and
   `TODO.md` item H.

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

19. **Multiscale / two-population RF basis — hypothesis from the
    2026-10-07 meeting, now TESTED twice (two directions, see
    `RESEARCH_LOG.md` Sections 34.1/34.2), with a refined follow-up
    direction from the 2026-10-14 meeting.**
    ```
    u_RF = sum_{k=1}^{N1} a_k cos(w_k^(1) x + b_k^(1))
         + sum_{k=1}^{N2} beta_k cos(w_k^(2) x + b_k^(2))

    w_k^(1) = z_k^(1) / sigma_1,  z_k^(1) ~ N(0,1)   =>  std(w^(1)) = 1/sigma_1
    w_k^(2) = z_k^(2) / sigma_2,  z_k^(2) ~ N(0,1)   =>  std(w^(2)) = 1/sigma_2
    N = N1 + N2
    ```
    (Matches the actual RF implementation's inverse-frequency-scale
    convention, `W = Z/sigma`, `std(W) = 1/sigma` — see `RESEARCH_LOG.md`
    Section 34's explicit statement of this convention.)
    Results so far (both with `N1=N2=250`, population 1 fixed at the
    canonical frequency std=25): an **aggressive** second population
    (std=312.5) substantially *worsened* global/smooth/shock/PDE metrics
    (Section 34.1); a **moderate** second population (std=75) *improved*
    all four metrics simultaneously relative to the canonical
    single-population baseline — the first controlled positive multiscale
    result (Section 34.2). **Refined hypothesis (not yet tested):** useful
    frequency scales and population allocations should be selected
    systematically, and ideally informed by PDE structure, rather than
    chosen arbitrarily — per the 2026-10-14 advisor direction. Do **not**
    treat std=75 as the correct or optimal scale; the N1/N2 allocation and
    total-N dependence are also still open. See `TODO.md` item E.

20. **Interaction between global RF capacity (N) and local Gaussian
    enrichment (K) — raised at the 2026-10-07 meeting.** The RF
    basis-size study (Section 31) and the `K=8` Gaussian-enrichment study
    (Section 33) are not fully independent: increasing `N` could change
    the RF-only residual's shape/magnitude near the shock, which would in
    turn change how much representation/training benefit a K-Gaussian
    local correction provides. The exact N-by-K experimental design
    remains **TBD** — this item records the conceptual interaction, not a
    committed experimental matrix. **Now secondary to Phase A** (see item
    22 below) — relevant again once the `u_RF` baseline itself is
    understood.

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
    committed experiment.** Section 33's K=8 result (worse PDE-trained t=1
    shock error than K=2, despite far stronger post-hoc capacity) is
    additional evidence this question is still open. **Now secondary to
    Phase A** (see item 22 below).

22. **Phase A / Phase B framing — 2026-10-14 advisor direction
    (`docs/meetings/2026-10-14.md`, `RESEARCH_LOG.md` Section 35).** The
    advisor's central assessment is that the current `u_RF` itself is not
    sufficiently accurate, and that a well-constructed RF approximation
    should reproduce the smooth part of the solution well, with the
    principal difficulty concentrated near the shock. All local-correction
    work (items 3, 6, 8, 11, 13, 14, 20, 21 in this file) is therefore
    **secondary** until a satisfactory smooth-region `u_RF` baseline is
    established (Phase A: multi-seed testing, systematic Nx/Nt study,
    learning-rate diagnostic, systematic multiscale study, joint-vs-
    separable space-time RF comparison — see `TODO.md` current priority).
    This does not invalidate the local-correction findings so far, it
    reorders priority.

23. **Joint space-time RF formulation — new idea from the 2026-10-14
    meeting, UNTESTED.** Replace the current separable product features
    ```
    phi_ij(x,t) = cos(wx_i*x+bx_i) * cos(wt_j*t+bt_j)
    u_RF(x,t) = sum_i sum_j c_ij phi_ij(x,t)
    ```
    with a single joint cosine over concatenated `(x,t)`:
    ```
    phi_k(x,t) = cos(wx_k*x + wt_k*t + b_k)
    u_RF(x,t) = sum_k c_k phi_k(x,t)
    ```
    Proposed as an idea to test, not an established improvement. A fair
    coefficient-budget comparison between the separable `Nx*Nt`
    representation and a joint `N`-feature representation is not automatic
    and is explicitly **TBD** — do not invent the comparison protocol yet.
    See `TODO.md` item F.

24. **Multi-seed / Nx-Nt joint convergence question — 2026-10-14 meeting,
    now TESTED (see `RESEARCH_LOG.md` Sections 36–37).** The RF basis-size
    sweep (Section 31) used only seed=42 with Nt fixed at 10. **Tested
    finding:** seed 42 is not a particularly poor realization (ranks
    2nd-best of six by global error among seeds 20–24 and 42, Section 36),
    though genuine seed-to-seed variability is substantial (~34% relative
    range in global error). The fixed-ratio Nx/Nt=50 sweep through
    `(3000,60)` (Section 37) confirms the degradation is **not** simply a
    property of varying Nx alone — jointly scaling Nx and Nt still gives
    non-monotonic error with no systematic improvement beyond the
    canonical scale, with `(750,15)` as a near-exception. **Remaining
    untested hypothesis:** whether a different seed, training budget, or
    learning rate changes this picture at larger `(Nx,Nt)` — not yet
    separated from the capacity question itself. `(3500,70)`/`(4000,80)`
    were never run. See `TODO.md`'s "Completed" list.

25. **Smaller-learning-rate diagnostic — 2026-10-14 meeting, now TESTED
    (see `RESEARCH_LOG.md` Section 38).** Test whether a smaller learning
    rate changes the RF basis-size sweep's finding (Section 31), as a
    controlled optimization diagnostic kept isolated from other
    architecture changes. **Tested finding:** at a fixed 1000-epoch
    budget, LR in {2e-4, 1e-4, 5e-5} are all worse than canonical LR=5e-4,
    but none have plateaued — all three are still improving substantially
    from epoch 500 to 1000. **Remaining untested hypothesis, not yet
    run:** whether a smaller LR would match or beat canonical given a
    longer training budget (explicitly out of scope for Section 38, which
    was capped at 1000 epochs by design). See `TODO.md`'s "Completed"
    list.

26. **Boundary-condition caveat for local Gaussian formulations —
    clarified at the 2026-10-14 meeting.** For
    `u_local(x,t) = t*sum_k d_k G_k(x)`, `u_local(x,0)=0` holds exactly,
    but the Gaussian local correction does **not** analytically satisfy
    the Burgers boundary condition at `x=±1`. For narrow Gaussians centered
    near the shock this is numerically negligible (e.g. `G(±1)` underflows
    to exactly `0.0` in float64 at the widths used so far, Section 23),
    but it is not identically zero by construction. Relevant to any future
    local-feature formulation, not an action item in itself.

27. **Self-adaptive collocation-point weighting — 2026-10-21 advisor
    direction (`docs/meetings/2026-10-21.md`, `RESEARCH_LOG.md` Section
    40), UNTESTED.** The professor pointed to a self-adaptive PINN method
    (paper reference not yet supplied — placeholder) that assigns a weight
    to each collocation point and adjusts those weights during training,
    with a reported viscous Burgers benchmark (the paper's Section 4.1).
    Candidate idea for this project: investigate whether an analogous
    adaptive per-collocation-point weighting (or another loss-function
    modification motivated by the same principle) could help the RF model
    — directly motivated by the repeatedly observed mismatch between PDE
    residual reduction and actual solution-error reduction (`RESEARCH_LOG.md`
    Sections 27–29, 33, 39). **This is entirely untested and depends on
    first reading the paper and reproducing its own Burgers result (see
    `TODO.md` items P1/P2) before any adaptation to the RF model is
    attempted.** Do not invent the paper's algorithm or hyperparameters.

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
