# Burgers RF Research Log

Scientific/technical chronology for the random-feature (RF) + local-correction
study of the viscous Burgers equation, on branch `burgers-local-features`.
Structure per entry: **Hypothesis → Experiment → Result → Interpretation**.

This log preserves the actual progression of the work, including negative
results and places where an earlier interpretation was later refined — it is
not rewritten to look more linear than it was.

---

## 0. Problem and baseline configuration

**Problem.** Viscous Burgers equation

```
u_t + u u_x - nu u_xx = 0,   x in [-1,1], t in [0,1], nu = 0.01/pi
u(-1,t) = u(1,t) = 0
u(x,0) = -sin(pi x)
```

**Baseline RF model.**

```
u_RF(x,t) = sum_i sum_j c_ij phi_i^x(x) phi_j^t(t)
```

**Default controlled configuration** used throughout (unless an experiment
deliberately varies one factor):

- float64, seed = 42
- Nx = 500, Nt = 10
- 3600 interior collocation points, 500 boundary points, 500 IC points
- 1000 epochs, LR = 5e-4, weight decay = 1, IC/BC weight = 1000
- Canonical shock evaluation region: `|x| <= 0.02`

Package code: `Burgers/burgers_rf/{config,data,model,physics}.py`.
Repository setup/refactor: `9b4b90d` (baseline notebooks), `2045bd6`
(refactor into package), `aa0af66` (show solution plots), `3727ebe` (save
experiment outputs), `18330f5` (reproducible seeding), `e7cdf60` (ignore
generated outputs).

**Baseline seed-42 result** (no local feature):

```
global relative L2 ~ 0.27149430095802507
shock  relative L2 ~ 0.8390487998889169
```

This is the number every later experiment is compared against.

---

## 1. Local Gaussian feature, width sweep (centered, standard temporal form)

**Hypothesis.** A localized correction near the shock,
`u_local = G_sigma(x) a(t)` with `G_sigma(x) = exp(-(x-center)^2/(2 sigma^2))`
and `a(t) = sum_j d_j phi_j^t(t)`, centered at `center=0`, could reduce the
shock-region error left by the global RF term.

**Experiment.** Width sweep over `sigma in {.02, .05, .10, .20, .30, .40, .50}`,
center fixed at 0.
Scripts: `Burgers/burgers_rf/model.py` (Gaussian local feature, `df59c0d`),
`Burgers/scripts/sweep_local_width.py` (`e2b72ee`, canonical shock metric
added in `9980d49`).

**Result:**

| sigma | global | shock |
|---|---|---|
| baseline (no local) | 0.271494 | 0.839049 |
| .02 | 0.322363 | 1.096815 |
| .05 | 0.271359 | 0.838750 |
| .10 | 0.270981 | 0.837707 |
| .20 | 0.270522 | 0.836578 |
| .30 | 0.271535 | 0.838894 |
| .40 | 0.271618 | 0.839296 |
| .50 | 0.271527 | 0.838652 |

**Interpretation.** sigma=.20 gave the best (marginal) improvement; sigma=.02
was clearly *detrimental*, substantially worse than baseline on both metrics.

A separate physical shock-width diagnostic (`diagnose_reference_gradient.py`,
`bbed61e`/`f2f4bfa`) estimated the analytic solution's characteristic shock
half-width at roughly 0.005–0.015, depending on time/threshold.

**Important early finding:** the empirically best Gaussian width (~0.20) was
*much broader* than the physical shock width, while sigma=.02 — close to the
physical width — was the worst case tested. Direct physical-width matching
was therefore **not** a successful design rule on its own.

---

## 2. Multi-Gaussian widths

**Hypothesis.** A sum of Gaussians at several widths might capture more of
the needed correction than a single width.

**Experiment.** `local_type="gaussian_sum"`, widths `[.05, .10, .20]`.
Script: `Burgers/scripts/test_gaussian_sum.py` (`a59a5aa`).

**Result:** global ~0.271118, shock ~0.838009.

**Interpretation.** Slightly better than the plain baseline, but *worse*
than the best single width (sigma=.20 alone). Combining widths did not
compound the benefit.

---

## 3. Longer training at sigma=.02

**Hypothesis.** sigma=.02's poor result might be an optimization-duration
problem rather than a representational one.

**Experiment.** Continuous training trajectory at sigma=.02, checkpointed at
1000 and 10000 epochs. Script: `Burgers/scripts/test_epoch_schedule_sigma0.02.py`
(`43385c9`).

**Result:**

| epochs | global | shock | ratio_shock (local/RF) |
|---|---|---|---|
| 1000 | 0.32236 | 1.09681 | 0.15040 |
| 10000 | 0.32499 | 1.10911 | 0.11414 |

**Interpretation.** No rescue — 10x more training made the result marginally
*worse*, and the local feature's relative contribution in the shock region
*decreased*, not increased.

---

## 4. Collocation diagnostics and collocation-density experiments

**Collocation diagnostic.** The fixed seed-42, 3600-point uniform collocation
set contains **76** points with `|x| <= 0.02`.
Script: `Burgers/scripts/diagnose_collocation_points.py`.

*Note on this number:* the diagnostic script's first version (`5745ca4`)
reported **77** due to a float32/float64 dtype inconsistency relative to
every training script (which run in float64); corrected to **76** in
`6d9d144` after the discrepancy was traced and verified. This episode is
preserved here because it affected how the two collocation-density
experiments below were interpreted at the time (expected totals of "720" vs.
the corrected "719").

**Hypothesis.** If the shock region is under-sampled by the fixed collocation
set, increasing collocation density there might let the local feature
activate.

**Experiment A — redistribution** (same total 3600 points, 20% redistributed
into the shock band → 720 shock points). Script:
`Burgers/scripts/test_collocation_density.py` (`44602e9`).

**Result:** global 0.338997, shock 1.077562. Shock error improved slightly
relative to the bad sigma=.02 baseline case, but global/smooth error
worsened, and the local contribution *decreased*.

**Experiment B — additive refinement** (original 3600 points preserved
unchanged, +643 points added in the shock band → 4243 total, actual shock
count 719, matching the corrected 76-point diagnostic above: 76+643=719).
Script: `Burgers/scripts/test_collocation_additive.py` (`f900246`).

**Result:** global 0.335328, shock 1.098753. No rescue.

**Interpretation.** Neither redistributing nor additively enriching shock
collocation density activated the local feature or improved the solution —
density alone is not the bottleneck.

---

## 5. Very-small-sigma sweep

**Hypothesis.** Perhaps sigma=.02 is simply too large/too small in some
other sense; sweep much narrower (and one broader) widths to characterize
the pattern.

**Experiment.** `sigma in {.0001, .0005, .0025, .0125, .0625, .3125}`, original
uniform collocation, canonical + fine (2001-pt) shock grids. Script:
`Burgers/scripts/sweep_local_width_small_sigma.py` (`41ae778`).

**Result:**

| sigma | global | shock | ratio_shock (local/RF) |
|---|---|---|---|
| .0001 | .270584 | .837857 | ~9.62e-05 |
| .0005 | .271843 | .841867 | ~.000671 |
| .0025 | .271587 | .840957 | ~.000786 |
| .0125 | .311738 | 1.050522 | ~.04699 |
| .0625 | .270168 | .835542 | ~.14034 |
| .3125 | .270831 | .837501 | ~.07341 |

Collocation counts within 3·sigma (from the corrected 76-point-consistent
diagnostic):

| sigma | count within 3·sigma |
|---|---|
| .0001 | 1 |
| .0005 | 6 |
| .0025 | 35 |
| .0125 | 141 |

**Interpretation.** For the narrowest widths (.0001–.0025) the local
feature's relative contribution (`ratio_shock`) is essentially negligible
(<0.1%), and error is close to baseline — consistent with near-zero
collocation support at those scales. sigma=.0125 reproduces the same kind of
degradation seen at sigma=.02.

---

## 6. Resolution-controlled small-sigma experiment

**Hypothesis.** Directly fix the under-resolution found above: add just
enough points so each small sigma has exactly 150 collocation points within
`|x| <= 3 sigma`, without touching the rest of the original set.

**Experiment.** `sigma in {.0001, .0005, .0025}` resolved to 150 points in
3·sigma each, 1000 epochs. Script:
`Burgers/scripts/sweep_local_width_small_sigma_resolved.py` (`3a7cf43`).

**Result (representative, sigma=.0025):** global .317527, shock 1.04408,
ratio_shock .000241.

**Interpretation.** No activation — resolving the collocation density to a
nominally "adequate" level did not move the local coefficient's relative
contribution meaningfully above the unresolved case.

**Follow-up — 10,000-epoch resolved sigma=.0025.** Combines the resolution
fix above with the long-training approach from Section 3. Script:
`Burgers/scripts/test_small_sigma_resolved_long_training.py` (`b5da981`).

**Result:** global ~.32481, shock ~1.07294, ratio_shock ~9.26e-05.

**Interpretation.** No delayed activation even with both more resolution and
10x more training combined.

---

## 7. Implementation and conditioning audit

**Hypothesis.** Given five consecutive negative results (width, multi-width,
duration, density, resolution), check whether there is an actual
implementation bug suppressing the local coefficients, rather than a
genuine representational/optimization limitation.

**Experiment.** Line-by-line audit of `model.py`/`physics.py`'s local-feature
computation path, plus direct numerical checks (gradient at initialization,
manual-coefficient forward sanity test, parameter registration). This audit
was conducted as ad hoc diagnostic analysis (not saved as a standalone
repository script).

**Result — no implementation bug found:**
- `local_coefficients` are genuine `nn.Parameter` objects and are included
  in the optimizer.
- The Gaussian formula is implemented correctly (verified against
  `exp(-1/2)`, `exp(-2)`, `exp(-9/2)` to floating-point precision).
- Manually setting nonzero coefficients and checking `u_total = u_RF +
  u_local` reproduces the expected values exactly.
- The PDE residual produces nonzero gradients with respect to the local
  coefficients at initialization.

**Conditioning analysis.** For the Gaussian feature,

```
G_x  = -(x/sigma^2) G
G_xx = (x^2/sigma^4 - 1/sigma^2) G,   so G_xx(0) = -1/sigma^2
```

Very narrow features therefore introduce severe derivative scaling
(`1/sigma^2`) directly inside the Burgers PDE residual.

**IC gradient decomposition.** For sigma=.0025, the (`IC_BC_WEIGHT`-)
weighted IC-loss gradient with respect to the local coefficients initially
*exceeded* the PDE-residual-driven local gradient in magnitude.

**Interpretation.** No bug; the negative results above are real. Two
concrete, quantified suspects emerge: (a) `1/sigma^2` derivative scaling for
narrow widths, and (b) IC-loss gradient dominance over the PDE-residual
gradient for the local coefficients.

---

## 8. IC-weight experiment

**Hypothesis.** If the IC loss's gradient dominates the local coefficients'
gradient (Section 7), reducing or removing the IC weight should let the
PDE-residual-driven signal activate the feature.

**Experiment.** sigma=.0025, `IC_WEIGHT in {1000, 100, 10, 0}` (BC weight
held at 1000). Script: `Burgers/scripts/test_ic_weight_sigma0.0025.py`
(`4bc0b3b`).

**Result:**

| IC weight | global | shock |
|---|---|---|
| 1000 | .271588 | .840958 |
| 100 | .273771 | .841650 |
| 10 | .313992 | .943526 |
| 0 | .961180 | 1.013653 |

(At IC=1000: IC error .002674, `\|\|d\|\|` .003932, ratio_shock .000786.)

**Interpretation.** Reducing or removing the IC weight did **not** activate
the local feature — and degraded the overall solution substantially (IC=0
gives global error close to 1, i.e. badly broken). The IC term's gradient
share is not simply "stealing" the local feature's budget in a way that
reducing it frees up; removing it is actively harmful.

---

## 9. Direct spatial-fit diagnostic (centered Gaussian, t=1)

**Hypothesis.** Separate the question of *representation* from *PDE
optimization* entirely: remove PDE training, Adam, and IC/BC losses, and
directly least-squares-fit a single Burgers snapshot (t=1) using the exact
seed-42 spatial RF realization, with and without one centered Gaussian
column.

**Experiment.** Two formulations (minimum-norm, `M_fit=250 < Nx`; dense
overdetermined LS, `M_fit=4001 >> Nx`), `sigma in {.0025, .02, .20}`,
`center=0`. Script: `Burgers/scripts/test_direct_spatial_fit.py` (`5d42737`,
plot-scaling corrected in `b379fe9`).

**Result — dense-LS RF-only baseline (no Gaussian):**

```
global = .082382
shock  = .36692
```

**Centered Gaussian, representational novelty relative to the RF span**
(`residual_g = ||G - P_RF G|| / ||G||`, where `P_RF` is the least-squares
projection onto the RF column space):

| sigma | residual_g (novelty) |
|---|---|
| .0025 | ~.885 |
| .02 | ~.167 |
| .20 | ~0 (exactly redundant with RF span) |

**Result — benefit of the centered Gaussian:** despite sigma=.0025 supplying
a direction that is ~88.5% *outside* the numerically-retained RF span (i.e.
genuinely novel), it provided **almost no approximation benefit** when
added to the RF-only fit.

**Separate numerical finding:** the 500-column spatial RF design matrix has
effective numerical rank ~64 under the default SVD tolerance (strongly
ill-conditioned — this is a property of the random-feature construction
itself, not of this diagnostic).

**Plotting correction (`b379fe9`):** an early version of this script's plots
used shared y-axis limits pooled across all three sigma values, so the
sigma=.20 case's pathological component magnitudes (a consequence of
`residual_g ~ 0`, i.e. a numerically unidentifiable augmented fit) visually
compressed the sigma=.0025/.02 plots to look flat/near-zero. After the fix,
the true `O(1)` solution curves and Gibbs-like oscillation/overshoot near
the shock were visible. The underlying numerical (CSV) results were
**unchanged** by this fix — only the plots had been misleading.

**Interpretation.** A centered narrow Gaussian supplies a mostly-novel
direction relative to the RF span, but that direction is apparently not
well-aligned with what the *direct-fit* target actually needs at `center=0`
— novelty of the feature and usefulness for the specific target are
different things. (This interpretation was later refined — see Section 13.)

---

## 10. IC-compatible local feature

**Hypothesis.** Make the local feature structurally unable to interfere with
the initial condition, removing any possible IC/local-feature competition
found in Section 7 by construction rather than by reducing the IC weight
(which Section 8 showed is harmful).

**Experiment.** New `local_time_mode="ic_compatible"`:
`u_local = G_sigma(x) * sum_j d_j [phi_j^t(t) - phi_j^t(0)]`, guaranteeing
`u_local(x,0) = 0` identically for any coefficients. Compared against the
original ("standard") temporal form, sigma=.0025. Code: `Burgers/burgers_rf/model.py`
(`cf396ad`); script: `Burgers/scripts/test_ic_compatible_local_feature.py`.

**Result:**

| mode | global | smooth | shock | IC error | coefficient norm (L2) | ratio_shock |
|---|---|---|---|---|---|---|
| standard | .27158737 | .24437060 | .84095736 | .0026754 | .0039318 | .0007861 |
| ic_compatible | .27157725 | .24441936 | .83997841 | .0015433 | .0066267 | .0009168 |

**Interpretation.** The coefficient norm increased substantially (`.0039`→
`.0066`) and IC error dropped as designed, but the local feature's relative
contribution in the shock region remained below ~0.1% of the RF magnitude.
**Direct IC-loss competition was therefore not the primary cause** of the
near-zero local contribution — removing it by construction didn't change
the outcome materially.

---

## 11. Explicit shock-region loss weighting

**Hypothesis.** Advisor suggestion (see `meetings/2026-09-23.md`): increase
the PDE-loss weight of existing shock-region collocation points, rather than
changing their number/density (Section 4) or the IC treatment (Sections 8,
10).

**Experiment.** Same original 3600 collocation points (no enrichment),
ic_compatible local feature, sigma=.0025, piecewise-constant weight
`w_i = alpha` for `|x_i| <= 0.02` else `1`, normalized weighted PDE loss,
`alpha in {1, 2, 5, 10, 20}`. Script:
`Burgers/scripts/test_shock_point_weighting.py` (`3f26833`).

Effective shock-region share of the normalized PDE objective by alpha:
~2.1%, 4.1%, 9.7%, 17.7%, 30.1% (alpha=1 through 20).

**Result:**

| alpha | global | shock | ratio_shock |
|---|---|---|---|
| 1 | .271577 | .839978 | .0009168 |
| 20 | .338139 | 1.053570 | .0003807 |

**Interpretation.** Explicit shock-region weighting did **not** activate the
feature — and at alpha=20, both global and shock accuracy *worsened*
substantially while the local feature's relative contribution *decreased*.
This directly tests the professor's first 09/23 suggestion and provides
evidence against simple fixed regional weighting as a standalone fix for
the current formulation.

---

## 12. Simple temporal local feature (`linear_t`)

**Hypothesis.** Advisor suggestion (see `meetings/2026-09-23.md`): simplify
the temporal part of the local feature away from a 10-dimensional random-
feature expansion, to the simplest possible deterministic form.

**Experiment.** New `local_time_mode="linear_t"`:
`u_local = d * t * G_sigma(x)`, exactly one trainable scalar `d` (replacing
10 trainable `d_j`), automatically IC-compatible (`u_local(x,0)=0`
identically since the factor is `d*t`). Compared against `ic_compatible`,
sigma=.0025. Code: `Burgers/burgers_rf/model.py` (`c753ad2`); script:
`Burgers/scripts/test_simple_time_local_feature.py`.

**Result:** global .27158712, smooth .24444595, shock .83973770, IC
.00152144, d = .000513716, ratio_shock .000836685.

**Interpretation.** No meaningful activation. **Temporal RF complexity is
not the primary cause** of the near-zero local contribution either —
reducing the temporal representation from 10 random features to a single
linear scalar changed almost nothing about the outcome.

---

## 13. Fixed/shifted Gaussian direct diagnostic — refining the Section 9 conclusion

**IMPORTANT CAVEAT, preserved throughout this section:** this diagnostic
uses the **direct least-squares-fitted RF baseline at t=1** from Section 9
(`test_direct_spatial_fit.py`'s dense-LS fit), **not** the actual
PDE-trained RF solution. Conclusions here are about representational
capacity against that direct-fit baseline, not about what the PDE-trained
model's residual needs.

**Hypothesis.** Section 9 tested only a *centered* (`center=0`), *unit-
amplitude* Gaussian. Test whether shifting the center and/or optimally
scaling the amplitude changes the conclusion.

**Experiment.** Grid search over center `c` and width `sigma`, three
corrections per grid point: fixed `+G`, fixed `-G` (no fitting), and the
closed-form least-squares-optimal scalar amplitude
`d* = <q, u_true - u_RF> / <q,q>` (not training, not an optimizer). Baseline
reused unchanged from Section 9. Script:
`Burgers/scripts/test_fixed_gaussian_correction.py` (uncommitted at time of
documentation reconstruction).

**Baseline (same as Section 9):** global .0823820, smooth .0484601, shock
.366923.

**Result — fixed unit amplitude:** fixed `+G` and `-G` **never** improved
the result at any tested `(c, sigma)`.

**Result — optimal amplitude, best grid point:**

```
center c = .010, sigma = .005477
d_global* ~ -.2854,  d_shock* ~ -.2835
global error ~ .067753/.067754  (~17.8% improvement vs baseline)
smooth error ~ .047927/.047929
shock error  ~ .265913/.265918  (~27.5% improvement vs baseline)
correlation: rho_global ~ -.569, rho_shock ~ -.689
```

Near the sigma actually used in the PDE-training experiments above
(sigma ~ .00281, closest grid point to .0025), at the same shifted center
`c=.010`: `d_shock* ~ -.349`, shock error ~ .291102 (~20.7% improvement).

**Updated interpretation — refines, does not invalidate, Section 9:** the
Section 9 finding (a *centered* narrow Gaussian gives almost no benefit) is
not evidence that Gaussian functions are fundamentally unsuitable as a local
basis. A **shifted and appropriately (negatively) scaled** Gaussian can
represent a substantial portion of the *direct-fit* RF residual — roughly a
quarter reduction in shock-region error — including at widths close to
sigma=.0025. The earlier "Gaussian is a poor shape match" conclusion was too
broad. The direct-fit diagnostic shows that center and amplitude strongly
affect whether a Gaussian aligns with the RF residual. Whether these were
also limiting factors in PDE training remains unresolved.

This does **not** prove the same shifted Gaussian is useful for the actual
PDE-trained RF model — the caveat above applies.

---

## 14. Current central question

**What local spatial structure is actually required by the error left by
the PDE-trained RF solution** — as opposed to the error left by the
direct-fit RF approximation used in Sections 9 and 13?

This is the open question carried into `TODO.md`.
