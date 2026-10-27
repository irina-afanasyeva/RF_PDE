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
`Burgers/scripts/test_fixed_gaussian_correction.py` (62156ef).

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

This is the open question carried into `TODO.md`. (Answered in part by
Sections 15–19 below; see Section 20 for the updated question.)

---

## 15. PDE-trained RF baseline and residual diagnostic

**Hypothesis.** Resuming `TODO.md` Task 1: before designing any new local
feature, analyze the residual `e_RF(x,t) = u_true(x,t) - u_RF(x,t)` of the
*actual PDE-trained* RF baseline — not the direct-fit approximation used in
Sections 9 and 13.

**Experiment.** Reproduced the seed-42 RF-only baseline (`use_local=False`)
via the existing, unmodified training path, saved a checkpoint, and
inspected the residual on a dense 4001-point grid at `t = 0.25, 0.5, 0.75,
1.0`. Script: `Burgers/scripts/test_pde_trained_rf_baseline.py` (`5c29cd3`).

**Result — canonical multi-time metrics** (consistent with Section 0's
global/shock numbers, now with the smooth split also confirmed):
```
global = 0.27149430095802507
smooth = 0.2443827084159696
shock  = 0.8390487998889169
```

**Result — residual structure.** At later times the residual develops a
localized, sign-changing structure straddling the shock: positive on the
left, negative on the right. At `t=1`:
```
max positive residual ≈ +0.36925 at x ≈ -0.0145
min negative residual ≈ -0.59707 at x ≈ +0.0135
```

**Interpretation.** This two-lobe, odd-ish residual shape around the shock
means a single **centered, even** Gaussian cannot represent the complete
sign-changing correction by itself. A single Gaussian has one sign (up to
its scalar amplitude), so it cannot simultaneously reproduce the
positive-left and negative-right residual lobes. It can therefore provide
only a partial correction to this sign-changing structure. This directly
motivates testing a *shifted* Gaussian (Section 17) and, ultimately, a
*pair* of Gaussians (Section 20 / `TODO.md`).

---

## 16. Checkpoint reconstruction — missing Wx/Wt/bx/bt

**Discovery.** The checkpoint saved by Section 15's script on the GPU server
contained only `model.0.weight` (the trained `c_ij` coefficients) — `Wx`,
`Wt`, `bx`, `bt` (the fixed random spatial/temporal frequencies and biases)
were absent from `state_dict()`.

**Root cause.** `model.py` constructs these fixed tensors with the pattern
`nn.Parameter(...).to(device)`. Calling `.to(device)` directly on an
`nn.Parameter` instance is a no-op (type preserved) when no actual
conversion is needed, but constructs a **plain `torch.Tensor`** (silently
losing the `Parameter` wrapper) when a genuine conversion occurs — which
happens on the GPU server (`device='cuda:0'`) but not on CPU-only machines.
Since this conversion happens *inside* `__init__`, before
`nn.Module.__setattr__` ever sees these as `Parameter` instances, they are
never registered in `self._parameters` on CUDA, hence absent from
`state_dict()`. **This affects checkpoint completeness only — not the
forward computation or the validity of any previous training run**:
`Wx/Wt/bx/bt` still held their correct, deterministically-seeded values and
were used correctly throughout training (`requires_grad=False`, never meant
to be optimized; this was simply the first experiment to rely on
`state_dict()` completeness).

**Reconstruction.** Reproduced the exact pre-model RNG sequence from the
checkpoint's own recorded seed/config metadata:
```
set_seed(42)
sample_training_points(M_TRAIN=3600, device="cpu")
BurgersRF(NX=500, NT=10, SIGMA_X=0.04, SIGMA_T=1/3, use_local=False, device="cpu")
```
then loaded *only* the saved `model.0.weight` on top — `Wx/Wt/bx/bt` were
never loaded from the checkpoint, only reconstructed from the seed. Script:
`Burgers/scripts/reconstruct_pde_trained_rf_checkpoint.py` (`e7deefa`).

**Verification.** Recomputed canonical metrics matched the Section 15
values essentially to machine precision:
```
global = 0.271494300958025   (canonical 0.27149430095802507)
smooth = 0.2443827084159696  (canonical 0.2443827084159696)
shock  = 0.8390487998889169  (canonical 0.8390487998889169)
```
A complete checkpoint was saved (original left untouched) at:
```
Burgers/outputs/pde_trained_rf_residual/checkpoint_complete.pt
```
containing `Wx, Wt, bx, bt, model.0.weight`.
`Burgers/scripts/test_pde_trained_gaussian_correction.py` was updated to
load this complete checkpoint instead (`2fa4a91`).

**Follow-up (not yet done, tracked in `TODO.md`):** fixing the underlying
`nn.Parameter(...).to(device)` registration pattern in `model.py`. This does
**not** retroactively invalidate any previous numerical result in this log —
it only affects checkpoint persistence on CUDA, not the forward computation
any experiment actually used.

---

## 17. PDE-trained single-Gaussian correction diagnostic

**Hypothesis.** Repeat the Section 13 shifted/scaled Gaussian correction
search, but against the *actual* PDE-trained RF (Section 15/16's
reconstructed, complete checkpoint) instead of the direct least-squares fit
— directly addressing the Sept-30 advisor instruction "take u_RF from
training, not direct fit."

**Experiment.** Same center/sigma grids and closed-form amplitude fitting as
Section 13 (`c = linspace(-0.02,0.02,21)`, `sigma =
logspace(log10(1e-4),log10(3e-1),25)`), evaluated at `t=1` only. Script:
`Burgers/scripts/test_pde_trained_gaussian_correction.py` (`2fa4a91`).

**IMPORTANT metric distinction:** `0.8390488` (Sections 0/15) is the
**canonical multi-time** shock metric. The **t=1-only, fine-grid** baseline
used for this Gaussian-correction comparison (matching Section 13's own
grid convention) is a *different*, smaller number:
```
global = 0.22924198524509445
smooth = 0.186308682793358
shock  = 0.7597470709454747
```

**Result — best shock-optimal single Gaussian:**
```
c = +0.014, sigma = 0.007646112, d_star_shock = -0.6578595
corrected: global = 0.181670, smooth = 0.169661, shock = 0.396438
improvement: global ≈ 20.75%, shock ≈ 47.82%
max |corr_shock| = -0.853067, at the same (c, sigma)
```

**Result — global-optimal:**
```
c = +0.020, sigma = 0.01490050, d_star_global = -0.5812977
global error = 0.166632   (≈27.31% improvement)
```
`c=+0.020` is the **edge of the searched center range**
(`linspace(-0.02,0.02,21)`) — this must **not** be read as a confirmed
global optimum; it may simply reflect the grid boundary, and a wider center
search would be needed to check whether the true optimum lies further out.

---

## 18. Comparison: direct-fit vs. PDE-trained single-Gaussian correction

| | direct-fit RF (Section 13) | PDE-trained RF (Section 17) |
|---|---|---|
| `c*` | +0.010 | +0.014 |
| `sigma*` | 0.005477 | 0.007646 |
| `d*_shock` | -0.2835 | -0.6579 |
| baseline shock | 0.366923 | 0.759747 |
| corrected shock | 0.265913 | 0.396438 |
| shock improvement | ≈27.5% | ≈47.82% |
| correlation magnitude (`corr_shock`) | ≈0.689 | ≈0.853 |

**Interpretation.** The shifted-Gaussian correction remains useful — and is
*more strongly aligned* with the residual (higher correlation, larger
improvement) — when the actual PDE-trained RF is used instead of the
direct-fit approximation. **Therefore, the earlier failure of the local
coefficient to activate during joint PDE training (Sections 1–12) cannot be
explained simply by saying that a Gaussian provides no useful correction to
the PDE-trained RF** — post-hoc, it clearly can.

---

## 19. Figure-based interpretation — large L2 improvement vs. incomplete shock reconstruction

Visual inspection of the Section 17 full-domain and shock-zoom plots adds an
important nuance not visible from the error numbers alone.

The single Gaussian produces a large L2 error reduction, but it does **not**
reconstruct the complete shock profile. Because the optimal Gaussian is
narrow and centered at `x=+0.014` with **negative** amplitude, it creates a
localized negative correction on the **right side** of the shock. In the
full-domain plot, the corrected solution makes a localized downward
excursion near the shock and then rapidly returns to the original RF curve
once the Gaussian decays — this is expected from the Gaussian's
localization and is **not a plotting artifact**.

The shock-zoom figure shows:
- the correction substantially improves the negative/right-hand residual
  lobe;
- the positive/left-hand residual remains largely uncorrected;
- the corrected curve therefore still does not reproduce the entire steep
  shock transition.

The correlation heatmap supports the same reading: a positive-correlation
region for Gaussians centered on the negative-`x` side, and a strong
negative-correlation region for Gaussians centered on the positive-`x` side
(matching Section 17's negative `d*`) — the best single Gaussian simply
selects the stronger of the two (the negative/right-hand lobe).

**Conclusion (phrased carefully):** *The single shifted Gaussian contains a
strongly useful correction direction, but one Gaussian is structurally
insufficient to represent the complete sign-changing shock residual.* It
would be incorrect to say "the Gaussian solves the shock."

---

## 20. Updated central question

Section 14's question — what local spatial structure is required by the
PDE-trained RF's residual — has now been partially answered (Sections
15–19): the residual has a two-lobe, sign-changing structure, and a single
shifted Gaussian can capture roughly one lobe, giving a substantial but
incomplete correction.

**Updated question carried into `TODO.md`:** can a **pair** of shifted
Gaussians (one per lobe) represent substantially more of the fixed
PDE-trained RF residual than one Gaussian can — tested post-hoc, before any
new PDE training is attempted?

(This question is revisited, and this section's framing partially
superseded, by Sections 21–25 below, following the joint-training and
two-stage experiments.)

---

## 21. Joint PDE training with the shifted/wider Gaussian (IC-compatible form)

**Hypothesis.** Sections 17/19 found that a shifted Gaussian (`center=0.014,
sigma=0.007646112`) is a substantially better post-hoc correction direction
for the PDE-trained RF residual than the old centered/narrow one (`center=0,
sigma=0.0025`). Test whether using these post-hoc-diagnosed spatial
parameters *in actual joint PDE training* (not just post-hoc fitting) lets
the IC-compatible local feature activate more strongly and/or improve the
solution.

**Experiment.** `local_time_mode="ic_compatible"` for both cases
(`u_local = G(x) * sum_j d_j [phi_j^t(t) - phi_j^t(0)]`), zero-initialized
coefficients, identical seed/RF realization/training procedure for both.
Script: `Burgers/scripts/test_ic_compatible_shifted_gaussian.py` (`acd2e13`).

**Result — CONTROL (old centered/narrow Gaussian, `center=0, sigma=0.0025`):**
```
global = 0.27157725346704753
smooth = 0.24441935968669457
shock  = 0.8399784150822186
local_coef_norm = 0.006626701293116823
ratio_shock = 0.0009167552106071844
```
(This reproduces Section 10's `ic_compatible` row to full precision — a
useful internal consistency check between the two scripts, not a new result
on its own.)

**Result — TEST (shifted/wider Gaussian, `center=0.014, sigma=0.007646112`):**
```
global = 0.286812083788391
smooth = 0.25496570138109376
shock  = 0.9443527283849813
local_coef_norm = 0.02345200946944816
ratio_full  = 0.002849256798371605
ratio_shock = 0.04213353649724597
```

**Observation.** The shifted Gaussian activated much more strongly during
joint training (`local_coef_norm` ≈3.54x larger, `ratio_shock` ≈46x larger
than CONTROL), but both global and shock solution error got *worse* than
CONTROL and worse than the RF-only baseline. Component plots showed that at
`t=1` the learned local contribution on the positive-`x`/right side was
**positive**, whereas the post-hoc-useful correction there required a
**negative** amplitude (Section 17). Because this was joint training, the RF
component itself was also free to change, so this is not simply "same RF,
wrong Gaussian parameters" — RF and local components could co-adapt.

**Interpretation.** Spatial placement/scale was evidently *part* of the
previous non-activation problem (the shifted Gaussian clearly engaged much
more strongly), but activation alone does not guarantee an accurate
correction — here joint optimization learned a decomposition that made the
solution worse, with the wrong sign relative to the post-hoc-useful
direction.

---

## 22. Simplifying the temporal form: `linear_t` with the shifted Gaussian

**Hypothesis.** Advisor suggestion from `meetings/2026-09-23.md` (already
used in Section 12) — test whether the 10-coefficient `ic_compatible`
temporal expansion itself, rather than the Gaussian's spatial placement, was
responsible for the wrong-sign result in Section 21.

**Experiment.** Same shifted Gaussian (`center=0.014, sigma=0.007646112`)
for both cases: (A) `ic_compatible` (10 trainable `d_j`, reproduces Section
21's TEST result) vs. (B) `linear_t` (`u_local = d*t*G(x)`, exactly one
trainable scalar `d`). Script:
`Burgers/scripts/test_shifted_gaussian_time_modes.py` (`bf88b03`).

**Result — `linear_t`:**
```
d = +0.02335906311371956
global = 0.2863551208182298
smooth = 0.25478499908981045
shock  = 0.9391014811156511
IC error = 0.002432348946430675
ratio_full  = 0.0023750908843608773
ratio_shock = 0.03611833061224904
final PDE loss = 0.2380526586649464
```

**Observation.** The single-scalar `linear_t` result was marginally better
than the 10-coefficient `ic_compatible` shifted case (Section 21's TEST),
but still much worse than the RF-only baseline. Most importantly, the
learned `d` was **positive**, even though the only known post-hoc-useful
amplitude at this spatial location (`t=1` shock-optimal, Section 17) was
**negative** (`d_star_shock ≈ -0.6578595`).

**Interpretation.** Reducing the temporal representation to a single scalar
did not change the sign outcome. Excessive flexibility of the temporal RF
expansion is therefore not sufficient to explain the wrong-sign local
correction — even the simplest possible one-parameter local model learned a
positive `d` during joint training.

---

## 23. Fixed-RF one-dimensional `d` sweep

**Hypothesis.** Determine whether the positive sign learned above (Sections
21–22) is already preferred by the fixed-RF PDE/IC/BC objective itself, or
whether it only emerges from RF/local co-adaptation during joint
optimization.

**Experiment.** Freeze the canonical PDE-trained RF checkpoint completely
(`checkpoint_complete.pt`) and study, without training anything,
```
u(x,t;d) = u_RF(x,t) + d*t*G(x),   center=0.014, sigma=0.007646112
```
over a dense sweep of `d`. Script:
`Burgers/scripts/diagnose_fixed_rf_linear_d_sweep.py` (`e3e2f0f`).

**Derivation.** With `q(x,t) = t*G(x)`, the Burgers residual decomposes
exactly as `r(d) = r0 + d*r1 + d^2*r2`, with `r0` the residual of `u_RF`
alone, `r1 = q_t + u_RF*q_x + q*u_RF_x - nu*q_xx`, `r2 = q*q_x`. This
decomposition was verified numerically against the full nonlinear residual
to machine precision.

**Observation — derivative at `d=0`:**
```
<r0,r1> = 0.047149718399925565
dL_PDE/dd|0 = 2<r0,r1> = +0.09429943679985113   (analytic)
dL_PDE/dd|0 = +0.09429889934210733               (finite-difference)
```
Since this derivative is positive, gradient descent starting from `d=0`
should initially move `d` **negative** for the fixed-RF objective.

**Observation — sweep optima:**
```
weighted training-loss optimum: d* ≈ -0.010
canonical global-error optimum: d* ≈ -1.110
canonical shock-error optimum:  d* ≈ -1.020
t=1 global-error optimum:       d* ≈ -0.725
t=1 shock-error optimum:        d* ≈ -0.660
```
The `t=1` shock-error optimum (`-0.660`) closely agrees with the independent
post-hoc oracle from Section 17 (`-0.6578595`).

**Observation — values at reference points:**
```
at d=0:      PDE loss = 0.2359001, canonical shock = 0.8390488, t=1 shock = 0.7597471
at d≈-0.66:  PDE loss ≈ 3.036739, canonical shock ≈ 0.634366, t=1 shock ≈ 0.396443
```

**Observation — IC/BC exact d-independence.** `IC` and `BC` losses were
exactly independent of `d` in this diagnostic: `q(x,0)=0` identically (so IC
is unaffected), and `G(x=±1)` underflows to exactly `0.0` in float64 at this
width (so BC is unaffected). All `d`-dependence in this diagnostic comes
from the interior Burgers residual.

**Interpretation.** The fixed-RF PDE training objective and solution-error
metrics **agree on the useful direction** (negative `d`) but **strongly
disagree on the useful magnitude** — the PDE objective prefers only a small
negative correction, while solution accuracy continues to improve for much
larger negative amplitudes.

---

## 24. Two-stage / frozen-RF training

**Hypothesis.** Directly test the co-adaptation explanation suggested by
Sections 21–23: if the RF is held exactly fixed (as in Section 23) but the
local scalar `d` is actually *trained* (not just swept) against the same
objective, does it move in the direction the fixed-RF objective predicts?

This design was suggested by the advisor at the 09/30 meeting
(`meetings/2026-09-30.md`: "Suggested a two-stage training approach: train
the RF term first, then freeze (or partially freeze) it and train the local
correction against the resulting residual.").

**Experiment.** Stage 1 (already complete): the canonical RF-only
checkpoint, loaded and frozen exactly. Stage 2: train only `d`
(`u_local=d*t*G(x)`, `center=0.014, sigma=0.007646112`, `d` initialized at
exactly `0`), using the normal training settings unchanged (`LR`,
`WEIGHT_DECAY=1`, `IC_BC_WEIGHT`, `EPOCHS=1000`). Script:
`Burgers/scripts/test_two_stage_linear_local_feature.py` (`873240d`).

**Observation — pre-training verification.** With `d=0`, the loaded model
reproduced the canonical RF-only metrics essentially exactly, and
`local_coefficients` was confirmed to be the only trainable parameter.

**Observation — sign tracking:**
```
grad_d at d=0 (before first step) = +0.09429943679985112
d after first Adam step = -0.0004999999469774197
final d (epoch 999)     = -0.010459279905632843
```
The initial gradient matches Section 23's independently-computed analytic
derivative (`+0.09429943679985113`) to within floating-point precision — a
cross-check between the two separate scripts. The final `d` (`-0.0105`) is
close to the fixed-RF sweep's weighted-training-loss optimum (`d* ≈ -0.010`,
Section 23).

**Observation — final metrics:**
```
global = 0.27117857732889317
smooth = 0.244349311488394
shock  = 0.8348521566708592
IC     = 0.0015740911161049949
ratio_full  = 0.0010636693472845784
ratio_shock = 0.02801549548933219
final PDE loss = 0.2353518646856644
```

**Observation — frozen-RF invariance.** `max|after-before| = 0.0` for `Wx,
Wt, bx, bt, model.0.weight` — the RF truly did not change during Stage 2.

**Comparison:**
```
RF-only:          d=0,                     global=0.27149430095802507, smooth=0.2443827084159696, shock=0.8390487998889169
JOINT linear_t:   d=+0.02335906311371956,  global=0.2863551208182298,  smooth=0.25478499908981045, shock=0.9391014811156511
TWO-STAGE frozen: d=-0.010459279905632843, global=0.27117857732889317, smooth=0.244349311488394,  shock=0.8348521566708592
```

**Interpretation.** This provides strong evidence for RF/local
**co-adaptation** during joint training: with the RF frozen, the local
correction moved negative exactly as predicted by the fixed-RF objective's
own derivative/sweep (Section 23), and modestly improved global/smooth/shock
error relative to the RF-only baseline. With the RF trainable jointly
(Sections 21–22), the local coefficient changed sign to positive and
solution accuracy became worse. This supports the interpretation that the
earlier joint-training failure is not adequately explained by: the Gaussian
spatial basis being useless (Section 17/18 refute this); the shifted
Gaussian failing to activate (Section 21 refutes this); the temporal RF
expansion being too flexible (Section 22 refutes this); or the fixed-RF PDE
objective itself preferring positive `d` (Section 23 refutes this). The
evidence instead points toward interaction/co-adaptation between the global
RF coefficients and the local feature during joint optimization.

---

## 25. Remaining issue and updated central question

Two-stage training (Section 24) resolves the **sign**/co-adaptation puzzle
diagnostically, but the resulting improvement remains small: canonical shock
error only moves `0.8390488 → 0.8348522`, despite the local feature now
moving in the useful direction. Section 23's sweep explains why — the
PDE-training objective's own optimum is only `d≈-0.01`, while solution-error
optima prefer much larger negative amplitudes (`d≈-0.66` to `-1.11`
depending on the metric).

**Updated central question (supersedes Section 20's framing):** the
question is no longer simply "how do we make `u_local` activate?" (Sections
21/24 show it can, with the right sign, once RF is frozen). It is now:

**Why does the PDE-residual training objective favor only a weak local
correction, when a substantially stronger local correction in the same
direction would reduce solution error in the shock region?**

Candidate contributors to investigate (none yet established as *the*
cause):
- the structure/conditioning of the PDE residual near the shock;
- an objective mismatch between residual minimization and solution
  accuracy;
- `WEIGHT_DECAY=1` suppressing the local amplitude (plausible, not yet
  tested in a controlled experiment — do not treat this as established);
- representational limitations of a single one-signed Gaussian (Section
  15's two-lobe residual finding);
- eventually, two shifted Gaussians (`G1+G2`) to represent both residual
  lobes (Section 15/19; still a future direction, not yet tested in joint
  training or even in the post-hoc two-Gaussian diagnostic).

(Sections 26–29 below report the completed two-Gaussian representation
diagnostic, the K-Gaussian capacity study, frozen-RF amplitude-only
training, and the weight-decay sensitivity sweep that together answer
parts of this question and motivate the next one, in Section 29.)

---

## 26. Two-Gaussian and K-Gaussian post-hoc representation capacity

**Hypothesis.** Section 15/19 established that the PDE-trained RF residual
at `t=1` has a two-lobe, sign-changing structure that a single Gaussian can
only partially correct. Test whether a **jointly searched pair** of
shifted Gaussians — one per lobe — can represent substantially more of the
fixed residual, and characterize how representation capacity scales with
more Gaussians.

**Experiment — residual-informed left/right two-Gaussian search.** Fixed,
verified PDE-trained RF checkpoint; `G1` center restricted to the left half
of the existing `C_VALUES` grid, `G2` to the right half, full `SIGMA_VALUES`
for each side; amplitudes solved by closed-form least squares (SVD-based
`lstsq`, never an optimizer) for every candidate pair. Script:
`Burgers/scripts/test_pde_trained_two_gaussian_correction.py` (`a6f6afc`).

**Result — RF-only vs. single-Gaussian vs. jointly searched two-Gaussian,
t=1 shock error:**
```
K=0 (RF-only):           0.7597470709454747
K=1 (best single Gaussian): ~0.396438   (Section 17)
K=2 (best jointly searched pair):
    c1 = -0.010,  sigma1 = 0.010673841067708996,  d1_oracle = +0.6296921638041192
    c2 = +0.010,  sigma2 ~ 0.01490050,             d2_oracle = -0.6813727781614515
    shock error = 0.03589738957120498
```
RF→2G shock improvement ≈ **95.275%**. Condition number of the two-column
fit ≈ **2.023** (well-conditioned, not a near-collinear/canceling pair).
Shock-residual R² ≈ **0.997768**.

**Interpretation.** The jointly searched two-Gaussian pair captures nearly
all of the `t=1` shock residual — strong confirmation that representation
capacity is not the bottleneck: two well-placed, oppositely-signed
Gaussians are sufficient to explain the two-lobe residual almost entirely.

**Experiment — greedy K-Gaussian representation-capacity study.** Greedy
forward basis selection (all amplitudes jointly refit by closed-form least
squares at every step, never Adam) from the same Gaussian candidate
library, `K=1..8`, with an independent, denser shock grid used only to
validate each step's fixed correction (never for selection/fitting), to
distinguish genuine representation gains from discrete-grid overfitting.
Script: `Burgers/scripts/test_pde_trained_k_gaussian_correction.py`
(`57024c1`).

**Result (qualitative).** The Gaussian dictionary has **very high**
representation capacity: the independent dense-grid validation error
tracked the fitting-grid error closely through `K=8`, with `K=8` dense
shock error ≈ **0.0033113** and dense R² ≈ **0.999981**. This confirms the
K=2 representation gain (above) is genuine, not an artifact of fitting the
sparse canonical shock grid.

**Important caveat — do not over-read this result.** `K=8` is **not**
claimed to be the optimal or necessary number of Gaussians — this was a
greedy, non-exhaustive search, and most of the achievable representation
gain was already captured at `K=2`. The clean, well-conditioned,
jointly-searched **K=2** pair above remains the preferred basis for
controlled trainability experiments; `K=8` is reported only to
characterize how quickly representation capacity saturates.

---

## 27. Frozen-RF two-Gaussian PDE training (amplitude-only)

**Hypothesis.** Section 26 shows a fixed, well-conditioned two-Gaussian
basis can represent almost all of the `t=1` shock residual. Does the
PDE-training objective itself, with the RF **and** the Gaussian geometry
completely frozen, learn amplitudes comparable to the useful post-hoc
values when only `d1,d2` are trained from zero? This directly extends
Section 24's frozen-RF single-Gaussian result to the representationally
sufficient K=2 basis, and tests whether the small-magnitude problem found
there (Section 25) persists when the representation bottleneck is removed.

**Experiment.** Model `u = u_RF + t*[d1*G1(x) + d2*G2(x)]`, with
`(c1,sigma1,c2,sigma2)` fixed to the Section 26 pair-search geometry; RF
and Gaussian geometry completely frozen; only `d1,d2` trained from exactly
zero, same training settings as Section 24 (`LR`, `WEIGHT_DECAY=1`,
`IC_BC_WEIGHT`, `EPOCHS=1000`). Script:
`Burgers/scripts/test_two_stage_two_gaussian_local_feature.py` (`65534b9`).

**Result:**
```
final d1 = +0.035453350169778575
final d2 = -0.03832831801950355

d1/d1_oracle = 0.05630267011041793
d2/d2_oracle = 0.05625161328417738

canonical: global=0.26946486173938333, smooth=0.243773446446857, shock=0.8169198499822581
t=1:       global=0.2221142171017243, smooth=0.183106156638777, shock=0.7170982076598021
```
Both initial PDE-descent directions (`-grad_d1`, `-grad_d2` at `d1=d2=0`)
agreed with the useful oracle signs. All frozen RF tensors
(`Wx,Wt,bx,bt,model.0.weight`) changed by exactly `0.0`.

**Interpretation.**
```
representation:          strong   (Section 26)
initial direction/sign:  correct  (both d1, d2)
learned amplitude:       much too small (~5.6% of oracle for both)
```
Because the RF is completely frozen throughout this experiment, **this
amplitude limitation cannot be attributed to RF/local co-adaptation** —
Section 24's co-adaptation explanation for the *sign* problem does not
apply here, since there is no RF left to co-adapt with. Something about the
frozen-RF PDE-training objective itself strongly limits the learned
amplitude, even when the representation is known (post-hoc) to be
excellent and the initial direction is already correct.

---

## 28. Weight-decay sensitivity of the frozen-RF two-Gaussian amplitude

**Hypothesis.** Section 27 left both amplitudes at only ~5.6% of their
useful post-hoc values despite correct initial direction. `WEIGHT_DECAY=1`
(vanilla, non-decoupled Adam) adds `weight_decay*param` directly to the
gradient, pulling parameters toward zero in proportion to their own
magnitude — a plausible, previously untested, contributor (`IDEAS.md` item
17). Test this directly with a controlled sweep rather than assuming it.

**Experiment.** `WEIGHT_DECAY in {1.0, 0.1, 0.01, 0.0}`, all other
conditions identical to Section 27 (same frozen checkpoint, same Gaussian
geometry, same `d1=d2=0` init, same seed/interior points reset identically
before each case, same LR/epochs/`IC_BC_WEIGHT`/optimizer). Script:
`Burgers/scripts/sweep_two_stage_two_gaussian_weight_decay.py` (`9d55e6e`).
The `weight_decay=1.0` case reproduced Section 27's result within tolerance
(control gate passed) before the other cases were interpreted.

**Result:**

| weight_decay | d1 | d2 | d1/oracle | d2/oracle | canonical shock | t=1 shock | final PDE loss |
|---|---|---|---|---|---|---|---|
| 1.0 | +0.03545335 | -0.03832832 | 0.05630 | 0.05625 | 0.81691985 | 0.71709821 | — |
| 0.1 | +0.02536062 | -0.20187330 | 0.04027 | 0.29627 | 0.77304914 | 0.61442670 | 0.22376522 |
| 0.01 | +0.02240566 | -0.22241667 | 0.03558 | 0.32642 | 0.76957076 | 0.60748817 | 0.22353109 |
| 0.0 | +0.02203028 | -0.22464604 | 0.03499 | 0.32970 | 0.76923542 | 0.60685667 | 0.22352864 |

**Interpretation.**
- `weight_decay=1` materially suppresses the useful local correction,
  **especially `d2`**: reducing it substantially improves both PDE loss
  and solution error.
- The effect **essentially saturates by `weight_decay≈0.01`** — going all
  the way to `0.0` changes very little further.
- **Even with `weight_decay=0`, both amplitudes remain far from the
  post-hoc solution-error oracle** — `d2` reaches only ~33% of its oracle
  value, `d1` only ~3.5%.
- `d1` and `d2` behave **asymmetrically**: `d2` (the negative/right-side
  Gaussian) grows substantially as weight decay is relaxed, while `d1`
  (the positive/left-side Gaussian) barely grows at all.
- **Therefore weight decay contributes to amplitude suppression but does
  NOT fully explain it.** The remaining, larger mismatch appears to arise
  from the PDE-training objective itself preferring a substantially
  different/weaker correction than the `t=1` solution-error oracle does —
  consistent with, and now sharpened by, the asymmetry between `d1` and
  `d2`.

This finding should be read as evidence **against** weight decay being
*the* cause, not evidence *for* it — it is one real, now-quantified
contributor among others, not a complete explanation.

---

## 29. Updated central question (supersedes Section 25's framing)

Sections 26–28 progressively isolated and ruled out several candidate
explanations for the weak local-amplitude problem:
- **representation capacity** is not the bottleneck (Section 26: a
  well-conditioned K=2 basis explains ~99.8% of the shock residual's
  variance, confirmed genuine on an independent dense grid, and capacity
  saturates further by K=8);
- **sign/direction** is not the bottleneck (Section 27: both `d1,d2`
  start moving in the correct direction from `d1=d2=0`);
- **RF/local co-adaptation** cannot explain the remaining amplitude gap
  (Section 27: the RF is completely frozen throughout, yet the amplitude
  problem persists);
- **`WEIGHT_DECAY`** is a real but only **partial** contributor (Section
  28: removing it entirely still leaves both amplitudes far from useful,
  and leaves `d1` and `d2` asymmetrically affected).

**Updated central question:** **Why does the unregularized PDE-training
objective strongly use the negative/right Gaussian (`d2`) but barely use
the positive/left Gaussian (`d1`), even though the `t=1` solution residual
requires both** to achieve the representational benefit established in
Section 26?

This reframes the question from "how large is the amplitude" to "why is
the objective's preference asymmetric between the two Gaussians," and is
carried forward into `TODO.md`.

---

## 30. October 7, 2026 meeting checkpoint and advisor directions

**Status at the meeting** (established by Sections 15–29, summarized here
for continuity — no new experiment in this section).

Representation: the actual PDE-trained RF residual at `t=1` is strongly
localized and sign-changing near the shock (Section 15). A single shifted
Gaussian substantially improves the shock residual (Sections 17–19); a
jointly searched two-Gaussian opposite-sign pair represents it extremely
well (Section 26, ~99.8% of the shock residual's variance, ≈95.3%
shock-error improvement); greedy K-Gaussian enrichment through `K=8`
demonstrates substantial additional representation capacity (Section 26).
**`K=8` is a representation-capacity diagnostic, not a globally optimized
8-Gaussian solution** — this distinction is preserved from Section 26 and
must continue to be preserved in any future K-Gaussian work.

Training: with the canonical RF frozen, PDE training moves the
two-Gaussian amplitudes initially in the same signs as the `t=1` oracle
amplitudes (Section 27). `weight_decay=1` strongly suppresses the local
correction; reducing/removing weight decay improves the learned
correction, especially the second Gaussian, but does **not** recover the
`t=1` oracle amplitudes (Section 28). Representation is therefore not the
main limitation on the learned correction's size. Regularization
contributes to amplitude suppression, but a large discrepancy remains
between the multi-time PDE-training objective and the `t=1` solution-error
oracle (Section 29).

**Important framing, reaffirmed at this meeting:** the oracle and the
PDE-training objective are genuinely different objectives (the oracle
minimizes `t=1` solution error only; PDE training minimizes a multi-time
PDE/IC/BC residual objective). This log does **not** claim that PDE
training "should" reproduce the oracle amplitudes — the open question
(Section 29) is why the two objectives' preferences diverge so
asymmetrically between the two Gaussians, not that they are expected to
agree.

**Advisor directions from the 2026-10-07 meeting** (see
`docs/meetings/2026-10-07.md` for the full discussion; none of the below
have been run yet — recorded here as planned future work, tracked in
`TODO.md`):

1. **RF basis-size study.** Test the number of random spatial features
   `N = 500 -> 1000 -> 2000` (handwritten values), comparing global
   relative L2 and shock-region relative L2, with an error-vs-`N` plot.
   Purpose: determine whether increasing RF capacity continues to reduce
   error or plateaus.
2. **Extend Gaussian enrichment to `K>=8`** for both the post-hoc/oracle
   representation diagnostics and the PDE-training diagnostics, and
   jointly with the RF basis-size study (item 1). **The exact N-by-K
   combinations to run are explicitly TBD** — not yet decided, and should
   not be treated as a full experimental matrix to run blindly.
3. **Trainable Gaussian geometry.** Move beyond fixed `(center, width)`
   geometry to training `{c_k, sigma_k, d_k}` jointly through the PDE
   objective, for one or more Gaussians. Recorded as a future experiment
   only.
4. **Reporting convention.** Consistently report both global relative L2
   error and shock-region relative L2 error in all experiment comparisons
   and plots going forward.
5. **Multiscale RF basis — new hypothesis, not yet tested.**
   ```
   u_RF = sum_{k=1}^{N1} a_k cos(w_k^(1) x + b_k^(1))
        + sum_{k=1}^{N2} beta_k cos(w_k^(2) x + b_k^(2))

   w_k^(1) ~ N(0, sigma_1^2),  w_k^(2) ~ N(0, sigma_2^2),  N = N1 + N2
   ```
   Motivation: one RF population may represent smoother spatial structure
   while a second, different frequency scale may provide additional
   resolution around the sharp shock. **This is a hypothesis, not an
   established conclusion.** The handwritten notes appear to contain
   candidate `N1,N2` values, but their precise interpretation is not
   settled — marked TBD/confirm with advisor, not invented here.

These five directions are reflected in `TODO.md`'s reordered "Current
priority" (items A–E) and in new/updated `IDEAS.md` entries for
multiscale RF, trainable Gaussian geometry, and the N-vs-K interaction
question.

---

## 31. RF basis-size sweep

**Hypothesis.** Per the 2026-10-07 advisor direction (Section 30, item 1):
test whether increasing the number of spatial random features (`Nx`)
continues to reduce error, or whether it plateaus.

**Experiment.** RF-only (`use_local=False`) canonical training at
`Nx = 500, 1000, 2000`, with `Nt=10`, seed=42, and all other settings
(collocation counts, optimizer, `LR`, `WEIGHT_DECAY`, `IC_BC_WEIGHT`,
1000 epochs) held fixed at their `config.py` values — isolating feature
*count* only. Script: `Burgers/scripts/sweep_rf_basis_size.py` (`0ca03dc`).
Numerical results below were supplied from the actual server run; they
have not been independently re-run locally (no GPU/CUDA available on this
machine, and the real checkpoints/outputs are gitignored and not present
here).

**Result:**
```
Nx=500:  global=0.27149430095802507, smooth=0.2443827084159696, shock=0.8390487998889169, PDE loss=0.23590008351925892
Nx=1000: global=0.2839157305655459,  smooth=0.25354926454918303, shock=0.9181808292859333, PDE loss=0.22939219902001096
Nx=2000: global=0.3160344409079877,  smooth=0.27944891957843243, shock=1.0849605011988503, PDE loss=0.22558102432851115
```

**Observation.** Under this fixed 1000-epoch canonical training setup,
increasing `Nx` did **not** improve solution accuracy — global, smooth,
and shock error all *increased* with `Nx` — even though the sampled PDE
loss *decreased*.

**Important limitations (do not over-read this result):**
- only seed 42 was used;
- `Nt` remained fixed at 10 throughout (only `Nx` varied);
- optimization/training-budget effects remain possible (see Section 32);
- the interior collocation points were controlled (bit-identical across
  `Nx`, since `sample_training_points` runs before model construction —
  see Section 23's RNG analysis), but `BurgersRF.__init__` draws an
  `Nx`-dependent number of random values for `Wx`/`bx`, so the per-epoch
  BC/IC resampling trajectory was **not** identical across the three `Nx`
  cases — this is an inherent, already-documented consequence of `Nx`
  itself changing the model's RNG footprint (see `sweep_rf_basis_size.py`'s
  own summary note), not a new confound.

**Interpretation.** Do **not** conclude that larger RF bases are
inherently worse — the result is specific to this single-seed,
fixed-`Nt`, fixed-training-budget setup. This question is reopened and
broadened in Section 35 following the 2026-10-14 advisor meeting.

---

## 32. Nx=1000 longer-training diagnostic

**Hypothesis.** Does Section 31's Nx=1000 degradation reflect insufficient
training duration rather than a genuine capacity/optimization issue?

**Experiment.** One continuous training trajectory for the Nx=1000 model,
checkpointed at 1000, 2000, 5000, and 10000 epochs (no restart between
checkpoints — same model/optimizer objects throughout, with RNG-safe,
deterministic intermediate diagnostics). Script:
`Burgers/scripts/test_rf_nx2000_long_training.py --nx 1000` (`5f576de`,
`c6b259f`, `0e4cd42`, `0b740f1`). As in Section 31, these numbers were
supplied from the actual server run and have not been independently
re-run locally.

**Result:**
```
epoch 1000:  global=0.2839157305655459,  smooth=0.25354926454918303, shock=0.9181808292859333, PDE=0.22939219902001096
epoch 2000:  global=0.2794635841570797,  smooth=0.2496847216718783,  shock=0.8992185808186575, PDE=0.21611259274922276
epoch 5000:  global=0.2800984603045993,  smooth=0.25071616222435966, shock=0.8945062214766912, PDE=0.2141470907329272
epoch 10000: global=0.27907511488597353, smooth=0.2494314600182936,  shock=0.8963994256493268, PDE=0.20236475819643004
```

**Interpretation.** Longer training gives only modest improvement, and
Nx=1000 at 10,000 epochs still does **not** outperform the canonical
Nx=500 result at 1,000 epochs (Section 31: shock=0.8390488). Therefore the
simple explanation "larger `Nx` only needs more epochs" is **weakened** as
the primary explanation for Section 31's result, but **not universally
ruled out** — other training configurations (learning rate, `Nt`, seed)
remain untested.

---

## 33. Frozen-RF K=8 Gaussian PDE training

**Hypothesis.** Section 26 established that a jointly-fit K=8 Gaussian
basis has very strong post-hoc representational capacity for the `t=1`
shock residual. With the canonical Nx=500 RF completely frozen (same
protocol as Section 27's K=2 case, now generalized to K=8), can PDE
training learn a correction that approaches this capacity?

**Experiment.** `u = u_RF + t * sum_{k=1}^{8} d_k G_k(x)`, with the exact
K=8 Gaussian geometry recovered from the Section 26 greedy capacity study,
RF completely frozen (zero trainable model parameters, verified), only
the 8 scalar amplitudes `d_k` trainable (initialized at exactly zero),
`WEIGHT_DECAY=0` (experiment-local, not `config.py`'s canonical value — a
deliberate controlled change, following Section 28's weight-decay
finding). Script:
`Burgers/scripts/test_two_stage_k8_gaussian_local_feature.py` (`0d43f54`).
As above, these result numbers were supplied from the actual server run
and have not been independently re-run locally.

**Result (at `t=1`):**
```
RF-only:          global=0.229241985245, smooth=0.186308682793, shock=0.759747070945
PDE-trained K=8:  global=0.226182356002, smooth=0.185368843075, shock=0.738366522323
Oracle/post-hoc:  global=0.155196900281, smooth=0.157785446046, shock=0.003397832223
```
The RF-only values match Section 17's already-documented `t=1` fine-grid
baseline to the precision given.

**IMPORTANT — two similar but distinct numbers, do not merge:** the
oracle/post-hoc K=8 shock error here (`0.003397832223`) comes from a
**fresh** joint-least-squares refit on the fitting grid, computed inside
`test_two_stage_k8_gaussian_local_feature.py` itself. This is a **different
diagnostic** from Section 26's own K=8 dense-grid *validation* shock error
(`≈0.0033113`), which came from the original greedy capacity script's
independent dense grid. The two values are close (as expected, since both
describe the same underlying K=8 basis' representational capacity) but are
not the same computation and should not be conflated.

**Observation.** Comparing against Section 27's earlier K=2 frozen-RF
result: K=2's PDE-trained `t=1` shock error was `0.7170982076598021` —
*better* (lower) than this K=8 result's PDE-trained shock error
(`0.738366522323`), despite K=8's dramatically stronger post-hoc
representational capacity. This is worth stating explicitly: a richer
post-hoc basis did not translate into a better PDE-trained outcome here.

**Interpretation.** The K=8 Gaussian basis has strong representational
capacity (confirmed independently in Section 26 and again here via the
fresh oracle fit), but PDE training activates it only weakly — the
PDE-trained correction captures only a small fraction of the available
post-hoc improvement. Because the RF was completely frozen throughout,
**RF/local co-adaptation is excluded as the primary explanation** in this
specific experiment (consistent with Section 27's K=2 finding). Do **not**
claim that the PDE objective itself is definitively wrong — finite
collocation, the `t*d_k` temporal restriction, loss weighting, and
optimization dynamics all remain possible contributors (see Section 29's
still-open amplitude-asymmetry question and `IDEAS.md` item 21's four-axis
framework).

---

## 34. Two-population multiscale RF experiments

Following the 2026-10-07 meeting's multiscale hypothesis (Section 30, item
5; `IDEAS.md` item 19), both experiments below modify **only** the
frequency content of the canonical Nx=500 spatial RF basis, via
`Burgers/scripts/test_rf_multiscale_two_population.py`. In both,
`Nx_total=500`, `Nt=10`; population 1 (the first 250 spatial frequencies)
is **unchanged** from the canonical seed-42 realization; population 2 (the
remaining 250) is **rescaled** from the same underlying random draw, not
redrawn. As in prior sections, these result numbers were supplied from the
actual server runs and have not been independently re-run locally.

**Sigma convention (stated explicitly to avoid ambiguity).** The original
implementation constructs
```
Wx = Z / sigma_x,   Z ~ N(0,1)
```
therefore
```
std(Wx) = 1 / sigma_x.
```
Canonical `sigma_x = 0.04` therefore corresponds to spatial frequency
`std = 25`. This convention comes from the original RF implementation
(`burgers_rf/model.py`), not from the multiscale modification.

**Experimental control (both experiments below).** The canonical
seed-42 construction trajectory (`set_seed(42) → sample_training_points →
BurgersRF(...)`) is reproduced exactly; `Wt`, `bx`, `bt`, and the initial
`model.0.weight` are bit-identical to the canonical single-population
model (verified via explicit control assertions in the script, including a
check that the global RNG state immediately before training is unchanged
from the canonical trajectory); training points, optimizer, LR, training
budget (1000 epochs), and canonical weight decay are all unchanged. The
**only** varied factor is the frequency scale applied to the second
population of 250 spatial frequencies.

### 34.1 Aggressive high-frequency population

**Setup.** Population 1: 250 features at frequency std=25 (canonical).
Population 2: 250 features at frequency std=312.5 (`sigma_high=0.0032`).
Commit: `1fc94d5`.

**Result:**
```
global=0.340315446922357, smooth=0.3083824300111887, shock=1.0643488610679261, PDE=0.3517382024333927
```

**Interpretation.** This aggressive high-frequency population
substantially **worsened** both solution accuracy and PDE residual
relative to the canonical single-population baseline (Section 31's
Nx=500 row). Do **not** conclude that multiscale RF fails in general —
this is one specific, aggressive frequency choice.

### 34.2 Moderate high-frequency population

**Setup.** Population 1: 250 features at frequency std=25 (canonical,
unchanged). Population 2: 250 features at frequency std=75
(`sigma_high=1/75`). Commit: `981a296`.

**Result (canonical multi-time metrics):**
```
global=0.24041498078112022, smooth=0.21231193413050437, shock=0.7900926655681234, PDE=0.1929321794963624
```

**Result (`t=1`-only):**
```
global=0.21680942270573933, smooth=0.177349557330002, shock=0.7071530431716319
```

**Comparison to canonical Nx=500 (Section 31):**
```
canonical: global=0.27149430095802507, smooth=0.2443827084159696, shock=0.8390487998889169, PDE=0.23590008351925892
```

**Interpretation.** This is the **first controlled positive multiscale
result**: with the same total `Nx=500`, redistributing spatial frequency
content (rather than changing the total feature count) improved global,
smooth, shock, *and* PDE-loss metrics simultaneously, relative to the
single-population canonical baseline. This provides evidence that
frequency *allocation* matters, not just total count. Do **not** describe
`sigma_high` corresponding to std=75 as optimal — it is one successful
moderate-scale diagnostic, not a tuned result (see Section 35's advisor
direction to study this systematically).

---

## 35. October 14, 2026 meeting checkpoint and advisor directions

**Status at the meeting.** Sections 31–34 above were presented: the RF
basis-size sweep showed solution error *increasing* with `Nx` under the
fixed canonical training budget (Section 31); longer training only
partially explained this (Section 32); a K=8 frozen-RF local correction
had strong post-hoc capacity but weak PDE-trained activation (Section 33);
and a moderate multiscale frequency split gave the first positive
multiscale result while an aggressive split made things worse (Section
34).

**Advisor's central conclusion.** The current `u_RF` itself is not
sufficiently accurate. The advisor expects a well-constructed RF
approximation to reproduce the *smooth* portions of the Burgers solution
well, with the principal approximation difficulty concentrated near the
shock — and reports that his own RF solution achieves this better than the
current implementation. **Therefore the immediate research priority shifts
from increasingly complex local Gaussian corrections to first understanding
and improving the baseline RF approximation itself.**

**Updated central question (supersedes the amplitude-asymmetry framing of
Section 29 as the *immediate* priority, without invalidating it as a
question):**

**Why is `u_RF` not producing a sufficiently accurate approximation of the
smooth part of the Burgers solution, and how can the baseline RF itself be
improved?**

**Advisor directions:**
1. **RF size/convergence study is not yet sufficient.** Section 31 used
   only seed 42 with `Nt` fixed at 10. Directions: run multiple random
   seeds (e.g. 20, 21, ...); investigate `Nx` and `Nt` jointly, possibly
   with a controlled `Nx`/`Nt` ratio; investigate a substantially broader
   RF-size range. **The exact `Nx`/`Nt` grid is explicitly TBD** — not
   decided, do not invent specific values.
2. **Optimization diagnostic:** try a smaller learning rate, as a
   controlled diagnostic of `u_RF` quality — isolated from other
   architecture changes, not combined with them in the same run.
3. **Multiscale RF, more systematically.** Continue the two-population
   investigation (Section 34), but σ1/σ2 should be selected in a
   principled way — ultimately informed by PDE structure rather than
   chosen arbitrarily — and the `N1`/`N2` allocation, and total `N`,
   should themselves be investigated. Explicitly: do **not** interpret the
   successful std=75 result (Section 34.2) as having found the correct or
   optimal scales.
4. **Alternative joint space-time RF formulation.** Proposed as an idea to
   test, not an established improvement: replace the current separable
   product features
   ```
   phi_ij(x,t) = cos(wx_i*x + bx_i) * cos(wt_j*t + bt_j)
   u_RF(x,t) = sum_i sum_j c_ij phi_ij(x,t)
   ```
   with a single joint cosine over concatenated `(x,t)`:
   ```
   phi_k(x,t) = cos(wx_k*x + wt_k*t + b_k)
   u_RF(x,t) = sum_k c_k phi_k(x,t)
   ```
   A fair comparison budget between the separable `Nx*Nt`-coefficient
   representation and a joint `N`-feature representation is **not** the
   same by default and is explicitly **TBD** — do not invent the
   comparison protocol yet.
5. **Local Gaussian correction / boundary conditions clarification.** For
   `u_local(x,t) = t * sum_k d_k G_k(x)`, `u_local(x,0)=0` holds exactly —
   but the Gaussian local correction does **not** analytically satisfy the
   Burgers boundary condition at `x=±1`. For narrow Gaussians centered near
   `x=0` the contribution at `x=±1` is numerically extremely small (as
   already measured for the canonical geometry — `G(±1)` underflows to
   exactly `0.0` in float64 at the widths used in this study, Section 23),
   but it is not identically zero by construction. Relevant to future
   local-feature formulation, not an immediate action item.
6. **Trainable local Gaussian parameters — postponed.** Do **not** proceed
   yet with making local Gaussian center `c`, width `sigma`, and amplitude
   `d` jointly trainable (`TODO.md`'s prior item D). The immediate priority
   is to make the RF baseline work well first; this item is deprioritized,
   not abandoned.
7. **Code sharing:** share the GitHub `burgers-local-features` branch with
   the advisor so implementations can be directly compared.

**New research structure going forward (not a finalized experimental
order — a structured summary of open directions):**

- **Phase A — understand and improve `u_RF`:** compare the current
  implementation/setup against the advisor's better RF result; multi-seed
  testing; systematic `Nx`/`Nt` study; learning-rate diagnostic; systematic
  multiscale frequency-scale/allocation study; joint vs. separable
  space-time RF comparison.
- **Phase B — localized shock enrichment:** resumed only after Phase A
  produces a satisfactory smooth-region `u_RF` approximation; then
  reconsider local basis design and PDE training of `u_local`, including
  the still-open `d1`/`d2` amplitude-asymmetry question (Section 29) and
  the deprioritized trainable-geometry item.

---

## 36. Random-seed sensitivity (Phase A item B)

**Hypothesis.** Section 31's RF basis-size sweep used only seed=42. Is the
canonical Nx=500, Nt=10 result a property of the RF method, or a
particularly good (or bad) random realization?

**Experiment.** Canonical RF-only training (Nx=500, Nt=10, sigma_x=0.04,
sigma_t=1/3, M_TRAIN=3600, M_BD=500, M_INT=500, 1000 epochs, LR=5e-4,
weight_decay=1, IC_BC_WEIGHT=1000, float64) at five new seeds
(20, 21, 22, 23, 24), each trained independently from `set_seed(seed)`
onward — BurgersRF construction and `physics.train()`/loss_fn reused
unmodified. The canonical seed=42 result was **reused, not retrained**,
reconstructed directly from the existing complete `model_state_dict`
checkpoint and gated against the canonical reference at `rtol=1e-4`
before inclusion. Script: `Burgers/scripts/sweep_rf_random_seeds.py`
(`ac6f577`). Output: `Burgers/outputs/rf_random_seed_sweep/summary.csv`
(verified locally — all six rows present).

**Result (canonical multi-time metrics):**
```
seed 20: global=0.28212115, smooth=0.25733275, shock=0.82372690
seed 21: global=0.35065153, smooth=0.31609976, shock=1.13111724
seed 22: global=0.31030160, smooth=0.27981009, shock=0.97772456
seed 23: global=0.30263023, smooth=0.28010646, shock=0.82286865
seed 24: global=0.26166848, smooth=0.23196671, shock=0.85814503
seed 42: global=0.27149430, smooth=0.24438271, shock=0.83904880  (canonical reference, reused)
```
Across all six seeds: global mean=0.296478, sample stdev=0.032267,
min=0.261668 (seed 24), max=0.350652 (seed 21); smooth mean=0.268283,
stdev=0.030230; shock mean=0.908772, stdev=0.123473, min=0.822869
(seed 23), max=1.131117 (seed 21). Seed 42 ranks 2nd-best of six by
global error and 3rd-best of six by shock error.

**Interpretation.** Seed 42 is **not unusually poor** — it sits on the
better side of the six-seed distribution, not the worse. However, there is
observed seed-to-seed variability across the six tested seeds (global
relative L2 spans ~0.26–0.35, roughly a 34% relative range). This observed
range should not be generalized into a universal uncertainty bound — it
describes variability among these six seeds under this specific
configuration, not a property proven to hold elsewhere — but it does
motivate caution (and ideally reporting this kind of range, or testing
additional seeds) when comparing single-seed conclusions elsewhere in this
log (including Sections 31–34). Collocation points, RF weights/phases, and
coefficient initialization are seed-determined and therefore differ across
seeds by design; this is not a confound, it is the variable under study.

---

## 37. Fixed Nx/Nt=50 ratio sweep (Phase A item C)

**Hypothesis.** Section 31 varied `Nx` alone with `Nt` fixed at 10. Does
scaling `Nt` together with `Nx`, preserving the canonical ratio
`Nx/Nt=50`, avoid the degradation seen when `Nx` alone was increased?

**Experiment.** RF-only training at seed=42, 1000 epochs, with all other
settings canonical, across eleven `(Nx, Nt)` pairs preserving `Nx/Nt=50`:
`(50,1), (100,2), (250,5), (500,10), (750,15), (1000,20), (1250,25),
(1500,30), (2000,40), (2500,50), (3000,60)`. The `(500,10)` case is the
canonical configuration and was gated against the known canonical
reference (`rtol=1e-4`) before trusting the rest of the sweep. For every
config, `set_seed(42)` is called fresh immediately before
`sample_training_points`/model construction, so collocation points are
bit-identical across all configs; `BurgersRF.__init__` draws an
`(Nx,Nt)`-dependent number of random values, so per-epoch BC/IC sampling
necessarily differs across configs from the first training epoch onward
(same documented behavior as Section 31, no new sampling methodology).
Script: `Burgers/scripts/sweep_rf_nx_nt_fixed_ratio.py` (`bf4f24b`,
`36019b4`). Output:
`Burgers/outputs/rf_nx_nt_fixed_ratio_sweep/summary.csv` (verified
locally). **`(3500,70)` and `(4000,80)` were not run** — the sweep's
configuration list includes them, but training stopped after `(3000,60)`;
no results exist for the two largest configurations and none are claimed
here.

**Result (canonical multi-time global / smooth / shock relative L2,
coefficient count in parentheses):**
```
( 50, 1,     50): 0.90501 / 0.90225 / 1.03501
(100, 2,    200): 0.50583 / 0.49452 / 0.87537
(250, 5,   1250): 0.40976 / 0.39439 / 0.84989
(500,10,   5000): 0.27149 / 0.24438 / 0.83905   <- canonical
(750,15,  11250): 0.27107 / 0.24416 / 0.83463
(1000,20, 20000): 0.32090 / 0.28550 / 1.07842
(1250,25, 31250): 0.29630 / 0.26300 / 0.99121
(1500,30, 45000): 0.28769 / 0.25719 / 0.92850
(2000,40, 80000): 0.32777 / 0.29053 / 1.12120
(2500,50,125000): 0.27658 / 0.24515 / 0.91793
(3000,60,180000): 0.37456 / 0.34421 / 1.10767
```
PDE loss (sampled, not shown above) does not track solution error
monotonically either — e.g. `(1000,20)` has a *lower* PDE loss
(0.22346) than canonical `(0.23590)` while its global/smooth/shock errors
are all worse.

**Interpretation.** Error vs. coefficient count is **non-monotonic** —
there is no systematic improvement from increasing capacity beyond the
canonical scale, consistent with Section 31's single-axis finding. The
one partial exception is `(750,15)`, which is essentially tied with (very
slightly better than) canonical on global/smooth/shock simultaneously —
worth noting as a possible local optimum near the canonical scale, not
evidence that larger capacity helps generally. Degradation reappears
immediately at `(1000,20)`, partially recovers by `(1500,30)`/`(2500,50)`,
and is worst at the largest completed configuration `(3000,60)`. Do
**not** conclude this rules out larger RF bases working well under a
different seed, training budget, or learning rate — see Sections 36 and
38.

---

## 38. Learning-rate sensitivity (Phase A item D)

**Hypothesis.** Is the canonical Adam `LR=5e-4` too aggressive, and does a
smaller learning rate produce a more accurate RF-only solution under the
same 1000-epoch budget?

**Experiment.** Canonical RF-only configuration (Nx=500, Nt=10, seed=42,
all settings otherwise unchanged), comparing `LR in {2e-4, 1e-4, 5e-5}`
(freshly trained, continuously from epoch 0 to 1000 via three
`train()` segments — 0→250, 250→500, 500→1000 — with RNG-free fixed-grid
intermediate diagnostics, the same proven-safe segmentation pattern as
Section 32) against the canonical `LR=5e-4` reference (reused/reconstructed
from the existing checkpoint, **not retrained**, gated at `rtol=1e-4`;
only its real epoch=1000 result exists — no 250/500-epoch values are
claimed for it). For every new LR, `set_seed(42)` → `sample_training_points`
→ `BurgersRF(...)` is identical and LR-independent, so collocation points,
RF weights/phases, and coefficient initialization are bit-identical across
all three new LR runs, and the per-epoch BC/IC draws are therefore also
bit-identical across them (LR affects only `optimizer.step()`, never a
sampling call). Script: `Burgers/scripts/sweep_rf_learning_rate.py`
(`479b761`). Output:
`Burgers/outputs/rf_learning_rate_sweep/summary.csv` (verified locally).

**Result (global relative L2, by epoch):**
```
LR=5e-4 (canonical): epoch 1000 = 0.27149  (reused, no 250/500 values exist)
LR=2e-4:  epoch 250 = 0.45493,  epoch 500 = 0.37387,  epoch 1000 = 0.30987
LR=1e-4:  epoch 250 = 0.52479,  epoch 500 = 0.44756,  epoch 1000 = 0.36598
LR=5e-5:  epoch 250 = 0.57251,  epoch 500 = 0.52036,  epoch 1000 = 0.44056
```
(Smooth/shock/IC/PDE-loss/BC-loss follow the same ordering; full values in
`summary.csv`.)

**Interpretation.** At the fixed 1000-epoch budget, **all three smaller
learning rates are worse than canonical**, but all three continued
improving substantially through epoch 1000 and had not demonstrated a
plateau. (The 250→500 and 500→1000 checkpoint intervals are of unequal
length — 250 vs. 500 epochs — so their absolute error reductions are not
directly comparable, and no claim is made here about the per-epoch rate of
improvement speeding up or slowing down.) **Do not conclude that smaller
learning rates are intrinsically inferior** — this
experiment only establishes that they are worse *under this specific
1000-epoch budget*, and motivates a longer-training follow-up (capped
intentionally at 1000 epochs here, per the experimental design) before any
claim about a smaller LR's achievable accuracy can be made.

---

## 39. Nx=500, Nt=10 longer-training diagnostic (restoring the original training duration)

**Hypothesis.** The original pre-refactor notebook/script trained for
5000 epochs; the current canonical configuration trains for 1000. Direct
inspection (git history) found this to be the **only** real numerical
difference between the original baseline and the current canonical
RF-only setup (besides the retired `N_TRIALS=10→1` change) — does
restoring the original training duration substantially improve the
canonical Nx=500, Nt=10 result?

**Experiment.** Continuation from the existing canonical seed=42,
1000-epoch checkpoint to 2000 and 5000 epochs, via repeated `train()`
calls on the same model/optimizer (same proven segmented-continuation
pattern as Section 32), with collocation points regenerated
bit-identically via `set_seed(42) → sample_training_points`. Script:
`Burgers/scripts/test_rf_nx500_long_training.py` (`bf4f24b`). Output:
`Burgers/outputs/rf_nx500_long_training/summary.csv` (verified locally).

**IMPORTANT caveat (stated explicitly, not to be dropped in any later
summary):** this project's checkpoint format has never stored optimizer
state — only `model_state_dict` and metadata. This continuation therefore
uses a **freshly initialized Adam optimizer** at epoch 1000 (zero moment
buffers), not a resumption of the original optimizer's internal state.
**This is not an exact uninterrupted 5000-epoch run** — it is "trained
weights at epoch 1000, continued with a fresh optimizer," which is a
different (and weaker) claim than a true continuous 5000-epoch trajectory.

**Result:**
```
epoch 1000: global=0.27149430, smooth=0.24438271, shock=0.83904880, PDE=0.23590008
epoch 2000: global=0.27013501, smooth=0.24283171, shock=0.83915919, PDE=0.23034099
epoch 5000: global=0.27041999, smooth=0.24296839, shock=0.84220162, PDE=0.22450591
```

**Interpretation.** Global and smooth relative L2 improve only slightly
from epoch 1000 to 2000 (global: 0.2715→0.2701, ≈0.5%) and then do **not**
improve further by epoch 5000 (global ticks back up to 0.2704). **Shock
error does not improve at all** — it is flat-to-slightly-worse across all
three checkpoints (0.83905→0.83916→0.84220), even as the sampled PDE loss
continues to decrease monotonically (0.23590→0.23034→0.22451). This is the
same PDE-residual-vs-solution-error dissociation already seen in Section
31 (and in Sections 27–29/33 for the local-correction work): **lower PDE
loss does not imply lower solution error**, here most clearly in the shock
region. Given the fresh-optimizer caveat above, this result should be read
as "restoring the original epoch count, under a necessarily
re-initialized optimizer, gives at most a very small improvement and does
not rescue the shock-region error" — not as a clean reproduction of what
the original 5000-epoch run would have produced with its own
(unrecorded) optimizer trajectory.

---

## 40. October 21, 2026 meeting checkpoint

**Status at the meeting.** Sections 36–39 above were presented: seed 42 is
not an unusually poor realization among seeds 20–24 (Section 36); the
fixed-ratio Nx/Nt=50 sweep through `(3000,60)` shows non-monotonic error
with no systematic gain from capacity beyond the canonical scale (Section
37); smaller learning rates are worse at a fixed 1000-epoch budget but are
still improving, not converged (Section 38); and restoring the original
5000-epoch training duration (under a necessarily re-initialized
optimizer) gives only a small global/smooth improvement and no shock-region
improvement (Section 39).

**Advisor direction.** The professor requested studying a self-adaptive
PINN method from a paper that includes a viscous Burgers benchmark
(Section 4.1 of that paper), where per-collocation-point weights are
assigned and adjusted during training. Exact quote: "For the self-adaptive
PINN, the idea is assigning weights to each collocation points and
adjusting them during the training. Please look at the paper and try to
reproduce their results. They did have viscous Burgers' equation (see
Section 4.1)." **Paper reference/citation: not yet supplied — placeholder,
to be filled in once available.**

**Proposed workflow (not yet started):** (1) read and understand the
paper; (2) reproduce its self-adaptive PINN viscous Burgers experiment;
(3) investigate transferring adaptive collocation-point weighting to the
RF model; (4) investigate loss-function modifications for RF-only and
RF+`u_local`, motivated directly by the PDE-residual-vs-solution-error
mismatch already documented in Sections 27–29, 33, and 39 above. See
`docs/meetings/2026-10-21.md` for the full meeting record and `TODO.md`
for the reordered actionable plan.
