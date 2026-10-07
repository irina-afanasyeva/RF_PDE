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
