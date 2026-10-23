# Phase 4: a gradient-boosted classifier and what it gains

## 1. What the classifier is for

The phase 3 fit uses one number per event, m4l. Signal and background also differ in how the
four leptons are produced: how hard they are, how the Z bosons are oriented, how central the
system is. A classifier compresses those differences into one score s(x) between 0 and 1.
We do not fit the score. We use it to split events into a high-score category (purer in
signal) and a low-score category, and fit m4l in both at once. If the categories have
different S/B, the combined fit separates signal from background better than the inclusive
one. The question this phase answers is how much better, measured honestly.

## 2. Gradient-boosted decision trees

A decision tree splits the feature space into boxes by a sequence of cuts, "x_j < c", and
assigns a value to each box (leaf). One tree is a weak, coarse model. Boosting adds many of them:

    F(x) = F_0 + η Σ_m f_m(x),     s(x) = 1 / (1 + e^{−F(x)})

F is the log-odds of signal, and η < 1 is the learning rate. The model is fitted by
minimizing the weighted binary cross-entropy

    ℓ = − Σ_i w_i [ y_i ln s(x_i) + (1 − y_i) ln(1 − s(x_i)) ]

one tree at a time. Tree m is grown on the gradient g_i = ∂ℓ_i/∂F and the curvature
h_i = ∂²ℓ_i/∂F² of every event at the current F. For the cross-entropy, g_i = w_i (s_i − y_i)
and h_i = w_i s_i (1 − s_i). A leaf containing events L gets the Newton step
−Σ_L g / (Σ_L h + λ), where λ is an L2 penalty. A split is chosen where it lowers the
second-order estimate of the loss the most. Each tree corrects what the previous trees got
wrong. The small η and the limits on tree size (leaf count, events per leaf) keep each
correction small, and that is the main defence against fitting noise.

scikit-learn's `HistGradientBoostingClassifier` first bins every feature into at most 255
quantile bins. Finding a split then costs a scan over 255 histogram bins instead of over all
events. This makes training on 170k events take seconds. With early stopping, a fixed
fraction of the training events is held out, and boosting stops when the loss on them has
not improved for a set number of iterations.

**What the score means.** With enough data and flexibility, the minimizer of ℓ is
s(x) = W_s f_s(x) / (W_s f_s(x) + W_b f_b(x)). Here f_s and f_b are the weighted feature
densities and W_s, W_b the class totals. This is a monotonic function of the likelihood ratio
f_s/f_b, the optimal test statistic (Neyman-Pearson). A cut on s is then the best cut on
these features. The class totals only shift where a given score sits, not the ordering. We
rescale both classes to the same total, so the score is spread over 0 to 1.

**Why the training weights are |w|.** With a negative weight, the event's term in ℓ is
unbounded below, and its h_i is negative. A leaf dominated by such events can have
Σh ≤ 0, and the Newton step diverges. The training weights are therefore |w|, as the phase 1
notes anticipated. The classifier then learns the |w|-weighted background, which has a
slightly different mixture from the true one. That can make the classifier a little
suboptimal, but it cannot bias the result, because every yield, template and significance
afterwards uses signed weights.

**The fallback: logistic regression.** The boosted trees overtrained on the background
(section 9), so the analysis uses the simplest classifier instead. Logistic regression models
the log-odds as a linear function of the standardized features, F(x) = β₀ + Σ_j β_j x_j, and
minimizes the same weighted cross-entropy (with the default L2 penalty, C = 1). It has 11
parameters for 166k signal and 289 background training events, so it cannot fit individual
events. The price is that it sees only monotonic, additive trends in each feature.

## 3. Features, and why m4l is not one of them

| Feature | Why it separates |
| --- | --- |
| pT of the four leptons, sorted | Reducible backgrounds (Z+jets, tt̄) contain soft leptons from heavy-flavour decays |
| m12, m34 | The Z1/Z2 mass sharing differs between H → ZZ* and the non-resonant processes |
| pT of the 4l system | VBF, VH and tt̄H Higgs events recoil against jets; qq̄ → ZZ* is soft |
| rapidity y of the 4l system | gg → H is central; qq̄ → ZZ* uses a valence quark and is more forward |
| cos θ* of Z1 in the 4l rest frame | A spin-0 decay is isotropic (flat before acceptance); qq̄ → ZZ* peaks near ±1 |
| smallest ΔR between leptons | Lepton isolation and topology of non-prompt leptons |

The leptons are sorted by pT, because their stored order is not pT-ordered (phase 1). θ* is
the angle between the Z1 momentum and the beam (z) axis, after boosting Z1 into the rest
frame of the four leptons. ΔR = √(Δη² + Δφ²) with Δφ wrapped into [−π, π]. The rapidity is
y = ½ ln((E + p_z)/(E − p_z)).

**m4l is excluded** because the score is used together with an m4l fit. A classifier that
saw m4l would learn that events near 125 GeV are signal-like. The background in the
high-score category would then be pushed into a peak under the signal: the classifier would
sculpt the background into the shape it is supposed to be distinguished from. The free
background normalization would become nearly degenerate with μ. A mismodelled background
could then fake or hide a signal, and the m4l fit would partly count the same information
twice.

Excluding m4l does not make the score independent of m4l. Kinematics ties the features to
it. For example, m12 + m34 ≤ m4l, so in 110-160 GeV the allowed m34 range grows with m4l, and
the four lepton pT also scale with it. A background event at 155 GeV can have a larger m34
than any 125 GeV Higgs event. The classifier can learn that, and learning it is learning
m4l. Two consequences:

- The classifier is trained only on events with 110 < m4l < 160 GeV, the range it will be
  used in. Training on 80-250 GeV would include the on-shell ZZ peak, where "m34 ≈ 91 GeV
  means background" is almost entirely a statement about m4l > 180 GeV.
- The residual correlation is measured: this is the sculpting check of section 8.

## 4. The even/odd split

The classifier must be tested on events it was not trained on. The score of a training event
is optimistic, because the trees partly fit that event's own fluctuation. We split by the
parity of eventNumber: even events train, odd events test. The parity of the event number is
unrelated to the physics of the event, so the two halves are statistically equivalent
samples of the same population. The split is reproducible without a random seed, and an
event is always in the same half in every script. In the simulation, 50.1% of signal and
49.9% of background events in the selection are even.

Each half contains about half the events, still with weights normalized to the full 36.6 fb⁻¹.
A yield computed on one half is therefore an estimate of half the expected yield, and we
multiply the test-half yields by 2. The bin errors √(Σw²) scale by 2 as well, so the relative
MC uncertainty of a half-sample template is √2 larger than that of the full sample. That is
the honest statistical precision of what we evaluate on. For a fair comparison, the inclusive
model the gain is measured against is also built from the odd half ×2. The phase 3 value
from the full sample is quoted next to it.

The background is the weak point. In 110-160 GeV there are only 567 simulated background
events, 289 of them even, and one Z+jets event (sample 700323) has weight 5.8 out of a total
of 53.7. A classifier trained on 289 weighted events can learn only coarse features. Whether
that one event lands in the training or the test half matters, and we report where it is.

## 5. Overtraining check (P4-a)

If the classifier has fitted noise, its score distribution on training events differs from
that on test events: training signal sits higher, training background lower. For each class
we compare the two distributions with a two-sample Kolmogorov-Smirnov test. Its statistic is
D = max_s |F_train(s) − F_test(s)|, the largest gap between the empirical cumulative
distributions. Under the null hypothesis, that both are samples of the same distribution,
D has a known distribution. The plan required p > 0.05. After the boosted trees failed it,
the p-values are reported with the score figure, with no target (section 9).

The test uses unweighted events. Its null distribution is then exact. Because the parity split
is random with respect to the physics, the sample composition is the same in both halves.
Unweighted, however, the test is blind to overtraining on a few events with large weights,
which matter most for the yields. A weighted version is therefore reported next to it: the
CDFs are weighted with |w|, and the sample size in the p-value is the effective number
n_eff = (Σw)² / Σw². That version is only approximate.

## 6. Choosing the category threshold

The two categories are score > t and score ≤ t. The value of t is chosen on the training
half only. Choosing it on the test half and then quoting the test-half gain would select a
favourable fluctuation and bias the gain upward. Once t is fixed on independent events, the
gain measured on the test half is unbiased for that t.

Candidate thresholds are the training-signal score quantiles that keep 30%, 35%, ..., 95% of
the signal in the high-score category. A grid in signal efficiency guarantees that neither
category is empty of signal, which a grid in raw score would not. For each candidate we build
the full two-category model of section 7 from training-half templates ×2 and compute its
Asimov Z. We keep the largest. The training-half scores are optimistic, so this figure of
merit overstates the gain, but it only chooses t. The overstatement does not enter the
reported number.

## 7. The two-category model and why an uninformative split gains nothing

The phase 3 model is duplicated into two channels, one per category, in 110-160 GeV with
2.5 GeV bins. The signal template of each channel comes from that category's own simulated
events. The channels share μ and the luminosity factor on the signal. Each channel has its own
free background normalization and its own staterror factors. A shared background
normalization would assume the simulation predicts the fraction of background in each
category. Phase 3 found the data background in this range at twice the simulation, so we do
not assume the simulation knows its composition.

**Background shape.** The simulated background is too sparse for a template per category.
With the events split by score, 2.5 GeV bins in the high category were empty or had a negative
sum of weights, which is not a valid Poisson mean, and an empty bin under signal makes the
μ = 0 fit infinite. Each category's background is therefore the inclusive background template
times that category's background fraction f (signed weights), with the bin errors times √f.
The √f is what the errors would be if the category held a fraction f of the simulated events
with similar weights. The categories then differ in S/B but not in background shape, and this
assumes the score does not sculpt the background (section 8 tests it).

The test of the machinery (P4-c) uses an exact identity. Take the inclusive templates s_i, b_i
with errors σ_i and split them by a fixed fraction f: one channel gets f·s_i, f·b_i with
errors √f·σ_i, the other gets the rest. In channel f, the Poisson term becomes

    ln Pois(f n_i | f ν_i) = f [ n_i ln ν_i − ν_i ] + const,

that is, f times the inclusive term. The staterror constraint has relative width δ_i/√f, so
its log is −f (γ_i − 1)² / (2δ_i²), which is also f times the inclusive one. The log-likelihood
of channel f is therefore f times the inclusive log-likelihood, written in its own nuisance
parameters. Channel 1 − f is the same with 1 − f. For fixed μ and luminosity, both channels
are maximized by the same nuisance values as the inclusive model. The profiled log-likelihood
of the pair equals the inclusive one for every μ, so q0 and Z_A are identical. A split that
carries no information gives exactly zero gain, despite the extra free parameters. Any gain
we measure comes from the categories having different S/B or different shapes.

## 8. Sculpting check (P4-b)

Sculpting means that the cut on the score changes the m4l shape of the background. The danger
is a background that piles up near 125 GeV in the high-score category and so looks like the
signal. Here the check matters twice, because the category model of section 7 assumes that
each category has the inclusive background shape. P4-b tests exactly that assumption.

On the odd background events in 110-160 GeV, in 5 bins of 10 GeV, we compare the normalized
m4l histograms of the two categories. With two categories this is the same as comparing each
with the inclusive one. The inclusive histogram is the weighted sum of the two, so either
category equals it in shape if and only if the two categories equal each other in shape. The
direct comparison uses two independent samples, while category vs inclusive would compare a
sample with a superset of itself.

For weighted histograms with bin sums W_1i, W_2i, sums of squared weights S_1i, S_2i and
totals W_1, W_2, the normalized contents W_1i/W_1 and W_2i/W_2 estimate the same
probabilities under the null. Their difference has variance ≈ S_1i/W_1² + S_2i/W_2².
Multiplying numerator and denominator by W_1² W_2² gives

    X² = Σ_i (W_1 W_2i − W_2 W_1i)² / (W_1² S_2i + W_2² S_1i)

which is approximately χ² with k − 1 = 4 degrees of freedom. One degree is lost to the
normalization (Gagunashvili, Nucl. Instrum. Meth. A 614 (2010) 287, arXiv:0905.4221; the form
used in ROOT's TH1::Chi2Test "WW"; to be verified against the paper before the report). The
plan required p > 0.05; the p-value is now reported with the figure, with no target. The bins are 10 GeV wide, not 2.5, so that each holds tens of simulated
events and the χ² approximation is usable.

**Limits of the test.** With fewer than 300 background events, the test can only detect large
distortions, so passing it is weak evidence. We therefore also report the weighted
correlation between score and m4l for background events, the mean m4l in each category, and
the figure of the two normalized shapes. A visible slope there is reported even if the χ²
passes.

## 9. Choices, and what changed after the first model failed

Fixed before any result, and kept:
- Paired selection, 110 < m4l < 160 GeV, all 8 signal samples, every other sample as
  background.
- Training on even eventNumber with |w| rescaled to equal class totals. Evaluation on odd
  eventNumber with signed weights, yields ×2.
- Threshold grid: high-category signal efficiency 0.30-0.95 in steps of 0.05, figure of
  merit = two-category Asimov Z on the training half.
- Two-category model: phase 3 binning and modifiers, with a free background normalization
  per category. Sculpting: 5 bins of 10 GeV, weighted χ².

Changed on simulation, after P4-a failed. No data were used for any of these.
1. The planned boosted trees (learning rate 0.05, up to 500 iterations, 15 leaves, early
   stopping) stopped after 62 iterations, and their background scores disagreed between
   halves (KS p ≈ 1e-10; this first model is not kept in results/). The cause is the
   imbalance in event counts. With 166k signal and 289 background training events balanced
   by weight, one background event weighs about 575 signal events, so a leaf of 50 events
   can be one background event plus 49 signal events and still be background-dominated. The
   trees isolate single background events.
2. Smaller trees (depth 3, 100 iterations, at least 50 events per leaf, L2 1.0, no early
   stopping) still disagreed: background KS p = 8.6e-14 (results/ml.json,
   `train_test_ks_by_model`). By the rule fixed with this change (KS p < 0.05 for either
   class), the analysis switched to logistic regression on the same features.
3. P4-a and P4-b became reported checks with no pass/fail target (docs/REFERENCE_VALUES.md).
4. Per-category background templates were replaced by the inclusive shape times the category
   fraction (section 7), because the per-category templates had empty and negative bins.

## 10. Results on simulation

From `scripts/classifier.py` → results/ml.json (odd half, signed weights, ×2 unless stated):

- **Classifier:** logistic regression. Train vs test KS p = 0.98 (signal) and 0.43
  (background); weighted versions 0.32 and 0.97. No sign of overtraining.
- **ROC AUC:** 0.702 on the test half, 0.763 on the training half. The difference is not
  overtraining, since the model has 11 parameters. It is mostly one Z+jets event (700323,
  weight 5.8) that fell in the training half at score 0.30, where it carries about a fifth of
  the training background weight. It is the flat stretch of the training ROC curve.
- **Threshold:** score > 0.750, which keeps 30% of the training signal: the edge of the grid.
  The training-half scan is noisy and not monotonic (Z_A from 5.5 to 6.8), and its maximum
  is a fluctuation of the 289 training background events as much as a real optimum.
- **Categories (test half ×2):** high S = 10.9, B = 4.9; low S = 29.7, B = 41.6.
- **Expected significance:** inclusive Z_A = 6.06 and two categories Z_A = 5.97, both
  from the test half. **The classifier does not improve the expected significance**
  (−0.09). The full-sample phase 3 value is 6.64. The odd half alone gives less because its
  templates have √2 larger MC errors and are a different fluctuation.
- **Why there is no gain (likely, not tested separately):** the high category has a
  background of only about 5 events and its own free normalization, which only its own
  sidebands can fix. The S/B gain of the purer category is spent on that extra unknown, and
  the threshold chosen on optimistic training scores is tighter than the test half supports.
- **Sculpting (P4-b):** high vs low χ² = 4.3 for 4 degrees of freedom, p = 0.37. This is weak
  evidence, as section 8 warned. The weighted mean m4l of the background is 129.4 GeV in the
  high category and 138.5 GeV in the low category, and the score-m4l correlation for
  background is −0.38. The score does push the background towards the signal region: a
  classifier on m12, m34 and the lepton pT learns part of m4l (section 3). The inclusive-shape
  assumption of the category model is therefore only approximate. It puts too little
  background under the peak in the high category, which if anything makes the category
  result optimistic.

## 11. Result on data (unblind step)

The frozen classifier and threshold were applied once to the 124 data events in 110-160 GeV
(`scripts/classifier_data.py` → results/ml_data.json). The templates are the odd simulation
half ×2, the same as for the expected numbers.

- **Two categories: observed Z = 3.29** (asymptotic, p0 = 5.0e-4), expected 5.97.
  μ̂ = 0.62 +0.24 −0.22. The data split into 20 high-score and 104 low-score events. The
  simulation predicts 15.8 and 71.3.
- **Inclusive with the same templates: Z = 3.90**, μ̂ = 0.73 +0.22 −0.20. Phase 3, with the
  full-sample templates, gave 3.96.
- The categories lose about 0.6σ on data against 0.09σ expected. This is not a tested claim,
  but it is consistent with section 10: the high category's background has its own free
  normalization fixed by very few events, and the model's inclusive background shape puts
  too little background at low m4l in that category. Three of the 20 high-score data events
  are in the first bin, 110-112.5 GeV. The inclusive phase 3 fit stays the headline result.
