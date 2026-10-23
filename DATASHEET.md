# Datasheet for the BBO Capstone Dataset

## 1. Motivation

### Origin
The dataset was created as part of the 2026 Imperial College Machine Learning and AI capstone challenge, involving the optimisation of eight unknown black-box functions under a limited evaluation budget.

The objective was to maximise each function by progressively selecting input coordinates, observing the corresponding outputs and using those observations to inform subsequent queries.

### Task
The dataset supports sequential black-box optimisation, particularly Bayesian optimisation (BO) using Gaussian Process (GP) surrogate models. It supports comparing alternative modelling approaches, examining uncertainty, identifying promising regions and evaluating the exploration/exploitation trade-off.

An important part of the exercise was understanding how different optimisation approaches perform when observations are limited and the underlying function is unknown.

## 2. Composition

### Datasets
The dataset contains input coordinates and corresponding scalar outputs for eight functions.

Each function has a different dimensionality:

| Function | Input dimensions | Initial observations | Final observations (after 13 rounds) |
|----------|------------------|----------------------|---------------------------------------|
| F1 | 2 | 10 | 23 |
| F2 | 2 | 10 | 23 |
| F3 | 3 | 15 | 28 |
| F4 | 4 | 30 | 43 |
| F5 | 4 | 20 | 33 |
| F6 | 5 | 20 | 33 |
| F7 | 6 | 30 | 43 |
| F8 | 8 | 40 | 53 |

The initial observations were provided by the programme. Additional observations were generated through successive weekly submissions. All input variables are bounded between 0 and 1, with coordinates specified to six decimal places.

### Data format
The original data is stored in NumPy files, with subsequent queries and results recorded in text and JSON formats. These were subsequently consolidated into a harmonised CSV dataset containing the function identifier, round, input coordinates and observed output. This provides a consistent dataset for modelling and analysis while preserving the original observations separately.

### Known gaps
The dataset is relatively sparse, particularly for higher-dimensional functions. There is also a deliberate sampling bias towards promising regions. As the optimisation progressed, queries increasingly concentrated around areas that had previously produced strong results. Consequently, substantial portions of the search space remain unexplored.

Some functions also demonstrated unusual characteristics, including extremely small output values, multiple promising regions and apparent stochasticity, demonstrated by functions providing differing results for identical input variables across different submissions. Specifically, F3 returned different outputs for an identical input, suggesting that observation noise needed to be considered.

## 3. Collection Process

### Query generation
The approach evolved considerably over the 13 weekly rounds, as follows.

- Initially, query selection was relatively heuristic during the learning phase. This relied on examining observations, identifying patterns and selecting promising coordinates.
- Subsequently, GP surrogate models and Bayesian optimisation techniques were introduced to estimate the underlying functions and identify candidate queries.
- As more observations became available, the approach became increasingly function-specific.
- Rather than applying a single GP configuration to every function, the methodology expanded to include comparisons of kernels, hyperparameters, alternative surrogate models and acquisition strategies.
- Out-of-sample diagnostics, including leave-one-out cross-validation (LOOCV), were increasingly used to assess model reliability.
- Later rounds added trust-region methods (TuRBO), neural-network cross-checks, calibration checks and, for one function, regression.
- LLMs were used selectively to assist with coding, interpreting diagnostics, comparing modelling approaches and challenging recommendations.
- However, final query selection remained a judgement based on the observed data and the available evidence.

### Time frame
The dataset was developed over 13 successive weekly rounds of the capstone challenge (rounds 1 to 13, after the initial observations in round 0), with one additional query permitted per function in each round.

## 4. Preprocessing and Uses

### Preprocessing applied
The original inputs were already normalised to the interval [0,1].

Depending on the function, preprocessing and modelling included:

- Standardisation of output variables where appropriate.
- Examination of output distributions and extreme values.
- Consideration of transformations for highly skewed functions.
- Assessment of observation noise.
- Comparison of alternative kernel specifications and hyperparameters.

Importantly, transformations were not applied uniformly. They were considered where there was evidence that they could improve modelling performance.

### Intended use
The dataset is intended for:

- Sequential black-box optimisation.
- Training and evaluating surrogate models.
- Comparing alternative optimisation approaches according to the idiosyncrasies of individual functions.
- Investigating exploration/exploitation trade-offs.
- Assessing how additional observations affect model performance.

The dataset should not be used to make definitive claims about the global structure of the underlying functions. The relatively small and non-random sample also limits its suitability for conventional supervised learning or for drawing general conclusions about the performance of particular optimisation algorithms.

Also, results should not be generalised to unrelated optimisation problems without further validation.

## 5. Function-level datasheets

Each function behaved differently, so the questions below are answered for each one. Observation counts are final counts after 13 rounds. Fuller narratives are in [`HISTORY.md`](HISTORY.md), and the notebooks are `notebooks/f1.ipynb` to `f8.ipynb`.

### Function 1: contamination detection (2D)

- **Overview.** Detecting contamination sources in a two-dimensional area, where only proximity to a source gives a non-zero reading. Two inputs, 10 initial points. The output is a detection reading that is essentially zero almost everywhere, spanning more than 120 orders of magnitude in absolute value.
- **Nature of the data.** Initial arrays X (10, 2) and y (10,), growing to 23 observations. The first useful signal came in round 4. After that, queries followed a narrow line (rounds 5 to 11), then took small steps off it (rounds 12 to 13). No input was ever repeated, so noise could not be assessed and the function was treated as noise-free. It looks like a thin ridge of positive values in a flat, near-zero field.
- **Strategy.** A local Gaussian process (RBF kernel, one length scale per input) on a signed-log transform, fitted only to observations near the incumbent, with expected improvement, probability of improvement, upper confidence bound and posterior-mean checks, a quadratic fitted along the trajectory, and small directional probes. A global model was not useful because almost every output is indistinguishable from zero. The acquisition winners always sat on the search-box boundary and were not used. Exploitation followed the ridge; exploration was limited to small perpendicular steps. Early rounds used a GP with UCB on the flat surface, then the approach became local once a signal appeared.
- **Preprocessing and modelling.** Inputs were already in [0, 1]. The signed-log transform scale was set from the data. No observations were removed: round 8 was a valid evaluation (at another function's input) that lies far from the ridge.
- **Learning.** Rounds 4 and 5 (the first non-zero reading, then a value about 50 times larger) and round 12 (a perpendicular step that improved on the line) were the most informative. A quadratic and the GP agreed on a maximum along the line, but later gains showed it was not the true local maximum. With hindsight, local modelling from the first non-zero reading and earlier perpendicular probes would have helped, and one query was lost to a submission error.
- **Results.** Best output **4.53 × 10⁻⁶** in round 13 at input `0.671996-0.699874`, up from 7.7 × 10⁻¹⁶ at the start. Confidence that this is near the global maximum is **low**: only one ridge was found, the value was still rising (12.8% in the last round) and the two-dimensional space is barely covered. The behaviour matched the description.

### Function 2: noisy log-likelihood score (2D)

- **Overview.** A noisy "mystery model" returning a log-likelihood-type score, with several local peaks. Two inputs, 10 initial points. Observed outputs run from −0.066 to 0.749.
- **Nature of the data.** X (10, 2) and y (10,), growing to 23. Round 2 found a second high region near (0.70, 0.25) and round 4 reached 0.648 near (0.69, 0.87). Rounds 5 to 9 alternated between regions, rounds 10 and 11 moved to x1 ≈ 0.51 for the best value, and rounds 12 to 13 (0.605, 0.688) did not beat it. The function is noisy: no input was repeated exactly, but results varied between nearby inputs by more than a smooth surface would explain, so the models carried a fixed noise floor and single results were read with caution. It is multimodal, with at least three high regions.
- **Strategy.** An ARD Matérn 3/2 GP with the noise floor as its noise term, scored with posterior mean, EI, PI and UCB on a dense grid, then local refinement along x1 and a noise-sensitivity check against a latent incumbent. Noise and several peaks called for a noise-aware model and local checks around each mode. Exploration alternated between modes, then refined the new region.
- **Preprocessing and modelling.** Inputs in [0, 1]; outputs normalised; no transformation; no observations removed. The fitted length scale for x2 was 100 (treated as irrelevant) against 0.05 for x1.
- **Learning.** Round 2 revealed a second mode, and rounds 10 to 11 showed the best region was elsewhere. Predictive accuracy was modest (leave-one-out RMSE 0.19, MAE 0.12 through round 12). Testing noise earlier and modelling mode by mode would have helped.
- **Results.** Best output **0.749040** in round 11 at `0.507000-0.550000`. Confidence is **low to moderate**: the function is noisy, so the best may reflect a favourable draw, and later probes were lower. The behaviour matched the description (noisy, several local peaks).

### Function 3: drug combinations (3D)

- **Overview.** Testing combinations of three compounds, minimising side effects, framed as maximisation (the negative of side effects). Three inputs, 15 initial points. All outputs are negative, from −0.399 to −0.0072.
- **Nature of the data.** X (15, 3) and y (15,), growing to 28. The function is noisy: the round-1 incumbent, repeated in round 5, returned −0.0125 and −0.0271. After an exploratory move in round 6, round 9 gave the best result, and rounds 10 to 13 stayed near the centre without beating it. Only x3 had well-identified structure.
- **Strategy.** Two GPs (RBF and Matérn 3/2) on a Yeo-Johnson transformed target, with candidates ranked by consensus across both models and four criteria, plus a corridor search between the incumbent and the next-best region. Noise and few observations favoured model agreement over trusting a single optimum. Round 6 was a deliberate exploratory step; later rounds exploited the central region. The strategy shifted from exploration to consensus ranking.
- **Preprocessing and modelling.** Yeo-Johnson improved grouped leave-one-out accuracy over the raw target; a white-noise term was fitted; repeated inputs were held out together in validation. No observations were removed.
- **Learning.** Round 10 contradicted round 9's direction, so the push to higher x1 was abandoned. The repeated incumbent was the most informative query because it exposed the noise. Predictive accuracy stayed limited (leave-one-out R² about 0.13 to 0.19 through round 12).
- **Results.** Best output **−0.00717** in round 9 at `0.628331-0.443178-0.440136`. Confidence is **low**: later results are within the observed noise, so the best may be a favourable draw. The behaviour matched the description.

### Function 4: warehouse product placement (4D)

- **Overview.** Placing products across warehouses, where the output is the difference from an expensive baseline calculation. Four inputs, 30 initial points. Outputs run from −32.6 to 0.731, and 10 of the 43 are positive.
- **Nature of the data.** X (30, 4) and y (30,), growing to 43. Changing one coordinate by 0.03 turned −0.137 into the first positive value (0.310) in round 4. The positive region is small and queries stayed local to it. The best (round 8) was not beaten in the five rounds that followed (latest 0.660). No input was repeated, so noise was not assessed. The shape is a narrow positive basin among mostly negative results.
- **Strategy.** A GP (Matérn 3/2, one length scale per input) as the primary model, with a neural-network ensemble and a TuRBO trust-region model scoring the same local candidates, and a Sobol search around the incumbent that excluded evaluated points. A support-vector classifier was tried and dropped. The positive region is narrow and exploring away from it is costly, so the search stayed local. A neural network was used in round 6 because the GP looked too smooth; a neural-network-backed proposal in round 11 returned 0.199, so the GP was reinstated.
- **Preprocessing and modelling.** Standardised inputs and a raw target (a Yeo-Johnson target did not help in leave-one-out comparison); the noise parameter was chosen by leave-one-out cross-validation. No observations were removed.
- **Learning.** Rounds 4 and 8 were the most informative. Leave-one-out R² was 0.925 for the GP and 0.966 for the neural network through round 12. Comparing model classes earlier and keeping a model vote would have helped.
- **Results.** Best output **0.730826** in round 8 at `0.420000-0.360000-0.360000-0.420000`. Confidence is **low to moderate**: the region is narrow and predictive uncertainty there is large. The behaviour matched the description (local optima).

### Function 5: chemical process yield (4D)

- **Overview.** The yield of a chemical process, described as unimodal. Four inputs, 20 initial points. Outputs run from 0.11 to 8,662.5.
- **Nature of the data.** X (20, 4) and y (20,), growing to 33. Early queries ran toward the upper boundary (1,410, 3,290, 4,477), then round 6 went to the corner (1, 1, 1, 1) and returned 8,662.48, with nearby probes in rounds 7 to 9. The corner was repeated in rounds 10 to 13. Those five evaluations returned identical values, so the function is deterministic. It is unimodal with the peak on the boundary, and a log-log fit on the corner observations had R² 0.9995.
- **Strategy.** Raw and log1p GPs (ARD Matérn 5/2), a conditional local search holding x1 at its incumbent value, and posterior-mean, EI, PI and UCB scoring. The very wide output range and the boundary behaviour called for a log model and a boundary check. Round 6 was a deliberate step to test x1, which had never been sampled above 0.37 near the optimum.
- **Preprocessing and modelling.** A log1p target (leave-one-out R² 0.9998, against 0.978 for the raw model through round 12); inputs already in [0, 1]; no observations removed.
- **Learning.** Round 6 showed x1 mattered and nearly doubled the yield. The corner repeats confirmed determinism and a boundary maximum. Testing the corner earlier would have saved rounds.
- **Results.** Best output **8,662.4825** in round 6 at `1.000000-1.000000-1.000000-1.000000`. Confidence that this is the global maximum within the domain is **high**: it sits on the boundary of a unimodal function, was reproduced exactly five times, and the raw and log models agree. The behaviour matched the description.

### Function 6: cake recipe score (5D)

- **Overview.** A recipe score combining flavour, consistency, calories, waste and cost, described as negative by design. Five inputs, 20 initial points. Outputs run from −2.57 to +0.0060, so a positive value did occur, which the description did not anticipate.
- **Nature of the data.** X (20, 5) and y (20,), growing to 33. The function is noisy (the fitted noise standard deviation was about 0.05 to 0.11). The best came in round 11 (+0.0060), followed by −0.348 in round 12 and −0.061 in round 13. Queries concentrated around a shallow optimum region.
- **Strategy.** A Gaussian process early on, then TuRBO-1 (Matérn 5/2 with a fitted noise term) with a trust region around the incumbent, then in the final round a walk-forward calibration check and a vote across four GP specifications between a conservative and a more exploratory candidate. Noise and few observations favoured conservative local refinement. After round 12, the GP was constrained and compared against alternatives.
- **Preprocessing and modelling.** Standardised outputs; length scales bounded to 0.05 to 2.0 and noise to 0.001 to 1.0; no transformation; no observations removed.
- **Learning.** The walk-forward check refits on earlier rounds to predict each round. Standardised errors reached +4.0 (round 9) and −5.1 (round 12), showing the model was overconfident. Round 12 was the most informative because it revealed that. Calibrating uncertainty earlier would have helped.
- **Results.** Best output **+0.00598** in round 11 at `0.429654-0.391293-0.666659-0.751633-0.055887`. Confidence is **low**: with noise of this size, a single best value may be a favourable draw, and later results were lower.

### Function 7: tuning an ML model (6D)

- **Overview.** The performance score of a model with six hyperparameters. Six inputs, 30 initial points. Outputs run from 0.0027 to 3.283, so values exceed the 0 to 1 range an accuracy score would have.
- **Nature of the data.** X (30, 6) and y (30,), growing to 43. Progress was steady: 2.31 in round 1 to 3.283 by round 13, with a reconstructed trust region that widened from 0.2 to 0.5. No input was repeated; the fitted noise level was very small (standard deviation about 0.005), so the function looks nearly deterministic and smooth.
- **Strategy.** Raw and log GPs, then TuRBO-1 (a GPyTorch Matérn 5/2 model with one length scale per input), with a gradient-boosted-tree challenger rejected in round 11 and a final quadratic interpolation through rounds 10 to 12. Smooth, steady improvement in six dimensions suited local trust regions. Exploitation dominated, with an exploratory draw reported alongside.
- **Preprocessing and modelling.** Standardised outputs; inputs in [0, 1]; computations in float32 on Apple MPS or CUDA when available; no observations removed.
- **Learning.** Rounds 6 to 9, when the trust region widened, gave the largest gains. By rounds 11 to 13 returns were tiny (3.2831 to 3.2833). Running diagnostics earlier on whether the surface was still improving would have helped.
- **Results.** Best output **3.28328** in round 13 at `0.230117-0.105291-0.483789-0.263529-0.313756-0.650578`. Confidence is **low to moderate**: the value was still creeping up and only one local region was explored.

### Function 8: eight-dimensional black box (8D)

- **Overview.** An eight-hyperparameter black-box function. Eight inputs, 40 initial points. Outputs run from 5.59 to 10.
- **Nature of the data.** X (40, 8) and y (40,), growing to 53. Weeks 1 to 5 used a Gaussian-process search and reached 9.955. Round 6 gave the maximum, 10.0. Rounds 7 to 9 were small deliberate steps around it and rounds 10 to 13 repeated it. Five evaluations at the optimum returned identical values, so the function is deterministic. It is smooth, unimodal and fits a separable quadratic.
- **Strategy.** A Gaussian-process and neural-network consensus search in weeks 1 to 5, then in week 6 a switch to ordinary least-squares quadratic regression (17 coefficients) with an analytic optimum. Gains from search had slowed to 0.155 over five rounds, so the problem was treated as regression instead.
- **Preprocessing and modelling.** Search stage: standardised inputs, a Yeo-Johnson target and a noise term chosen by leave-one-out cross-validation (leave-one-out R² about 0.99 in the round-5 workbook). Regression stage: none required.
- **Learning.** The regression fits every observation to machine precision (RMSE about 1.7 × 10⁻¹⁵), with exact round-number coefficients. A retrospective analysis suggests the structure could have been identified from the 40 initial rows alone. Running low-order structure checks before choosing a surrogate would have saved the search rounds.
- **Results.** Best output **10.000000** in round 6 at `0.100000-0.150000-0.130000-0.150000-0.800000-0.500000-0.200000-0.600000`. Confidence is **very high**: exact fit, correct forecasts of the confirmation rounds, symmetric probe results and identical replicates. The caveat is that this holds within the sampled domain. The observed maximum of 10 is on a different scale from the 0 to 1 accuracy described in the brief.

## 6. Distribution and Maintenance

### Dataset
The dataset is held in the project's GitHub repository (`postulix/bbo_final`), alongside the notebooks and documentation. The data and documentation are released under CC BY 4.0, so reuse requires only attribution; the code is under the MIT licence (see [`LICENSE`](LICENSE)). The black-box functions and their descriptions come from the Imperial College Business School Executive Education programme (with Emeritus), and the repository is shared for educational and portfolio use.

The original observations are preserved separately from subsequent submissions, while a consolidated CSV provides the working dataset. The project also contains Jupyter notebooks and supporting Python code used for modelling and optimisation.

### Terms of use
The dataset was supplied and developed for educational purposes within the capstone programme.

### Maintenance
The dataset is maintained by the project author, who incorporates new observations, validates the data and maintains the modelling notebooks.

## 7. Ethical, practical and general considerations

### Relation to real-world applications
The tasks mirror problems where each evaluation is expensive: contamination mapping (F1), drug-combination testing (F3), chemical process tuning (F5) and machine-learning hyperparameter tuning (F7). The weekly query limit stands in for the cost of a real experiment.

### Limitations of the synthetic functions
The functions are synthetic, so they are cleaner than real systems: there is no drift over time, no safety limit on inputs, and no competing objectives. Some are exactly solvable (F8), which real systems rarely are. The descriptions also did not always match the observed scales (F6 is described as negative but reached a positive value; F7 and F8 exceed the 0 to 1 range of an accuracy score), so documentation of a real problem should be checked against its data.

### Would the strategy scale?
Partly. Local trust regions and ARD Gaussian processes extend to higher dimensions, but Gaussian-process cost grows with the number of observations and more data would be needed for reliable uncertainty. A quick low-order structure check before any surrogate is cheap and scales well, but only helps if the function really is simple. For costly or safety-critical problems, replicate checks, constraints and human oversight would be essential.

### Risks for anyone analysing this data
- The best observed value may reflect noise (F2, F3, F6), not a true optimum.
- Surrogate uncertainty can be overconfident (the F6 calibration check showed this).
- The dataset is small and sampled mostly near promising regions, so conclusions about the whole input space are weak.
- Retrospective analyses (such as the F8 replay) are subject to hindsight bias: they show what the data would have supported, not what an analyst would have chosen.
- Extreme output scales (F1) can break standard modelling choices.
- LLM-assisted analysis can make basic errors, so recommendations need checking against the data.

