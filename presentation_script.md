# Restaurant Survival Classification — 3.5-Minute Presentation Script

**Target:** ~3 min 30 sec spoken (~480–520 words at a natural pace).
**Focus:** algorithms considered → selection process → expected results.

---

## Slide 1 — The problem (≈ 25 sec)

>The classes are imbalanced — only about **10 %** of restaurants in the training set are closed — so the official scoring metric is **Balanced Accuracy (BA)**, which weights the two classes equally.

---

## Slide 2 — Feature engineering (≈ 25 sec)

> Before model selection I added roughly **40** engineered features: log-transforms on heavy-tailed counts (reviews, POIs, restaurant counts), recency-share ratios (1m / 3m / 12m of total reviews), **momentum** features (e.g. `momentum_3m_to_12m`), local-**competition** ratios (`reviews_per_local_restaurant`, `restaurant_density`), and rating-strength interactions. These features consistently raised BA across every candidate, and the `catch_…` competition columns turned out to be among the most influential.

---

## Slide 3 — Algorithms considered (≈ 50 sec)

> I screened five base learners with **5-fold stratified cross-validation** on the engineered feature set:
>
> 1. **L2 Logistic Regression** — strong linear baseline, calibrated probabilities, handles imbalance through `class_weight`.
> 2. **L1 Logistic Regression** — same family, but with embedded feature selection.
> 3. **K-Nearest Neighbours (k = 25, distance-weighted)** — non-parametric, sanity check for local structure.
> 4. **Decision Tree (depth-8, balanced weights)** — interpretable non-linear baseline.
> 5. **Random Forest (500 trees)** — bagged tree ensemble, captures non-linear interactions among competition and recency features.
>
> KNN was clearly the weakest — it suffers in the high-dimensional, mostly numeric feature space after one-hot encoding. The single Decision Tree underfit. The two **logistic** models and the **Random Forest** all clustered at the top of the leaderboard, so they became the candidates for hyperparameter tuning.

---

## Slide 4 — Selection process (≈ 55 sec)

> Hyperparameters were then tuned with **OOF CV** — using two grids:
>
> - **Logistic Regression:** swept `C ∈ {0.10, 0.20, 0.40, 1.00}` and a positive-class-weight multiplier `∈ {0.5 … 1.0}`. Best L2 config: `C = 0.40`, weight multiplier `0.60`.
> - **Random Forest:** swept `n_estimators`, `max_features` and `min_samples_leaf`.
>
> Because the linear model and the tree ensemble make **different mistakes**, I tested three ensemble strategies on a held-out 20 % validation set:
>
> | Ensemble                | Mechanism                                |
> | ----------------------- | ---------------------------------------- |
> | Soft **voting**         | average of `predict_proba`               |
> | **Stacking**            | logistic meta-learner over base outputs  |
> | **Probability blend**   | tuned convex combination of two models   |
>
> The blend won. A grid search over blend weights chose **0.775 × elastic-net + 0.225 × bagged trees**, where the linear model is now **elastic-net** (`l1_ratio = 0.10`) for embedded selection plus L2 stability, and the tree side uses **400 bagged decision trees** (`min_samples_leaf = 20`).
>
> Finally, the decision **threshold** was tuned on the validation probabilities — about **0.43**, not the default 0.5 — which directly recovers BA points lost to class imbalance.

---

## Slide 5 — Expected results (≈ 35 sec)

> Honest expectations, all measured **without touching the test set**:
>
> - **Tuned L2 logistic, OOF with optimal threshold:** **BA ≈ 0.677**.
> - **Elastic-net + Bagging blend, hold-out validation BA\*:** **≈ 0.68 – 0.69** at threshold ≈ 0.43.
> - **Honest 20 % internal test BA** (single look, after full OOF tuning of weights + threshold): in the same **0.67 – 0.69** range.
>
> So my expectation on the Kaggle test set is **Balanced Accuracy in the high 0.6s**, with the blend giving a modest but consistent **+0.005 – 0.01 BA** lift over the best single model.

---

## Slide 6 — Why this choice (≈ 20 sec)

> Elastic-net gives a **regularised, interpretable** linear backbone with embedded feature selection; bagged trees add **non-linear** signal on competition and recency features; the threshold tuning addresses the imbalance head-on; and every hyperparameter was selected on OOF data, so the reported BA is an honest estimate of test performance.

---
