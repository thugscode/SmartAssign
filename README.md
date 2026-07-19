# SmartAssign

Predict-then-optimize employee-to-department assignment pipeline for the Kaggle dataset *Employee Performance and Productivity Data*.

## Problem Statement

The project predicts employee `Performance_Score` from available employee attributes, then uses those predictions inside an assignment model that maps employees to departments treated as project slots. The goal is to compare the ML-driven assignment against simpler baselines.

## Methodology

The workflow is split into five notebook steps plus one optimization script:

1. `notebooks/01_eda.ipynb` explores the data, checks missingness, visualizes the target, and reports correlations.
2. `notebooks/02_data_prep.ipynb` documents cleaning decisions, removes `Monthly_Salary` as a leakage feature, and builds both `full` and `conservative` feature sets.
3. `notebooks/03_model_training.ipynb` trains `RandomForestRegressor` and `LightGBMRegressor` models on both feature sets and saves the best model bundle.
4. `src/04_optimize_assignment.py` builds department counterfactual profiles, scores every employee-department pair, and solves the assignment problem with PuLP.
5. `notebooks/05_evaluation_comparison.ipynb` compares the optimal assignment against random, greedy, and current-assignment baselines.

## Key Findings

The dataset shows a strong leakage relationship between `Monthly_Salary` and `Performance_Score`, so salary is excluded from modeling. The target is only weakly predictable from the remaining features: the best model in the generated results is `LightGBM` on the `conservative` feature set, but its test `R^2` is still near zero. That means the optimization layer can still re-rank candidates, but the gain is limited by weak predictive signal.

The assignment step does improve the objective. On the saved 50-employee sample, the PuLP solution beats the current-assignment baseline by about 0.89 predicted-score points.

## Results

### Model Comparison

| feature_set | model | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| conservative | lightgbm | 1.2151 | 1.4174 | -0.0062 |
| conservative | random_forest | 1.2263 | 1.4244 | -0.0161 |
| full | lightgbm | 1.2156 | 1.4176 | -0.0065 |
| full | random_forest | 1.2227 | 1.4211 | -0.0115 |

### Assignment Comparison

| method | total_predicted_score | mean_predicted_score |
| --- | ---: | ---: |
| optimal_pulp | 150.4663 | 3.0093 |
| current_assignment | 149.5808 | 2.9916 |
| random | 149.1276 | 2.9826 |
| greedy | 148.9708 | 2.9794 |

## Caveats

`Monthly_Salary` is treated as leakage and removed from all model features.

The department-profile counterfactuals are an assumption, not observed ground truth. Each department is represented by median department-level feature values, which introduces selection bias because employees were not randomly assigned to departments in the original data.

The “true score” proxy in the assignment comparison is historical `Performance_Score` for the sampled employees, so it does not change across assignment methods. It is included only as a reference point.

## Outputs

Generated artifacts are saved in:

* `outputs/figures/`
* `outputs/tables/`
* `results/model_metrics.csv`
* `results/best_model_metadata.json`
* `results/best_model_feature_importances.csv`
* `results/assignment_comparison.csv`
* `results/optimal_assignment.csv`

The best model bundle is stored at `models/best_model_bundle.joblib`.

## Project Files

* [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)
* [notebooks/02_data_prep.ipynb](notebooks/02_data_prep.ipynb)
* [notebooks/03_model_training.ipynb](notebooks/03_model_training.ipynb)
* [src/04_optimize_assignment.py](src/04_optimize_assignment.py)
* [notebooks/05_evaluation_comparison.ipynb](notebooks/05_evaluation_comparison.ipynb)

## Run Order

1. Open and run `01_eda.ipynb`.
2. Run `02_data_prep.ipynb`.
3. Run `03_model_training.ipynb`.
4. Run `python src/04_optimize_assignment.py --sample-size 50`.
5. Run `05_evaluation_comparison.ipynb`.
