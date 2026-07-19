# SmartAssign

This project takes employee data, predicts performance, and then uses those predictions to assign people to departments in a smarter way.

## What This Project Does

The idea is simple:

1. Look at the employee dataset.
2. Clean the data and remove anything that would give the model unfair help.
3. Train a model to predict `Performance_Score`.
4. Use those predictions to assign employees to departments.
5. Compare that assignment against simpler baselines.

Departments are treated like project slots here, so the model is basically asking: “Which employee should go to which department if we want the best overall predicted performance?”

## How The Project Is Built

The work is split into a few notebook steps and one script:

1. `notebooks/01_eda.ipynb` looks at the raw data, checks missing values, and shows the main patterns.
2. `notebooks/02_data_prep.ipynb` cleans the data, removes `Monthly_Salary` because it leaks information, and creates two feature sets: `full` and `conservative`.
3. `notebooks/03_model_training.ipynb` trains a `RandomForestRegressor` and a `LightGBMRegressor`, then saves the best model.
4. `src/04_optimize_assignment.py` builds department profiles, predicts scores for employee-department pairs, and solves the assignment problem with PuLP.
5. `notebooks/05_evaluation_comparison.ipynb` compares the optimized assignment with random, greedy, and current-assignment baselines.

## Math Behind The Project

This section shows the main formulas in the project in a simple way.

### 1. Predicting Performance

The model tries to learn a function:

$$
\hat{y}_i = f(x_i)
$$

where $x_i$ is the feature vector for employee $i$ and $\hat{y}_i$ is the predicted `Performance_Score`.

For the two feature sets:

$$
X_{full} = X_{base} \cup \{Projects\_Handled, Promotions, Sick\_Days\}
$$

$$
X_{conservative} = X_{base}
$$

`Monthly_Salary` is removed because it leaks information about the target.

### 2. Missing Value Handling

Numeric columns are filled with the median:

$$
x_{filled} = \mathrm{median}(x)
$$

Categorical columns are filled with the most common value:

$$
x_{filled} = \mathrm{mode}(x)
$$

### 3. One-Hot Encoding

For a categorical value, one-hot encoding turns it into a binary vector. If a category has $k$ possible values, then:

$$
z_j =
\begin{cases}
1 & \text{if the row belongs to category } j \\
0 & \text{otherwise}
\end{cases}
$$

### 4. Department Profiles

Because we do not have true counterfactual data, each department is represented by the median of the employees already in that department:

$$
p_d = \mathrm{median}(X_{\{i: dept_i = d\}})
$$

This is an assumption, not ground truth.

### 5. Score Matrix

For each sampled employee $i$ and department $d$, the project builds a counterfactual feature vector and predicts a score:

$$
s_{i,d} = f(x_i^{(d)})
$$

The matrix $S = [s_{i,d}]$ is the employee-by-department score matrix.

### 6. Optimization Problem

We define a binary decision variable:

$$
x_{i,d} =
\begin{cases}
1 & \text{if employee } i \text{ is assigned to department } d \\
0 & \text{otherwise}
\end{cases}
$$

The goal is to maximize the total predicted score:

$$
\max \sum_i \sum_d s_{i,d} x_{i,d}
$$

Subject to:

$$
\sum_d x_{i,d} = 1 \quad \forall i
$$

Each employee must go to exactly one department.

$$
\sum_i x_{i,d} \le c_d \quad \forall d
$$

Each department has a capacity $c_d$.

The capacity is proportional to the department's share in the full dataset:

$$
c_d \approx \text{round}\left(\frac{n_d}{N} \cdot n_{sample}\right)
$$

where $n_d$ is the current department size, $N$ is the full dataset size, and $n_{sample}$ is the number of sampled employees.

### 7. Evaluation Metrics

The model is judged using MAE, RMSE, and $R^2$.

Mean Absolute Error:

$$
MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

Root Mean Squared Error:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

Coefficient of determination:

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

### 8. Baselines

The project compares the optimized solution against three simpler baselines:

* Random assignment
* Greedy assignment
* Current assignment

The main comparison value is the total predicted score:

$$
T = \sum_i s_{i,a(i)}
$$

where $a(i)$ is the assigned department for employee $i$.

## What I Found

`Monthly_Salary` is strongly tied to `Performance_Score`, so I removed it from the model to avoid leakage.

The remaining features do not predict performance very strongly. The best result in the saved run is LightGBM on the conservative feature set, but the test `R^2` is still close to zero. So the model is useful, but it is not especially strong on its own.

Even so, the optimization step still helps. On the saved 50-employee sample, the PuLP assignment gives a slightly better total predicted score than the current assignment baseline.

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

## Important Notes

`Monthly_Salary` is treated as leakage and is not used in the model.

The department profiles used for the counterfactual assignment are an assumption, not real observed truth. Each department is represented by median values from the employees already in that department, which means the setup has some selection bias.

The “true score” shown in the assignment comparison is only a historical proxy based on the sampled employees’ actual `Performance_Score`. It stays the same across methods because the employees themselves do not change.

## Output Files

The project saves figures and tables in these folders:

* `outputs/figures/`
* `outputs/tables/`

It also saves result files here:

* `results/model_metrics.csv`
* `results/best_model_metadata.json`
* `results/best_model_feature_importances.csv`
* `results/assignment_comparison.csv`
* `results/optimal_assignment.csv`

The trained model bundle is saved at `models/best_model_bundle.joblib`.

## Install

Install the needed packages with:

```bash
pip install -r requirements.txt
```

## Files To Run

* [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)
* [notebooks/02_data_prep.ipynb](notebooks/02_data_prep.ipynb)
* [notebooks/03_model_training.ipynb](notebooks/03_model_training.ipynb)
* [src/04_optimize_assignment.py](src/04_optimize_assignment.py)
* [notebooks/05_evaluation_comparison.ipynb](notebooks/05_evaluation_comparison.ipynb)

## Run Order

1. Run `01_eda.ipynb`.
2. Run `02_data_prep.ipynb`.
3. Run `03_model_training.ipynb`.
4. Run `python src/04_optimize_assignment.py --sample-size 50`.
5. Run `05_evaluation_comparison.ipynb`.
