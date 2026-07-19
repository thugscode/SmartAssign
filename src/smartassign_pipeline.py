"""Shared pipeline utilities for the SmartAssign project.

The module keeps the notebooks compact by centralizing data loading, cleaning,
model training, counterfactual score generation, and assignment optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import importlib

import joblib
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TARGET_COL = "Performance_Score"
ID_COL = "Employee_ID"
LEAKAGE_COLS = {"Monthly_Salary"}
DROP_FROM_FEATURES = {ID_COL, TARGET_COL, "Resigned", "Hire_Date"} | LEAKAGE_COLS
OUTCOME_LIKE_COLS = ["Projects_Handled", "Promotions", "Sick_Days"]
STATIC_FEATURES = ["Age", "Gender", "Education_Level", "Job_Title", "Department"]
NUMERIC_CONTEXT_COLS = [
    "Age",
    "Years_At_Company",
    "Work_Hours_Per_Week",
    "Overtime_Hours",
    "Remote_Work_Frequency",
    "Team_Size",
    "Training_Hours",
    "Employee_Satisfaction_Score",
]
FULL_NUMERIC_EXTRAS = OUTCOME_LIKE_COLS
CONSERVATIVE_NUMERIC_EXTRAS: list[str] = []
TARGET_ENCODING_THRESHOLD = 10
DEFAULT_RANDOM_STATE = 42


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Simple target mean encoder for high-cardinality categorical columns."""

    def __init__(self):
        self.maps_: Dict[str, Dict[object, float]] = {}
        self.global_mean_: float = 0.0
        self.feature_names_in_: list[str] = []

    def fit(self, X, y):
        frame = pd.DataFrame(X).copy()
        frame.columns = frame.columns.astype(str)
        target = pd.Series(y).reset_index(drop=True)
        frame = frame.reset_index(drop=True)
        self.global_mean_ = float(target.mean())
        self.feature_names_in_ = list(frame.columns)
        self.maps_ = {}
        for column in frame.columns:
            mapping = (
                pd.DataFrame({"value": frame[column], "target": target})
                .groupby("value", dropna=False)["target"]
                .mean()
                .to_dict()
            )
            self.maps_[column] = mapping
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        frame.columns = frame.columns.astype(str)
        encoded_columns = []
        for column in frame.columns:
            mapping = self.maps_.get(column, {})
            encoded_columns.append(frame[column].map(mapping).fillna(self.global_mean_).astype(float).to_numpy())
        if not encoded_columns:
            return np.empty((len(frame), 0), dtype=float)
        return np.vstack(encoded_columns).T

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array([f"{name}__target_mean" for name in input_features], dtype=object)


@dataclass
class ModelBundle:
    """Container for a fitted model pipeline and the metadata needed later."""

    pipeline: Pipeline
    feature_columns: List[str]
    categorical_columns: List[str]
    numeric_columns: List[str]
    feature_set_name: str
    metrics: Dict[str, float]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        return joblib.load(path)


def ensure_output_directories() -> None:
    for directory in [PROCESSED_DATA_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_data(filename: str = "Employee_Data.csv") -> pd.DataFrame:
    path = RAW_DATA_DIR / filename
    return pd.read_csv(path)


def basic_data_overview(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    overview = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct}).sort_values(
            "missing_count", ascending=False
        ),
    }
    return overview


def get_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    base_categoricals = ["Department", "Job_Title", "Gender", "Education_Level"]
    shared_numeric = [
        "Age",
        "Years_At_Company",
        "Work_Hours_Per_Week",
        "Overtime_Hours",
        "Remote_Work_Frequency",
        "Team_Size",
        "Training_Hours",
        "Employee_Satisfaction_Score",
    ]
    full_features = base_categoricals + shared_numeric + OUTCOME_LIKE_COLS
    conservative_features = base_categoricals + shared_numeric
    # Keep only columns that exist in the provided frame.
    full_features = [column for column in full_features if column in df.columns and column not in DROP_FROM_FEATURES]
    conservative_features = [column for column in conservative_features if column in df.columns and column not in DROP_FROM_FEATURES]
    return {"full": full_features, "conservative": conservative_features}


def get_counterfactual_profile_columns(feature_set_name: str, df: pd.DataFrame) -> List[str]:
    """Return the department profile columns that will be overwritten in counterfactual rows.

    The conservative feature set excludes the outcome-like cumulative metrics, while the
    full feature set keeps them in the department profile to reflect the modeling assumption.
    """

    base_columns = [
        "Years_At_Company",
        "Work_Hours_Per_Week",
        "Overtime_Hours",
        "Remote_Work_Frequency",
        "Team_Size",
        "Training_Hours",
        "Employee_Satisfaction_Score",
    ]
    if feature_set_name == "full":
        base_columns = base_columns + OUTCOME_LIKE_COLS
    return [column for column in base_columns if column in df.columns]


def split_train_test(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = TARGET_COL,
    stratify_column: str = "Department",
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = df.loc[:, list(feature_columns)].copy()
    target = df[target_column].copy()
    stratify = df[stratify_column] if stratify_column in df.columns and df[stratify_column].nunique() > 1 else None
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def get_feature_types(feature_frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_columns = [
        column for column in feature_frame.columns if feature_frame[column].dtype == object or str(feature_frame[column].dtype) == "category"
    ]
    numeric_columns = [column for column in feature_frame.columns if column not in categorical_columns]
    return categorical_columns, numeric_columns


def build_preprocessor(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    high_cardinality_threshold: int = TARGET_ENCODING_THRESHOLD,
) -> ColumnTransformer:
    categorical_columns, numeric_columns = get_feature_types(feature_frame)
    high_cardinality_columns = [column for column in categorical_columns if feature_frame[column].nunique(dropna=True) > high_cardinality_threshold]
    low_cardinality_columns = [column for column in categorical_columns if column not in high_cardinality_columns]

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns))
    if low_cardinality_columns:
        transformers.append(
            (
                "categorical_onehot",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                low_cardinality_columns,
            )
        )
    if high_cardinality_columns:
        transformers.append(
            (
                "categorical_target",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", TargetMeanEncoder()),
                    ]
                ),
                high_cardinality_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def build_model(model_name: str, random_state: int = DEFAULT_RANDOM_STATE):
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=250,
            random_state=random_state,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        )
    if model_name == "lightgbm":
        lgbm_module = importlib.import_module("lightgbm")
        return lgbm_module.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            random_state=random_state,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def fit_and_evaluate_model(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_y: pd.Series,
    test_y: pd.Series,
    model_name: str,
    feature_set_name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[ModelBundle, pd.DataFrame]:
    preprocessor = build_preprocessor(train_x, train_y)
    regressor = build_model(model_name, random_state=random_state)
    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
    pipeline.fit(train_x, train_y)

    predictions = pipeline.predict(test_x)
    metrics = {
        "feature_set": feature_set_name,
        "model": model_name,
        "mae": float(mean_absolute_error(test_y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(test_y, predictions))),
        "r2": float(r2_score(test_y, predictions)),
    }
    bundle = ModelBundle(
        pipeline=pipeline,
        feature_columns=list(train_x.columns),
        categorical_columns=get_feature_types(train_x)[0],
        numeric_columns=get_feature_types(train_x)[1],
        feature_set_name=feature_set_name,
        metrics=metrics,
    )
    metric_frame = pd.DataFrame([metrics])
    return bundle, metric_frame


def feature_importance_dataframe(bundle: ModelBundle, top_n: int = 20) -> pd.DataFrame:
    regressor = bundle.pipeline.named_steps["regressor"]
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out(bundle.feature_columns)
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    else:
        raise AttributeError("The fitted regressor does not expose feature_importances_.")
    frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    frame = frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return frame


def compute_department_profiles(
    df: pd.DataFrame,
    profile_columns: Sequence[str],
    department_column: str = "Department",
) -> pd.DataFrame:
    available_columns = [column for column in profile_columns if column in df.columns]
    profiles = df.groupby(department_column)[available_columns].median(numeric_only=True).reset_index()
    profiles["department_size"] = df.groupby(department_column).size().values
    return profiles


def build_counterfactual_frame(
    employee_row: pd.Series,
    department: str,
    department_profiles: pd.DataFrame,
    profile_columns: Sequence[str],
    feature_columns: Sequence[str],
    department_column: str = "Department",
) -> pd.DataFrame:
    frame = pd.DataFrame([employee_row.loc[list(feature_columns)]])
    frame.loc[:, department_column] = department
    profile_row = department_profiles.loc[department_profiles[department_column] == department]
    if not profile_row.empty:
        profile_row = profile_row.iloc[0]
        for column in profile_columns:
            if column in frame.columns and column in profile_row.index:
                frame.loc[:, column] = profile_row[column]
    return frame


def build_score_matrix(
    bundle: ModelBundle,
    sample_df: pd.DataFrame,
    department_profiles: pd.DataFrame,
    profile_columns: Sequence[str],
    department_column: str = "Department",
) -> pd.DataFrame:
    departments = department_profiles[department_column].tolist()
    score_matrix = pd.DataFrame(index=sample_df.index, columns=departments, dtype=float)
    for department in departments:
        counterfactual_rows = []
        for _, row in sample_df.iterrows():
            counterfactual_rows.append(
                build_counterfactual_frame(
                    employee_row=row,
                    department=department,
                    department_profiles=department_profiles,
                    profile_columns=profile_columns,
                    feature_columns=bundle.feature_columns,
                    department_column=department_column,
                )
            )
        counterfactual_frame = pd.concat(counterfactual_rows, ignore_index=True)
        score_matrix.loc[:, department] = bundle.pipeline.predict(counterfactual_frame)
    return score_matrix


def compute_department_capacities(
    full_df: pd.DataFrame,
    sample_size: int,
    department_column: str = "Department",
) -> Dict[str, int]:
    counts = full_df[department_column].value_counts(normalize=True).sort_index()
    raw_capacities = counts * sample_size
    capacities = raw_capacities.round().astype(int)
    capacity_total = int(capacities.sum())
    departments = capacities.index.tolist()

    if capacity_total != sample_size:
        remainders = (raw_capacities - raw_capacities.round()).sort_values(ascending=False)
        if capacity_total < sample_size:
            gap = sample_size - capacity_total
            for department in remainders.index.tolist():
                if gap <= 0:
                    break
                capacities.loc[department] += 1
                gap -= 1
        else:
            gap = capacity_total - sample_size
            for department in remainders.index[::-1].tolist():
                if gap <= 0:
                    break
                if capacities.loc[department] > 0:
                    capacities.loc[department] -= 1
                    gap -= 1

    if int(capacities.sum()) != sample_size:
        adjustment = sample_size - int(capacities.sum())
        first_department = capacities.index[0]
        capacities.loc[first_department] += adjustment

    return {department: int(capacities.loc[department]) for department in departments}


def solve_assignment_ortools(
    score_matrix: pd.DataFrame,
    capacities: Dict[str, int],
    time_limit_seconds: int = 30,
) -> Tuple[pd.DataFrame, float]:
    model = cp_model.CpModel()
    employees = list(score_matrix.index)
    departments = list(score_matrix.columns)
    scale = 1000

    variables = {}
    for employee in employees:
        for department in departments:
            variables[(employee, department)] = model.NewBoolVar(f"x_{employee}_{department}")

    for employee in employees:
        model.Add(sum(variables[(employee, department)] for department in departments) == 1)

    for department in departments:
        model.Add(
            sum(variables[(employee, department)] for employee in employees) <= int(capacities.get(department, 0))
        )

    objective_terms = []
    for employee in employees:
        for department in departments:
            weight = int(round(float(score_matrix.loc[employee, department]) * scale))
            objective_terms.append(weight * variables[(employee, department)])
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"OR-Tools did not find a feasible assignment. Status={status}")

    records = []
    for employee in employees:
        for department in departments:
            if solver.Value(variables[(employee, department)]) == 1:
                records.append(
                    {
                        "employee_index": employee,
                        "department": department,
                        "predicted_score": float(score_matrix.loc[employee, department]),
                    }
                )
                break

    assignment = pd.DataFrame(records)
    total_score = float(assignment["predicted_score"].sum())
    return assignment, total_score


def random_assignment(
    score_matrix: pd.DataFrame,
    capacities: Dict[str, int],
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    departments = []
    for department, capacity in capacities.items():
        departments.extend([department] * int(capacity))
    departments = np.array(departments, dtype=object)
    rng.shuffle(departments)
    if len(departments) != len(score_matrix.index):
        raise ValueError("Department capacities must sum to the number of employees.")
    employees = np.array(score_matrix.index)
    rng.shuffle(employees)
    records = []
    for employee, department in zip(employees, departments):
        records.append(
            {
                "employee_index": employee,
                "department": department,
                "predicted_score": float(score_matrix.loc[employee, department]),
            }
        )
    return pd.DataFrame(records)


def greedy_assignment(
    sample_df: pd.DataFrame,
    score_matrix: pd.DataFrame,
    capacities: Dict[str, int],
    department_column: str = "Department",
    target_column: str = TARGET_COL,
) -> pd.DataFrame:
    department_need = sample_df.groupby(department_column)[target_column].mean().sort_values(ascending=True)
    departments_by_need = department_need.index.tolist()
    for department in score_matrix.columns:
        if department not in departments_by_need:
            departments_by_need.append(department)

    employees_sorted = sample_df.sort_values(target_column, ascending=False).index.tolist()
    department_slots = []
    for department in departments_by_need:
        department_slots.extend([department] * int(capacities.get(department, 0)))
    if len(department_slots) != len(employees_sorted):
        raise ValueError("Department capacities must sum to the number of employees.")

    records = []
    for employee, department in zip(employees_sorted, department_slots):
        records.append(
            {
                "employee_index": employee,
                "department": department,
                "predicted_score": float(score_matrix.loc[employee, department]),
            }
        )
    return pd.DataFrame(records)


def current_assignment(sample_df: pd.DataFrame, score_matrix: pd.DataFrame, department_column: str = "Department") -> pd.DataFrame:
    records = []
    for employee_index, row in sample_df.iterrows():
        department = row[department_column]
        records.append(
            {
                "employee_index": employee_index,
                "department": department,
                "predicted_score": float(score_matrix.loc[employee_index, department]),
            }
        )
    return pd.DataFrame(records)


def compare_assignment_methods(
    sample_df: pd.DataFrame,
    score_matrix: pd.DataFrame,
    capacities: Dict[str, int],
    department_column: str = "Department",
    target_column: str = TARGET_COL,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    methods = {
        "optimal_ortools": None,
        "random": None,
        "greedy": None,
        "current_assignment": None,
    }

    optimal_assignment, optimal_total = solve_assignment_ortools(score_matrix, capacities)
    methods["optimal_ortools"] = optimal_assignment
    methods["random"] = random_assignment(score_matrix, capacities, random_state=random_state)
    methods["greedy"] = greedy_assignment(sample_df, score_matrix, capacities, department_column=department_column, target_column=target_column)
    methods["current_assignment"] = current_assignment(sample_df, score_matrix, department_column=department_column)

    proxy_true_total = float(sample_df[target_column].sum())
    comparison_rows = []
    for method_name, assignment in methods.items():
        comparison_rows.append(
            {
                "method": method_name,
                "total_predicted_score": float(assignment["predicted_score"].sum()),
                "mean_predicted_score": float(assignment["predicted_score"].mean()),
                "proxy_true_total_score": proxy_true_total,
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("total_predicted_score", ascending=False).reset_index(drop=True)
    return comparison, methods


def summarize_metrics_table(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    return metrics_frame.sort_values(["feature_set", "rmse", "mae"]).reset_index(drop=True)


def save_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
