"""Optimize employee-to-department assignments with PuLP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smartassign_pipeline import (  # noqa: E402
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    ModelBundle,
    build_score_and_uncertainty_matrices,
    build_score_matrix,
    compare_assignment_methods,
    compute_department_capacities,
    compute_department_profiles,
    ensure_output_directories,
    get_counterfactual_profile_columns,
    load_raw_data,
    save_csv,
    save_json,
    solve_assignment_with_overrides,
    split_train_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=50, help="Number of employees to include in the optimization sample.")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "best_model_bundle.joblib", help="Path to the saved model bundle.")
    parser.add_argument("--output-path", type=Path, default=RESULTS_DIR / "proposed_assignment.csv", help="Where to save the pre-review assignment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_directories()

    df = load_raw_data()
    bundle = ModelBundle.load(args.model_path)

    train_x, test_x, train_y, test_y = split_train_test(df, bundle.feature_columns)
    test_frame = df.loc[test_x.index].copy().sample(n=min(args.sample_size, len(test_x)), random_state=42)

    profile_columns = get_counterfactual_profile_columns(bundle.feature_set_name, df)
    department_profiles = compute_department_profiles(df, profile_columns)
    department_profiles.to_csv(PROCESSED_DATA_DIR / f"department_profiles_{bundle.feature_set_name}.csv", index=False)

    score_matrix_df, uncertainty_matrix_df = build_score_and_uncertainty_matrices(
        bundle,
        test_frame,
        department_profiles,
        profile_columns,
    )

    capacities = compute_department_capacities(df, len(test_frame))
    comparison, methods = compare_assignment_methods(test_frame, score_matrix_df, capacities)
    proposed_assignment, proposed_total = solve_assignment_with_overrides(
        score_matrix=score_matrix_df,
        employees_ids=test_frame.index,
        projects_ids=score_matrix_df.columns,
        capacity=capacities,
    )

    save_csv(score_matrix_df.reset_index().rename(columns={"index": "employee_index"}), RESULTS_DIR / "score_matrix_sample.csv")
    save_csv(uncertainty_matrix_df.reset_index().rename(columns={"index": "employee_index"}), RESULTS_DIR / "uncertainty_matrix_sample.csv")
    save_csv(proposed_assignment, args.output_path)
    save_csv(comparison, RESULTS_DIR / "assignment_comparison.csv")
    save_json(
        {
            "sample_size": int(len(test_frame)),
            "feature_set": bundle.feature_set_name,
            "model_path": str(args.model_path),
            "department_capacities": capacities,
            "profile_columns": profile_columns,
            "proposed_total_predicted_score": proposed_total,
        },
        RESULTS_DIR / "optimization_metadata.json",
    )

    print("Proposed assignment saved to:", args.output_path)
    print("Comparison table saved to:", RESULTS_DIR / "assignment_comparison.csv")
    print("Objective score summary:\n", comparison)


if __name__ == "__main__":
    main()