import os
import sys

import numpy as np
import pandas as pd

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from src.dr_utils import dr_learner

tmax_dict = {"exp": 200, "sin": 60, "logistic_sin": 80}

def _compute_bias_metrics(dr_df, n_bins=12):
    err = dr_df["DR_est"] - dr_df["True_CATE"]
    global_bias = float(np.mean(err))
    global_abs_bias = float(np.mean(np.abs(err)))
    global_rmse = float(np.sqrt(np.mean(err**2)))

    pointwise = dr_df[["X"]].copy()
    pointwise["bias"] = err
    pointwise["X_bin"] = pd.qcut(pointwise["X"], q=n_bins, duplicates="drop")
    pointwise_bias_df = (
        pointwise.groupby("X_bin", observed=False)
        .agg(
            X_mean=("X", "mean"),
            pointwise_bias=("bias", "mean"),
            pointwise_abs_bias=("bias", lambda s: np.mean(np.abs(s))),
            n=("bias", "size"),
        )
        .reset_index(drop=True)
        .sort_values("X_mean")
    )
    summary_df = pd.DataFrame(
        [{
            "global_bias": global_bias,
            "global_abs_bias": global_abs_bias,
            "global_rmse": global_rmse,
            "n_bins_used": int(pointwise_bias_df.shape[0]),
            "n_samples": int(dr_df.shape[0]),
        }]
    )
    return summary_df, pointwise_bias_df


def run_censoring_strength_exp(root_data_path, response_type, censoring_strength):

    # -------------------- Load data --------------------

    data_path = os.path.join(root_data_path, f"{response_type}_function_{censoring_strength}.csv")
    df = pd.read_csv(data_path)
    X = df[['X']].to_numpy()
    A = df['A'].to_numpy()
    y = df['y'].to_numpy()
    Delta = df['Delta'].to_numpy()
    ground_truth = df['y1_minus_y0'].to_numpy()

    # -------------------- Fit DR-learner --------------------

    dr_est = dr_learner(X, A, y, Delta, t_max=tmax_dict[response_type], nuisance_type="lgb")

    result_df = pd.DataFrame({
        'X': X.flatten(),
        'A': A,
        'Delta': Delta,
        'True_CATE': ground_truth,
        'DR_est': dr_est,
    })
    bias_summary_df, pointwise_bias_df = _compute_bias_metrics(result_df)

    return result_df, bias_summary_df, pointwise_bias_df

if __name__ == "__main__":
    censoring_strength_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    for response_type in ["sin", "logistic_sin", "exp"]:
        for seed in [21, 22, 23, 24, 25, 26, 27]:
            for censoring_strength in censoring_strength_list:
                root_data_path = os.path.join(CODE_DIR, "data", "synthetic_data", "rct", f"seed_{seed}")
                output_dir = os.path.join(CODE_DIR, "results", "synthetic_data", "rct", f"seed_{seed}", "dr_est")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                dr_df, bias_summary_df, pointwise_bias_df = run_censoring_strength_exp(
                    root_data_path, response_type, censoring_strength
                )
                dr_df.to_csv(os.path.join(output_dir, f"{response_type}_function_{censoring_strength}.csv"), index=False)
                bias_summary_df.to_csv(
                    os.path.join(output_dir, f"{response_type}_function_{censoring_strength}_bias_summary.csv"),
                    index=False,
                )
                pointwise_bias_df.to_csv(
                    os.path.join(output_dir, f"{response_type}_function_{censoring_strength}_pointwise_bias.csv"),
                    index=False,
                )
                print(
                    f"[{response_type}][seed={seed}][censoring strength = {censoring_strength}] "
                    f"global_bias={bias_summary_df['global_bias'].iloc[0]:.4f}, "
                    f"global_abs_bias={bias_summary_df['global_abs_bias'].iloc[0]:.4f}"
                )
                print(f'Saved DR estimates for seed {seed}')
