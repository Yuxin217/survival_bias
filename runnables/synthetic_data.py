import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.synthetic_data_generation import SyntheticDataModule
from src.exp_utils import (
    run_censoring_strength_rct,
    rmse_from_csv_gamma,
    rmse_from_csv_non_informative,
)
from src.plot_utils import plot_censoring_strength_rmse

# ------------------------------------------------------------------ #
#  Config                                                             #
# ------------------------------------------------------------------ #

RESPONSE_TYPES          = ["exp", "sin", "logistic_sin"]
CENSORING_STRENGTH_LIST = [0.2, 0.4, 0.6]
SEED_LIST               = [22, 23, 24, 25, 26]


# ------------------------------------------------------------------ #
#  Step 1: Synthetic data generation                                  #
# ------------------------------------------------------------------ #

def generate_data(seed_list=SEED_LIST, censoring_strength_array=CENSORING_STRENGTH_LIST, response_types=RESPONSE_TYPES):
    for seed in seed_list:
        for censoring_strength in censoring_strength_array:
            data_gen = SyntheticDataModule(
                n_train=2000, censoring_strength=censoring_strength,
                x_dim=1, seed=int(42 + 10 * censoring_strength + seed)
            )
            for response_type in response_types:
                df = data_gen._generate_data(response_type=response_type, target_value=censoring_strength)
                output_dir = f"{project_root}/data/synthetic_data/seed_{seed}"
                os.makedirs(output_dir, exist_ok=True)
                df.to_csv(f"{output_dir}/{response_type}_function_{censoring_strength}.csv", index=False)
        print(f"Generated data for seed {seed}")


# ------------------------------------------------------------------ #
#  Step 2: Run SurvB-learner                                         #
# ------------------------------------------------------------------ #

def run_rct(seed_list=SEED_LIST, censoring_strength_array=CENSORING_STRENGTH_LIST, response_types=RESPONSE_TYPES):
    for seed in seed_list:
        root_data_path = f"{project_root}/data/synthetic_data/seed_{seed}"
        output_path    = f"{project_root}/results/synthetic_data/rct/seed_{seed}"

        for response_idx, response_type in enumerate(response_types):
            print(f"Processing seed={seed} | response_type={response_type}")
            pbar = tqdm(total=len(censoring_strength_array), desc="Progress", leave=False)

            for censoring_strength in censoring_strength_array:
                df = pd.read_csv(f"{root_data_path}/{response_type}_function_{censoring_strength}.csv")
                gamma_df, non_informative_df = run_censoring_strength_rct(df, response_type=response_type, censoring_strength=censoring_strength)

                gamma_dir = f"{output_path}/gamma"
                os.makedirs(gamma_dir, exist_ok=True)
                gamma_df.to_csv(f"{gamma_dir}/{response_type}_function_gamma_{censoring_strength}.csv", index=False)

                ni_dir = f"{output_path}/non_informative"
                os.makedirs(ni_dir, exist_ok=True)
                non_informative_df.to_csv(f"{ni_dir}/{response_type}_function_non_informative_{censoring_strength}.csv", index=False)

                pbar.update(1)
            pbar.close()


# ------------------------------------------------------------------ #
#  Step 3: Table A + Figure B                                        #
# ------------------------------------------------------------------ #

def report_rct(seed_list=SEED_LIST, response_types=RESPONSE_TYPES):
    root_path = f"{project_root}/results/synthetic_data/rct"
    print("=" * 60)
    print("RCT RESULTS SUMMARY")
    print("=" * 60)
    for response_type in response_types:
        rmse_from_csv_gamma(root_path, response_type, seed_list)
    for response_type in response_types:
        rmse_from_csv_non_informative(root_path, response_type, seed_list)


def plot_rct(seed_list=SEED_LIST):
    root_path   = f"{project_root}/results/synthetic_data/rct"
    output_path = f"{project_root}/figures/synthetic_data/rct"
    plot_censoring_strength_rmse(root_path, censoring_strength_list=[0.2], seed_list=seed_list, output_path=output_path)





# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # Step 1: Generate data
    generate_data()

    # Step 2: Run SurvB-learner
    run_rct()

    # Step 3: Report and plot
    report_rct()
    plot_rct()
