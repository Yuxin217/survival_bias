import pandas as pd
import numpy as np
from src.run_utils import survb_bounds, oracle_bounds, plug_in_bounds, survb_bounds_oracle
from src.load_utils import *

gamma_dict = {"sin": 30, "exp": 50, "logistic_sin": 30}
tmax_dict={"exp": 200, "sin": 60, "logistic_sin": 80}

def run_censoring_strength_exp(root_data_path, response_type, censoring_strength):

    # -------------------- Load data --------------------

    data_path = f"{root_data_path}/{response_type}_function"

    df = pd.read_csv(f"{data_path}_{str(censoring_strength)}.csv")
    X = df[['X']].to_numpy()
    A = df['A'].to_numpy()
    y = df['y'].to_numpy()
    Delta = df['Delta'].to_numpy()
    ground_truth = df['y1_minus_y0'].to_numpy()

    # -------------------- Gamma experiment --------------------
    # print("Gamma experiment")

    plug_in_est = plug_in_bounds(X, A, y, Delta, "gamma", gamma_dict[response_type])
    oracle_est = oracle_bounds(X, A, y, Delta, "gamma", gamma_dict[response_type], response_type)
    survB_est = survb_bounds(X, A, y, Delta, "gamma", gamma_dict[response_type])

    # print("Finished censoring strength: ", censoring_strength)

    gamma_df = pd.DataFrame({'X':X.flatten(), 
                       'A':A,
                       'Delta':Delta,
                       'True_CATE': ground_truth,
                       'SurvB_lb': survB_est[-2],
                       'SurvB_ub': survB_est[-1],
                       'Plug_in_lb': plug_in_est[-2],
                       'Plug_in_ub': plug_in_est[-1],
                       'Oracle_lb': oracle_est[-2],
                       'Oracle_ub': oracle_est[-1]})

    # -------------------- Non-informative experiment --------------------

    # print("Non-informtiave experiment")
    survB_est = survb_bounds(X, A, y, Delta, "non_informative", tmax_dict[response_type])
    plug_in_est = plug_in_bounds(X, A, y, Delta, "non_informative", tmax_dict[response_type])
    oracle_est = oracle_bounds(X, A, y, Delta, "non_informative", tmax_dict[response_type], response_type)

    # print("Finished censoring strength: ", censoring_strength)

    non_informative_df = pd.DataFrame({'X':X.flatten(), 
                       'A':A,
                       'Delta':Delta,
                       'True_CATE': ground_truth,
                       'SurvB_lb': survB_est[-2],
                       'SurvB_ub': survB_est[-1],
                       'Plug_in_lb': plug_in_est[-2],
                       'Plug_in_ub': plug_in_est[-1],
                       'Oracle_lb': oracle_est[-2],
                       'Oracle_ub': oracle_est[-1]})
    
    return gamma_df, non_informative_df

def run_censoring_strength_rct(df, response_type, censoring_strength):

    # -------------------- Load data --------------------

    X = df[['X']].to_numpy()
    A = df['A'].to_numpy()
    y = df['y'].to_numpy()
    Delta = df['Delta'].to_numpy()
    ground_truth = df['y1_minus_y0'].to_numpy()

    # -------------------- Gamma experiment --------------------
    # print("Gamma experiment")

    plug_in_est = plug_in_bounds(X, A, y, Delta, "gamma", gamma_dict[response_type])
    oracle_est = oracle_bounds(X, A, y, Delta, "gamma", gamma_dict[response_type], response_type)
    survB_est = survb_bounds_oracle(X, A, y, Delta, "gamma", gamma_dict[response_type])

    # print("Finished censoring strength: ", censoring_strength)

    gamma_df = pd.DataFrame({'X':X.flatten(), 
                       'A':A,
                       'Delta':Delta,
                       'True_CATE': ground_truth,
                       'SurvB_lb': survB_est[-2],
                       'SurvB_ub': survB_est[-1],
                       'Plug_in_lb': plug_in_est[-2],
                       'Plug_in_ub': plug_in_est[-1],
                       'Oracle_lb': oracle_est[-2],
                       'Oracle_ub': oracle_est[-1]})

    # -------------------- Non-informative experiment --------------------

    # print("Non-informtiave experiment")
    survB_est = survb_bounds_oracle(X, A, y, Delta, "non_informative", tmax_dict[response_type])
    plug_in_est = plug_in_bounds(X, A, y, Delta, "non_informative", tmax_dict[response_type])
    oracle_est = oracle_bounds(X, A, y, Delta, "non_informative", tmax_dict[response_type], response_type)

    # print("Finished censoring strength: ", censoring_strength)

    non_informative_df = pd.DataFrame({'X':X.flatten(), 
                       'A':A,
                       'Delta':Delta,
                       'True_CATE': ground_truth,
                       'SurvB_lb': survB_est[-2],
                       'SurvB_ub': survB_est[-1],
                       'Plug_in_lb': plug_in_est[-2],
                       'Plug_in_ub': plug_in_est[-1],
                       'Oracle_lb': oracle_est[-2],
                       'Oracle_ub': oracle_est[-1]})
    
    return gamma_df, non_informative_df

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def rmse_from_csv_gamma(root_path, response_type, seed_list):
    """
    Calculate RMSE for gamma experiments and return results in table format
    """
    results = {}
    
    for censoring_strength in [0.2, 0.4, 0.6]:
        survB_rmse = []
        plug_in_rmse = []
        for seed in seed_list:
            file_path = f"{root_path}/seed_{str(seed)}/gamma/{response_type}_function_gamma_{str(censoring_strength)}.csv"
            df = pd.read_csv(file_path)
            df = results_preprocess(df)
            
            survB_rmse.append(rmse(df['Oracle_lb_mean'], df['SurvB_lb_mean']))
            survB_rmse.append(rmse(df['Oracle_ub_mean'], df['SurvB_ub_mean']))
            plug_in_rmse.append(rmse(df['Oracle_lb_mean'], df['Plug_in_lb_mean']))
            plug_in_rmse.append(rmse(df['Oracle_ub_mean'], df['Plug_in_ub_mean']))

        results[censoring_strength] = {
            'SurvB': {'mean': np.mean(survB_rmse), 'std': np.std(survB_rmse)},
            'Plug_in': {'mean': np.mean(plug_in_rmse), 'std': np.std(plug_in_rmse)}
        }
    
    # Create table
    print(f"\n=== {response_type.upper()} FUNCTION RESULTS (Domain knowledge)===")
    print("Sample Size   | 0.2                    | 0.4                    | 0.6")
    print("-" * 70)
    
    # First row: Plug_in
    plug_in_02 = f"{results[0.2]['Plug_in']['mean']:.3f} ± {results[0.2]['Plug_in']['std']:.3f}"
    plug_in_04 = f"{results[0.4]['Plug_in']['mean']:.3f} ± {results[0.4]['Plug_in']['std']:.3f}"
    plug_in_06 = f"{results[0.6]['Plug_in']['mean']:.3f} ± {results[0.6]['Plug_in']['std']:.3f}"
    print(f"Plug_in       | {plug_in_02:<20} | {plug_in_04:<20} | {plug_in_06}")
    
    # Second row: SurvB
    survB_02 = f"{results[0.2]['SurvB']['mean']:.3f} ± {results[0.2]['SurvB']['std']:.3f}"
    survB_04 = f"{results[0.4]['SurvB']['mean']:.3f} ± {results[0.4]['SurvB']['std']:.3f}"
    survB_06 = f"{results[0.6]['SurvB']['mean']:.3f} ± {results[0.6]['SurvB']['std']:.3f}"
    print(f"SurvB         | {survB_02:<20} | {survB_04:<20} | {survB_06}")
    
    return results


def rmse_from_csv_non_informative(root_path, response_type, seed_list):
    """
    Calculate RMSE for non-informative experiments and return results in table format
    """
    results = {}
    
    for censoring_strength in [0.2, 0.4, 0.6]:
        survB_rmse = []
        plug_in_rmse = []
        for seed in seed_list:
            file_path = f"{root_path}/seed_{str(seed)}/non_informative/{response_type}_function_non_informative_{str(censoring_strength)}.csv"
            df = pd.read_csv(file_path)
            df = results_preprocess(df)
            
            survB_rmse.append(rmse(df['Oracle_lb_mean'], df['SurvB_lb_mean']))
            survB_rmse.append(rmse(df['Oracle_ub_mean'], df['SurvB_ub_mean']))
            plug_in_rmse.append(rmse(df['Oracle_lb_mean'], df['Plug_in_lb_mean']))
            plug_in_rmse.append(rmse(df['Oracle_ub_mean'], df['Plug_in_ub_mean']))

        results[censoring_strength] = {
            'SurvB': {'mean': np.mean(survB_rmse), 'std': np.std(survB_rmse)},
            'Plug_in': {'mean': np.mean(plug_in_rmse), 'std': np.std(plug_in_rmse)}
        }
    
    # Create table
    print(f"\n=== {response_type.upper()} FUNCTION RESULTS (Non-informative) ===")
    print("Method        | 0.2                    | 0.4                    | 0.6")
    print("-" * 70)
    
    # First row: Plug_in
    plug_in_02 = f"{results[0.2]['Plug_in']['mean']:.3f} ± {results[0.2]['Plug_in']['std']:.3f}"
    plug_in_04 = f"{results[0.4]['Plug_in']['mean']:.3f} ± {results[0.4]['Plug_in']['std']:.3f}"
    plug_in_06 = f"{results[0.6]['Plug_in']['mean']:.3f} ± {results[0.6]['Plug_in']['std']:.3f}"
    print(f"Plug_in       | {plug_in_02:<20} | {plug_in_04:<20} | {plug_in_06}")
    
    # Second row: SurvB
    survB_02 = f"{results[0.2]['SurvB']['mean']:.3f} ± {results[0.2]['SurvB']['std']:.3f}"
    survB_04 = f"{results[0.4]['SurvB']['mean']:.3f} ± {results[0.4]['SurvB']['std']:.3f}"
    survB_06 = f"{results[0.6]['SurvB']['mean']:.3f} ± {results[0.6]['SurvB']['std']:.3f}"
    print(f"SurvB         | {survB_02:<20} | {survB_04:<20} | {survB_06}")
    
    return results

def rmse_from_csv_gamma_sample_size(root_path, response_type, seed_list):
    """
    Calculate RMSE for non-informative experiments and return results in table format
    """
    results = {}
    
    for sample_size in [100, 200, 500, 1000, 2000, 5000, 10000]:
        survB_rmse = []
        plug_in_rmse = []
        for seed in seed_list:
            file_path = f"{root_path}/seed_{str(seed)}/gamma/{response_type}_function_gamma_{str(sample_size)}.csv"
            df = pd.read_csv(file_path)
            df = results_preprocess(df)
            
            survB_rmse.append(rmse(df['Oracle_lb_mean'], df['SurvB_lb_mean']))
            survB_rmse.append(rmse(df['Oracle_ub_mean'], df['SurvB_ub_mean']))
            plug_in_rmse.append(rmse(df['Oracle_lb_mean'], df['Plug_in_lb_mean']))
            plug_in_rmse.append(rmse(df['Oracle_ub_mean'], df['Plug_in_ub_mean']))

        results[sample_size] = {
            'SurvB': {'mean': np.mean(survB_rmse), 'std': np.std(survB_rmse)},
            'Plug_in': {'mean': np.mean(plug_in_rmse), 'std': np.std(plug_in_rmse)}
        }
    
    # Create LaTeX table
    print(f"\n=== {response_type.upper()} FUNCTION RESULTS (Domain knowledge) ===")
    print()

    # LaTeX table format
    
    # Plug_in row
    plug_in_100 = f"{results[100]['Plug_in']['mean']:.3f} \\pm {results[100]['Plug_in']['std']:.3f}"
    plug_in_200 = f"{results[200]['Plug_in']['mean']:.3f} \\pm {results[200]['Plug_in']['std']:.3f}"
    plug_in_500 = f"{results[500]['Plug_in']['mean']:.3f} \\pm {results[500]['Plug_in']['std']:.3f}"
    plug_in_1000 = f"{results[1000]['Plug_in']['mean']:.3f} \\pm {results[1000]['Plug_in']['std']:.3f}"
    plug_in_2000 = f"{results[2000]['Plug_in']['mean']:.3f} \\pm {results[2000]['Plug_in']['std']:.3f}"
    plug_in_5000 = f"{results[5000]['Plug_in']['mean']:.3f} \\pm {results[5000]['Plug_in']['std']:.3f}"
    plug_in_10000 = f"{results[10000]['Plug_in']['mean']:.3f} \\pm {results[10000]['Plug_in']['std']:.3f}"
    print(f"& Plug-in learner & ${plug_in_100}$ & ${plug_in_200}$ & ${plug_in_500}$ & ${plug_in_1000}$ & ${plug_in_2000}$ & ${plug_in_5000}$ & ${plug_in_10000}$ \\\\")
    
    # SurvB row
    survB_100 = f"{results[100]['SurvB']['mean']:.3f} \\pm {results[100]['SurvB']['std']:.3f}"
    survB_200 = f"{results[200]['SurvB']['mean']:.3f} \\pm {results[200]['SurvB']['std']:.3f}"
    survB_500 = f"{results[500]['SurvB']['mean']:.3f} \\pm {results[500]['SurvB']['std']:.3f}"
    survB_1000 = f"{results[1000]['SurvB']['mean']:.3f} \\pm {results[1000]['SurvB']['std']:.3f}"
    survB_2000 = f"{results[2000]['SurvB']['mean']:.3f} \\pm {results[2000]['SurvB']['std']:.3f}"
    survB_5000 = f"{results[5000]['SurvB']['mean']:.3f} \\pm {results[5000]['SurvB']['std']:.3f}"
    survB_10000 = f"{results[10000]['SurvB']['mean']:.3f} \\pm {results[10000]['SurvB']['std']:.3f}"
    print(f"& \\method (\\emph{{ours}}) & \\textbf{{${survB_100}$}} & \\textbf{{${survB_200}$}} & \\textbf{{${survB_500}$}} & \\textbf{{${survB_1000}$}} & \\textbf{{${survB_2000}$}} & \\textbf{{${survB_5000}$}} & \\textbf{{${survB_10000}$}} \\\\")
    
    
    return results

def rmse_from_csv_non_informative_sample_size(root_path, response_type, seed_list):
    """
    Calculate RMSE for non-informative experiments and return results in table format
    """
    results = {}
    
    for sample_size in [100, 200, 500, 1000, 2000, 5000, 10000]:
        survB_rmse = []
        plug_in_rmse = []
        for seed in seed_list:
            file_path = f"{root_path}/seed_{str(seed)}/non_informative/{response_type}_function_non_informative_{str(sample_size)}.csv"
            df = pd.read_csv(file_path)
            df = results_preprocess(df)
            
            survB_rmse.append(rmse(df['Oracle_lb_mean'], df['SurvB_lb_mean']))
            survB_rmse.append(rmse(df['Oracle_ub_mean'], df['SurvB_ub_mean']))
            plug_in_rmse.append(rmse(df['Oracle_lb_mean'], df['Plug_in_lb_mean']))
            plug_in_rmse.append(rmse(df['Oracle_ub_mean'], df['Plug_in_ub_mean']))

        results[sample_size] = {
            'SurvB': {'mean': np.mean(survB_rmse), 'std': np.std(survB_rmse)},
            'Plug_in': {'mean': np.mean(plug_in_rmse), 'std': np.std(plug_in_rmse)}
        }
    
    # Create LaTeX table
    print(f"\n=== {response_type.upper()} FUNCTION RESULTS (Non-informative) ===")
    print()
    
    # LaTeX table format
    
    # Plug_in row
    plug_in_100 = f"{results[100]['Plug_in']['mean']:.3f} \\pm {results[100]['Plug_in']['std']:.3f}"
    plug_in_200 = f"{results[200]['Plug_in']['mean']:.3f} \\pm {results[200]['Plug_in']['std']:.3f}"
    plug_in_500 = f"{results[500]['Plug_in']['mean']:.3f} \\pm {results[500]['Plug_in']['std']:.3f}"
    plug_in_1000 = f"{results[1000]['Plug_in']['mean']:.3f} \\pm {results[1000]['Plug_in']['std']:.3f}"
    plug_in_2000 = f"{results[2000]['Plug_in']['mean']:.3f} \\pm {results[2000]['Plug_in']['std']:.3f}"
    plug_in_5000 = f"{results[5000]['Plug_in']['mean']:.3f} \\pm {results[5000]['Plug_in']['std']:.3f}"
    plug_in_10000 = f"{results[10000]['Plug_in']['mean']:.3f} \\pm {results[10000]['Plug_in']['std']:.3f}"
    print(f"& Plug-in learner & ${plug_in_100}$ & ${plug_in_200}$ & ${plug_in_500}$ & ${plug_in_1000}$ & ${plug_in_2000}$ & ${plug_in_5000}$ & ${plug_in_10000}$ \\\\")
    
    # SurvB row
    survB_100 = f"{results[100]['SurvB']['mean']:.3f} \\pm {results[100]['SurvB']['std']:.3f}"
    survB_200 = f"{results[200]['SurvB']['mean']:.3f} \\pm {results[200]['SurvB']['std']:.3f}"
    survB_500 = f"{results[500]['SurvB']['mean']:.3f} \\pm {results[500]['SurvB']['std']:.3f}"
    survB_1000 = f"{results[1000]['SurvB']['mean']:.3f} \\pm {results[1000]['SurvB']['std']:.3f}"
    survB_2000 = f"{results[2000]['SurvB']['mean']:.3f} \\pm {results[2000]['SurvB']['std']:.3f}"
    survB_5000 = f"{results[5000]['SurvB']['mean']:.3f} \\pm {results[5000]['SurvB']['std']:.3f}"
    survB_10000 = f"{results[10000]['SurvB']['mean']:.3f} \\pm {results[10000]['SurvB']['std']:.3f}"
    print(f"& \\method (\\emph{{ours}}) & \\textbf{{${survB_100}$}} & \\textbf{{${survB_200}$}} & \\textbf{{${survB_500}$}} & \\textbf{{${survB_1000}$}} & \\textbf{{${survB_2000}$}} & \\textbf{{${survB_5000}$}} & \\textbf{{${survB_10000}$}} \\\\")

    
    return results