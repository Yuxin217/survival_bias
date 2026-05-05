import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)


def plot_details(axes, df, censoring_strength, y_lim, dr_df=None):
    df = df.sort_values('X')

    axes.plot(df['X'], df['True_CATE'],  color='red',     label='True CATE',          alpha=1 - censoring_strength)
    axes.plot(df['X'], df['Plug_in_lb'], color="#78A040", label="Plug-in learner",     alpha=1 - censoring_strength)
    axes.plot(df['X'], df['Plug_in_ub'], color="#78A040",                              alpha=1 - censoring_strength)
    axes.plot(df['X'], df['Oracle_lb'],  color="#990099", label="Oracle Bounds",       alpha=1 - censoring_strength)
    axes.plot(df['X'], df['Oracle_ub'],  color="#990099",                              alpha=1 - censoring_strength)
    axes.fill_between(df['X'], df['SurvB_lb'], df['SurvB_ub'], color="#0066CC", alpha=0.2, label="SurvB-learner (Ours)")

    if dr_df is not None and len(dr_df) > 0:
        dr_df = dr_df.sort_values('X')
        axes.plot(dr_df['X'], dr_df['DR_est'], color="#808080", label='Point estimator', alpha= 1 - censoring_strength)

    axes.set_ylim(y_lim)


def _load_dr(root_path, response_type, censoring_strength, seed_list):
    frames = []
    for seed in seed_list:
        path = f"{root_path}/seed_{seed}/dr_est/{response_type}_function_{censoring_strength}.csv"
        if os.path.exists(path):
            frames.append(pd.read_csv(path)[['X', 'DR_est']])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).groupby('X', as_index=False).mean()



def plot_gamma(root_path, response_idx, subfig_idx, censoring_strength_list, response_types, y_lim_dic, axes, seed_list, dr_seed_list=None):
    for censoring_strength in censoring_strength_list:
        df = pd.read_csv(f"{root_path}/seed_{seed_list[0]}/gamma/{response_types[response_idx]}_function_gamma_{censoring_strength}.csv")
        dr_df = _load_dr(root_path, response_types[response_idx], censoring_strength, dr_seed_list or seed_list)
        plot_details(axes[response_idx, subfig_idx], df, censoring_strength, y_lim_dic[response_types[response_idx]], dr_df=dr_df)
        add_rmse_overlay(axes[response_idx, subfig_idx], df)


def plot_non_informative(root_path, response_idx, subfig_idx, censoring_strength_list, response_types, y_lim_dic, axes, seed_list, dr_seed_list=None):
    for censoring_strength in censoring_strength_list:
        df = pd.read_csv(f"{root_path}/seed_{seed_list[0]}/non_informative/{response_types[response_idx]}_function_non_informative_{censoring_strength}.csv")
        dr_df = _load_dr(root_path, response_types[response_idx], censoring_strength, dr_seed_list or seed_list)
        plot_details(axes[response_idx, subfig_idx], df, censoring_strength, y_lim_dic[response_types[response_idx]], dr_df=dr_df)
        add_rmse_overlay(axes[response_idx, subfig_idx], df)

    
DR_CENSORING_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def plot_censoring_strength_convergence(root_path, response_idx, subfig_idx, response_types, y_lim_dic, axes, seed, dr_seed=None):
    response_type = response_types[response_idx]
    data_dir = os.path.join(root_path, f"seed_{seed}", "non_informative")
    prefix = f"{response_type}_function_non_informative_"

    censoring_pairs = []
    for filename in os.listdir(data_dir):
        if filename.startswith(prefix) and filename.endswith(".csv"):
            xi_str = filename[len(prefix):-4]
            try:
                xi_val = float(xi_str)
                if 0.0 <= xi_val <= 1.0:
                    censoring_pairs.append((xi_val, xi_str))
            except ValueError:
                continue
    if not censoring_pairs:
        raise FileNotFoundError(f"No non_informative files found for {response_type} in {data_dir}")
    censoring_pairs = sorted(censoring_pairs)

    true_cate, survb_lb, survb_ub, oracle_lb, oracle_ub, plug_in_lb, plug_in_ub = [], [], [], [], [], [], []
    censoring_array = []

    for xi, xi_token in censoring_pairs:
        df = pd.read_csv(f"{data_dir}/{response_type}_function_non_informative_{xi_token}.csv")
        idx = int(np.argmin(np.abs(df['X'].to_numpy() - 50)))
        censoring_array.append(xi)
        true_cate.append(df['True_CATE'].iloc[idx])
        survb_lb.append(df['SurvB_lb'].iloc[idx])
        survb_ub.append(df['SurvB_ub'].iloc[idx])
        oracle_lb.append(df['Oracle_lb'].iloc[idx])
        oracle_ub.append(df['Oracle_ub'].iloc[idx])
        plug_in_lb.append(df['Plug_in_lb'].iloc[idx])
        plug_in_ub.append(df['Plug_in_ub'].iloc[idx])

    ax = axes[response_idx, subfig_idx]
    ax.plot(censoring_array, true_cate,  color="red",     label="True CATE")
    ax.plot(censoring_array, plug_in_lb, color="#78A040", label="Plug-in learner")
    ax.plot(censoring_array, plug_in_ub, color="#78A040")
    ax.plot(censoring_array, oracle_lb,  color="#990099", label="Oracle Bounds")
    ax.plot(censoring_array, oracle_ub,  color="#990099")
    ax.fill_between(censoring_array, survb_lb, survb_ub, color="#0066CC", alpha=0.2, label="SurvB-learner (Ours)")

    if dr_seed is not None:
        dr_x, dr_y = [], []
        _dr_seed = dr_seed
        for xi in DR_CENSORING_LIST:
            dr_path = f"{root_path}/seed_{_dr_seed}/dr_est/{response_type}_function_{xi}.csv"
            if not os.path.exists(dr_path):
                continue
            dr_df = pd.read_csv(dr_path)
            dr_idx = int(np.argmin(np.abs(dr_df['X'].to_numpy() - 50)))
            dr_x.append(xi)
            dr_y.append(dr_df['DR_est'].iloc[dr_idx])
        if dr_x:
            ax.plot(dr_x, dr_y, color="#808080", marker='o', markersize=3, label='Point estimator')

    ax.invert_xaxis()
    ax.set_ylim(y_lim_dic[response_types[response_idx]])


def add_rmse_overlay(ax, df):
    plug_in_lb_rmse = np.sqrt(np.mean((df['Plug_in_lb'] - df['Oracle_lb'])**2))
    plug_in_ub_rmse = np.sqrt(np.mean((df['Plug_in_ub'] - df['Oracle_ub'])**2))
    survb_lb_rmse   = np.sqrt(np.mean((df['SurvB_lb']   - df['Oracle_lb'])**2))
    survb_ub_rmse   = np.sqrt(np.mean((df['SurvB_ub']   - df['Oracle_ub'])**2))

    plug_in_rmse = (plug_in_lb_rmse + plug_in_ub_rmse) / 2
    survb_rmse   = (survb_lb_rmse   + survb_ub_rmse)   / 2
    reduction    = (1 - survb_rmse / plug_in_rmse) * 100

    text = (f'Plug-in learner: {plug_in_rmse:.3f}\n'
            f'SurvB-learner (Ours): {survb_rmse:.3f}\n'
            f'Error reduction: {reduction:.0f}%')
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=7,
            ha='left', va='top',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='white', alpha=0.9, edgecolor='black'))


def plot_censoring_strength_rmse(root_path, censoring_strength_list, seed_list, output_path, dr_seed_list=None, dr_seed=None):
    response_types = ["exp", "sin", "logistic_sin"]
    y_lim_dic = {"sin": [-10, 40], "exp": [0, 170], "logistic_sin": [-7, 60]}

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    for response_idx in range(3):
        for subfig_idx in range(3):
            if subfig_idx == 0:
                plot_gamma(root_path, response_idx, subfig_idx, censoring_strength_list, response_types, y_lim_dic, axes, seed_list, dr_seed_list=dr_seed_list)
            elif subfig_idx == 1:
                plot_non_informative(root_path, response_idx, subfig_idx, censoring_strength_list, response_types, y_lim_dic, axes, seed_list, dr_seed_list=dr_seed_list)
            else:
                plot_censoring_strength_convergence(root_path, response_idx, subfig_idx, response_types, y_lim_dic, axes, seed_list[0], dr_seed=dr_seed)

    axes[2, 0].set_xlabel('X')
    axes[2, 1].set_xlabel('X')
    axes[2, 2].set_xlabel('Censoring strength ' + r'$\xi$')

    for response_idx in range(3):
        axes[response_idx, 0].set_ylabel('CATE')

    function_names = ["Exponential", "Sin", "Logistic-Sin"]
    for response_idx in range(3):
        axes[response_idx, 0].text(-0.25, 0.5, function_names[response_idx],
                                   transform=axes[response_idx, 0].transAxes,
                                   rotation=90, ha='center', va='center', fontsize=14)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels, unique_handles = [], []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    handles_new, labels_new = axes[0, 2].get_legend_handles_labels()
    for handle, label in zip(handles_new, labels_new):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    fig.legend(unique_handles, unique_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=6, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/results_{censoring_strength_list[0]}_{dr_seed}.pdf", bbox_inches='tight')


if __name__ == "__main__":
    root_path = os.path.join(CODE_DIR, "results", "synthetic_data", "rct")
    output_path = os.path.join(CODE_DIR, "figures", "synthetic_data", "rct")
    censoring_strength_list = [0.2]
    seed_list = [22]
    dr_seed_list = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

    for dr_seed in [23]:
        plot_censoring_strength_rmse(
            root_path=root_path,
            censoring_strength_list=censoring_strength_list,
            seed_list=seed_list,
            output_path=output_path,
            dr_seed_list=dr_seed_list,
            dr_seed=dr_seed,
        )
        print(f"Saved 3x3 RCT plot to {output_path} (seed={seed_list}, dr_seed={dr_seed})")
