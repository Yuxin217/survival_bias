import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys, os
from scipy import stats

from lifelines import KaplanMeierFitter
from matplotlib.colors import LinearSegmentedColormap

# Set normal font style
plt.rcParams.update({'font.style': 'normal'})

def create_custom_colormap():
    """Create custom colormap"""
    # Method 1: from color list
    colors = ['#ffffff', '#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#ff0000']  # white to red
    custom_cmap = LinearSegmentedColormap.from_list('custom_red', colors)
    
    return custom_cmap

# Create common custom colormaps
custom_red_cmap = create_custom_colormap()
custom_blue_cmap = LinearSegmentedColormap.from_list('custom_blue', 
                                                    ['#ffffff', '#e6f3ff', '#b3d9ff', '#80bfff', '#4da6ff', '#0066cc'])
custom_green_cmap = LinearSegmentedColormap.from_list('custom_green', 
                                                     ['#c6dfb8', '#a8d19a', '#8ac37c', '#6cb55e', '#4ea740', '#24a645'])

covariates = ['Age', 'Sex', 'Pathology', 'N_stage', 'CDK4', 'MYC_Gain', 'NKX2-1', 'SMAD4', 'TP53_alteration']
cov_name_dic = {
    'Age': 'Age',
    'Sex': 'Sex',
    'Pathology': 'Pathology',
    'N_stage': 'Clinical Stage',
    'TP53_alteration': 'TP53',
    'SMAD4': 'SMAD4',
    'CDK4': 'CDK4',
    'NKX2-1': 'NKX2-1',
    'MYC_Gain': 'MYC'}

def sample_size_table(df):
    
    print("Covariates".ljust(15) + "|".ljust(3) + "Sample sizes".ljust(15))
    print("=" * 35)
    print("Baseline characteristics")
    print("-" * 35)
    
    # Age
    age_55_plus = (df['Age'] >= 55).sum()
    age_55_minus = (df['Age'] < 55).sum()
    print(f"Age".ljust(15) + "|".ljust(3) + f"{age_55_plus + age_55_minus}".ljust(15))
    print(f"  >=55".ljust(15) + "|".ljust(3) + f"{age_55_plus}".ljust(15))
    print(f"  <55".ljust(15) + "|".ljust(3) + f"{age_55_minus}".ljust(15))
    
    # Sex
    male_count = (df['Sex'] == 0).sum()
    female_count = (df['Sex'] == 1).sum()
    print(f"Sex".ljust(15) + "|".ljust(3) + f"{male_count + female_count}".ljust(15))
    print(f"  Male".ljust(15) + "|".ljust(3) + f"{male_count}".ljust(15))
    print(f"  Female".ljust(15) + "|".ljust(3) + f"{female_count}".ljust(15))
    
    # Pathology
    path_I = (df['Pathology'] == 0).sum()
    path_II = (df['Pathology'] == 1).sum()
    path_III = (df['Pathology'] == 2).sum()
    print(f"Pathology".ljust(15) + "|".ljust(3) + f"{path_I + path_II + path_III}".ljust(15))
    print(f"  I".ljust(15) + "|".ljust(3) + f"{path_I}".ljust(15))
    print(f"  II".ljust(15) + "|".ljust(3) + f"{path_II}".ljust(15))
    print(f"  III".ljust(15) + "|".ljust(3) + f"{path_III}".ljust(15))
    
    # Clinical stage
    stage_II = (df['N_stage'] == 1).sum()
    stage_III = (df['N_stage'] == 2).sum()
    print(f"{cov_name_dic['N_stage']}".ljust(15) + "|".ljust(3) + f"{stage_II + stage_III}".ljust(15))
    print(f"  II".ljust(15) + "|".ljust(3) + f"{stage_II}".ljust(15))
    print(f"  III".ljust(15) + "|".ljust(3) + f"{stage_III}".ljust(15))
    
    print("\nGenes")
    print("-" * 35)
    
    # Define genes in alphabetical order
    genes = [
        ('CDK4', 'CDK4'),
        ('MYC_Gain', 'MYC'),
        ('NKX2-1', 'NKX2-1'),
        ('SMAD4', 'SMAD4'),
        ('TP53_alteration', 'TP53')
    ]
    
    # Process genes in alphabetical order
    for gene_col, gene_name in genes:
        gene_0 = (df[gene_col] == 0).sum()
        gene_1 = (df[gene_col] == 1).sum()
        print(f"{gene_name}".ljust(15) + "|".ljust(3) + f"{gene_0 + gene_1}".ljust(15))
        print(f"  0".ljust(15) + "|".ljust(3) + f"{gene_0}".ljust(15))
        print(f"  1".ljust(15) + "|".ljust(3) + f"{gene_1}".ljust(15))

def survival_curve_with_est(df, time_indicator, biomarker, project_root):
    plt.clf()
    
    if df['lb_A_0_est'].shape[0] == 1:
         df = pd.concat([df, df], ignore_index=True)
         # Add small Gaussian noise to the second row
         noise_std = 0.01  # Adjust this value to control noise level
         df.loc[1, 'lb_A_1_est'] += np.random.normal(0, noise_std)
         df.loc[1, 'up_A_1_est'] += np.random.normal(0, noise_std)
         df.loc[1, 'lb_A_0_est'] -= np.random.normal(0, noise_std)
         df.loc[1, 'up_A_0_est'] -= np.random.normal(0, noise_std)
        
    ''' CAPO: A = 0'''
    lb = df['lb_A_0_est']
    ub = df['up_A_0_est']
    est = df['A_0_est']
    kmf_lb_0 = KaplanMeierFitter()
    kmf_ub_0 = KaplanMeierFitter()
    kmf_est_0 = KaplanMeierFitter()
    common_times = np.unique(np.concatenate([lb, ub, est]))
    event_observed  = np.ones(len(df))
    kmf_lb_0.fit(lb, event_observed=event_observed, timeline=common_times)
    kmf_ub_0.fit(ub, event_observed=event_observed, timeline=common_times)
    kmf_est_0.fit(est, event_observed=event_observed, timeline=common_times)
    
    # Fill between LB and UB for control group using the actual survival function data
    times_0 = kmf_lb_0.survival_function_.index
    surv_lb_0 = kmf_lb_0.survival_function_.values.flatten()
    surv_ub_0 = kmf_ub_0.survival_function_.values.flatten()
    surv_est_0 = kmf_est_0.survival_function_.values.flatten()

    # Ensure both have the same length by taking the minimum
    min_len = min(len(surv_lb_0), len(surv_ub_0))
    times_0 = times_0[:min_len]
    surv_lb_0 = surv_lb_0[:min_len]
    surv_ub_0 = surv_ub_0[:min_len]
    surv_est_0 = surv_est_0[:min_len]
    diff_0 = surv_ub_0 - surv_lb_0
    mask_0 = diff_0 > 0
    surv_lb_0 = surv_lb_0[mask_0]
    surv_ub_0 = surv_ub_0[mask_0]
    surv_est_0 = surv_est_0[mask_0]
    times_0 = times_0[mask_0]
    

    ''' CAPO: A = 1'''
    lb = df['lb_A_1_est']
    ub = df['up_A_1_est']
    est = df['A_1_est']
    kmf_lb_1 = KaplanMeierFitter()
    kmf_ub_1 = KaplanMeierFitter()
    kmf_est_1 = KaplanMeierFitter()

    common_times = np.unique(np.concatenate([lb, ub, est]))
    kmf_lb_1.fit(lb, event_observed=event_observed, timeline=common_times)
    kmf_ub_1.fit(ub, event_observed=event_observed, timeline=common_times)
    kmf_est_1.fit(est, event_observed=event_observed, timeline=common_times)
    
    # Fill between LB and UB for control group using the actual survival function data
    times_1 = kmf_lb_1.survival_function_.index
    surv_lb_1 = kmf_lb_1.survival_function_.values.flatten()
    surv_ub_1 = kmf_ub_1.survival_function_.values.flatten()
    surv_est_1 = kmf_est_1.survival_function_.values.flatten()
    diff_1 = surv_ub_1 - surv_lb_1

    if time_indicator == "OS":
        mask_1 = (diff_1 > 0) & (times_1 <= 52)

    else:
        mask_1 = diff_1 > 0

    surv_lb_1 = surv_lb_1[mask_1]
    surv_ub_1 = surv_ub_1[mask_1]
    times_1 = times_1[mask_1]
    
    # Ensure both have the same length by taking the minimum
    min_len = min(len(surv_lb_1), len(surv_ub_1))
    times_1 = times_1[:min_len]
    surv_lb_1 = surv_lb_1[:min_len]
    surv_ub_1 = surv_ub_1[:min_len]
    surv_est_1 = surv_est_1[:min_len]

    '''plot part'''
    plt.plot(times_0, 100 * surv_est_0, color='#0066CC', label='Chemotherapy group')
    plt.fill_between(times_0, 100 * surv_lb_0, 100 * surv_ub_0, alpha=0.3, color='#0066CC', label='SurvB-learner')
    
    plt.plot(times_1, 100 * surv_est_1, color='#990099', label = 'Gefitinib group')
    plt.fill_between(times_1, 100 * surv_lb_1, 100 * surv_ub_1, alpha=0.3, color='#990099', label='SurvB-learner')

    plt.xlabel("Time [months]")
    plt.ylabel("Survival probability [%]")
    plt.grid(True, alpha=0.3)
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.savefig(f"{project_root}/figures/adjuvant_data/survival_curve_{time_indicator}_{biomarker}.svg")
    plt.savefig(f"{project_root}/figures/adjuvant_data/survival_curve_{time_indicator}_{biomarker}.pdf")
    plt.show()
    plt.close()


def survival_curve(df, time_indicator, biomarker, project_root):
    plt.clf()
    
    if df['lb_A_0_est'].shape[0] == 1:
         df = pd.concat([df, df], ignore_index=True)
         # Add small Gaussian noise to the second row
         noise_std = 0.01  # Adjust this value to control noise level
         df.loc[1, 'lb_A_1_est'] += np.random.normal(0, noise_std)
         df.loc[1, 'up_A_1_est'] += np.random.normal(0, noise_std)
         df.loc[1, 'lb_A_0_est'] -= np.random.normal(0, noise_std)
         df.loc[1, 'up_A_0_est'] -= np.random.normal(0, noise_std)
        
    ''' CAPO: A = 0'''
    lb = df['lb_A_0_est']
    ub = df['up_A_0_est']
    kmf_lb_0 = KaplanMeierFitter()
    kmf_ub_0 = KaplanMeierFitter()
    common_times = np.unique(np.concatenate([lb, ub]))
    event_observed  = np.ones(len(df))
    kmf_lb_0.fit(lb, event_observed=event_observed, timeline=common_times)
    kmf_ub_0.fit(ub, event_observed=event_observed, timeline=common_times)
    # kmf_lb_0.fit(lb, event_observed=df['Delta'], timeline=common_times)
    # kmf_ub_0.fit(ub, event_observed=df['Delta'], timeline=common_times)
    # kmf_est_0.fit(est, event_observed=df['Delta'], timeline=common_times)
    
    # Fill between LB and UB for control group using the actual survival function data
    times_0 = kmf_lb_0.survival_function_.index
    surv_lb_0 = kmf_lb_0.survival_function_.values.flatten()
    surv_ub_0 = kmf_ub_0.survival_function_.values.flatten()
    # surv_est_0 = kmf_est_0.survival_function_.values.flatten()

    # Ensure both have the same length by taking the minimum
    min_len = min(len(surv_lb_0), len(surv_ub_0))
    times_0 = times_0[:min_len]
    surv_lb_0 = surv_lb_0[:min_len]
    surv_ub_0 = surv_ub_0[:min_len]
    # surv_est_0 = surv_est_0[:min_len]
    diff_0 = surv_ub_0 - surv_lb_0
    mask_0 = diff_0 > 0
    surv_lb_0 = surv_lb_0[mask_0]
    surv_ub_0 = surv_ub_0[mask_0]
    # surv_est_0 = surv_est_0[mask_0]
    times_0 = times_0[mask_0]
    

    ''' CAPO: A = 1'''
    lb = df['lb_A_1_est']
    ub = df['up_A_1_est']
    # est = df['A_1_est']
    kmf_lb_1 = KaplanMeierFitter()
    kmf_ub_1 = KaplanMeierFitter()
    # kmf_est_1 = KaplanMeierFitter()

    common_times = np.unique(np.concatenate([lb, ub]))
    kmf_lb_1.fit(lb, event_observed=event_observed, timeline=common_times)
    kmf_ub_1.fit(ub, event_observed=event_observed, timeline=common_times)
    # kmf_est_1.fit(est, event_observed=event_observed, timeline=common_times)
    
    # Fill between LB and UB for control group using the actual survival function data
    times_1 = kmf_lb_1.survival_function_.index
    surv_lb_1 = kmf_lb_1.survival_function_.values.flatten()
    surv_ub_1 = kmf_ub_1.survival_function_.values.flatten()
    # surv_est_1 = kmf_est_1.survival_function_.values.flatten()
    diff_1 = surv_ub_1 - surv_lb_1

    if time_indicator == "OS":
        mask_1 = (diff_1 > 0) & (times_1 <= 52)

    else:
        mask_1 = diff_1 > 0

    surv_lb_1 = surv_lb_1[mask_1]
    surv_ub_1 = surv_ub_1[mask_1]
    times_1 = times_1[mask_1]
    
    # Ensure both have the same length by taking the minimum
    min_len = min(len(surv_lb_1), len(surv_ub_1))
    times_1 = times_1[:min_len]
    surv_lb_1 = surv_lb_1[:min_len]
    surv_ub_1 = surv_ub_1[:min_len]

    plt.fill_between(times_0, 100 * surv_lb_0, 100 * surv_ub_0, alpha=0.3, color='#0066CC', label='SurvB-learner (Chemotherapy group)')
    plt.fill_between(times_1, 100 * surv_lb_1, 100 * surv_ub_1, alpha=0.3, color='#990099', label='SurvB-learner (Gefitinib group)')

    plt.xlabel("Time [months]")
    plt.ylabel("Survival probability [%]")
    plt.grid(True, alpha=0.3)
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.savefig(f"{project_root}/figures/adjuvant_data/survival_curve_{time_indicator}_{biomarker}.svg")
    plt.savefig(f"{project_root}/figures/adjuvant_data/survival_curve_{time_indicator}_{biomarker}.pdf")
    plt.show()
    plt.close()



def bootstrap_analysis(df, subgroup_func, n_bootstrap=2000, random_state=42):

    np.random.seed(random_state)
    
    # Get the original subgroup
    original_subgroup = subgroup_func(df)
    n_original = len(original_subgroup)
    
    if n_original == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'samples': []
        }
    
    # Bootstrap samples
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_indices = np.random.choice(df.index, size=len(df), replace=True)
        bootstrap_df = df.loc[bootstrap_indices]
        
        # Get subgroup from bootstrap sample
        bootstrap_subgroup = subgroup_func(bootstrap_df)
        
        if len(bootstrap_subgroup) > 0:
            # Calculate mean of the statistic of interest (e.g., lb_cate_est)
            bootstrap_mean = bootstrap_subgroup['lb_cate_est'].mean()
            bootstrap_means.append(bootstrap_mean)
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate statistics
    mean_bootstrap = np.mean(bootstrap_means)
    std_bootstrap = np.std(bootstrap_means)
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return {
        'mean': mean_bootstrap,
        'std': std_bootstrap,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'samples': bootstrap_means
    }

def filter_subgroup(data, label):
    """
    Filter data based on the specified label/subgroup
    """
    if label.startswith('Age:'):
        # Age subgroups
        if label == 'Age: 26-40':
            return data[(data['Age'] >= 26) & (data['Age'] < 40)]
        elif label == 'Age: 40-50':
            return data[(data['Age'] >= 40) & (data['Age'] < 50)]
        elif label == 'Age: 50-60':
            return data[(data['Age'] >= 50) & (data['Age'] < 60)]
        elif label == 'Age: 60-70':
            return data[(data['Age'] >= 60) & (data['Age'] < 70)]
        elif label == 'Age: 70-74':
            return data[(data['Age'] >= 70) & (data['Age'] <= 74)]
    
    elif label.startswith('Pathology:'):
        # Pathology subgroups
        if label == 'Pathology: I':
            return data[data['Pathology'] == 0]
        elif label == 'Pathology: II':
            return data[data['Pathology'] == 1]
        elif label == 'Pathology: III':
            return data[data['Pathology'] == 2]
    
    elif label.startswith('Clinical stage:'):
        # Clinical stage subgroups
        if label == 'Clinical stage: I':
            return data[data['N_stage'] == 0]
        elif label == 'Clinical stage: II':
            return data[data['N_stage'] == 1]
        elif label == 'Clinical stage: III':
            return data[data['N_stage'] == 2]
    
    elif label in ['CDK4', 'MYC', 'NKX2-1', 'SMAD4', 'TP53']:
        # Gene subgroups - map display names to actual column names
        gene_mapping = {
            'CDK4': 'CDK4',
            'MYC': 'MYC_Gain',
            'NKX2-1': 'NKX2-1',
            'SMAD4': 'SMAD4',
            'TP53': 'TP53_alteration'
        }
        
        gene_col = gene_mapping[label]
        return data[data[gene_col] == 1]  # Positive cases
    
    else:
        # Default: return empty dataframe if label not recognized
        return data.iloc[0:0]  # Empty dataframe with same structure

def forest_plot_bootstrap(df, time_indicator, project_root, bound_type='lb', n_bootstrap=2000):
    """Create a vertical forest plot using bootstrap estimates with confidence intervals"""
    
    # Define labels and their corresponding subgroups
    # Genes in alphabetical order: CDK4, MYC, NKX2-1, SMAD4, TP53
    labels = ['Age: 26-40', 'Age: 40-50', 'Age: 50-60', 'Age: 60-70', 'Age: 70-74',
              'Pathology: I', 'Pathology: II', 'Pathology: III',
              'Clinical stage: II', 'Clinical stage: III',
              'CDK4', 'MYC', 'NKX2-1', 'SMAD4', 'TP53']
    
    # Define markers for different types of subgroups
    markers = ['o', 'o', 'o', 'o', 'o',           # Age groups (circles)
               's', 's', 's',                           # Pathology (squares)
               '^', '^',                           # Clinical stage (triangles)
               'D', 'D', 'D', 'D', 'D']           # Genes (diamonds)
    
    # Define colors for different types of subgroups
    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4',    # Age groups (blue)
              '#ff7f0e', '#ff7f0e', '#ff7f0e',                          # Pathology (orange)
              '#d62728', '#d62728',                                     # Clinical stage (green)
              '#2ca02c', '#2ca02c', '#2ca02c', '#2ca02c', '#2ca02c']    # Genes (red)
    
    # Calculate bootstrap statistics for each subgroup
    bootstrap_results = []
    sample_sizes = []
    prob_positive = []
    
    # print("Performing bootstrap analysis...")
    for i, label in enumerate(labels):
        # print(f"Processing {label}... ({i+1}/{len(labels)})")
        
        # Create a lambda function for this specific subgroup
        def subgroup_func(data):
            return filter_subgroup(data, label)
        
        # Perform bootstrap analysis
        result = bootstrap_analysis(df, subgroup_func, n_bootstrap=n_bootstrap)
        bootstrap_results.append(result)
        
        # Get original sample size and probability
        original_subgroup = filter_subgroup(df, label)
        sample_sizes.append(len(original_subgroup))
        if len(original_subgroup) > 0:
            prob = np.mean(original_subgroup[f'{bound_type}_cate_est'] > 0)
            prob_positive.append(prob)
        else:
            prob_positive.append(np.nan)
    
    # Extract means and confidence intervals
    means = [result['mean'] for result in bootstrap_results]
    ci_lower = [result['ci_lower'] for result in bootstrap_results]
    ci_upper = [result['ci_upper'] for result in bootstrap_results]
    
    # Create vertical forest plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # First plot error bars (without markers)
    x_pos = np.arange(len(labels))

    # plot all the lb samples
    for i in range(len(labels)):
        subgroup = filter_subgroup(df, labels[i])
        ax.scatter(np.repeat(x_pos[i], len(subgroup)), subgroup[f'{bound_type}_cate_est'], 
                  marker=markers[i], s=16, color=colors[i], alpha=0.2, edgecolor='white', linewidth=0.5)
    
    # Plot bootstrap confidence intervals
    for i in range(len(labels)):
        if not np.isnan(means[i]):
            # Calculate error bar from bootstrap CI
            ci_lower_val = ci_lower[i]
            ci_upper_val = ci_upper[i]
            mean_val = means[i]
            
            # Error bar extends from mean to CI bounds
            lower_err = mean_val - ci_lower_val
            upper_err = ci_upper_val - mean_val
            
            ax.errorbar(x_pos[i], mean_val, yerr=[[lower_err], [upper_err]], 
                       fmt='none', capsize=8, capthick=3, 
                       color=colors[i], alpha=0.7, elinewidth=2)
    
    # Then plot markers on top to cover the error bars
    for i in range(len(labels)):
        if not np.isnan(means[i]):
            # Calculate marker size based on sample size
            marker_size = max(4, min(15, 6 + sample_sizes[i] * 0.5))
            
            ax.plot(x_pos[i], means[i], marker=markers[i], markersize=12, color=colors[i], 
                   alpha=0.8, markeredgecolor='white', markeredgewidth=1.5)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.6, linewidth=2)
    
    x_labels_show = ['Age:\n 26-40', '\n 40-50', '\n 50-60', '\n 60-70', '\n 70-74',
            'Pathology:\n I', '\n II', '\n III',
            'Clinical stage:\n II', '\n III',
            'CDK4', 'MYC', 'NKX2-1', 'SMAD4', 'TP53']
    # Customize the plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels_show, ha='center', fontsize=12)
    ax.set_ylabel('Lower bounds (LBs)', fontsize=12)
    ax.set_ylim(-16, 12)
    ax.grid(True, alpha=0.3, axis='y', linewidth=1)
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make left and bottom borders thicker
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Adjust layout to reduce bottom whitespace
    plt.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.95)
    
    # Save the plot with minimal padding
    plt.savefig(f"{project_root}/figures/adjuvant_data/forest_plot_bootstrap_{time_indicator}.svg", 
                bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{project_root}/figures/adjuvant_data/forest_plot_bootstrap_{time_indicator}.pdf", 
                bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    return bootstrap_results

def calculate_proportion_ci(successes, total, confidence_level=0.95):
    """
    Calculate confidence interval for a proportion using the Wilson score interval
    which performs better than the normal approximation for small samples.
    
    Parameters:
    -----------
    successes : int
        Number of successes (LBs > 0)
    total : int
        Total sample size
    confidence_level : float
        Confidence level (default 0.95 for 95% CI)
    
    Returns:
    --------
    tuple
        (lower_bound, upper_bound) as percentages
    """
    if total == 0:
        return (0, 0)
    
    # Wilson score interval
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    p = successes / total
    n = total
    
    # Wilson score interval formula
    denominator = 1 + z**2 / n
    centre_adjusted_probability = (p + z**2 / (2 * n)) / denominator
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) * 100
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) * 100
    
    # Ensure bounds are within [0, 100]
    lower_bound = max(0, lower_bound)
    upper_bound = min(100, upper_bound)
    
    return (lower_bound, upper_bound)

def LBs_single_cov_table(df, time_indicator):
    """
    Generate table showing LBs > 0 percentages with confidence intervals for different subgroups
    """
    print("Subgroup".ljust(20) + "|".ljust(3) + "Sample size".ljust(12) + "|".ljust(3) + "LBs > 0 (95% CI)")
    print("=" * 70)
    
    # Age subgroups
    age_groups = [
        (26, 40, "26-40"),
        (40, 50, "40-50"), 
        (50, 60, "50-60"),
        (60, 70, "60-70"),
        (70, 74, "70-74")
    ]
    
    for min_age, max_age, label in age_groups:
        mask = (df['Age'] >= min_age) & (df['Age'] < max_age)
        if mask.sum() > 0:
            lb_positive = (df[mask]['lb_cate_est'] > 0).sum()
            total = mask.sum()
            percentage = (lb_positive / total) * 100
            ci_lower, ci_upper = calculate_proportion_ci(lb_positive, total)
            print(f"Age {label}".ljust(20) + "|".ljust(3) + f"{total}".ljust(12) + "|".ljust(3) + f"{percentage:.2f}% ({ci_lower:.2f}%-{ci_upper:.2f}%)")
    
    # Pathology subgroups
    path_groups = [0, 1, 2]  # I, II, III
    path_labels = ['I', 'II', 'III']
    
    for path_val, label in zip(path_groups, path_labels):
        mask = (df['Pathology'] == path_val)
        if mask.sum() > 0:
            lb_positive = (df[mask]['lb_cate_est'] > 0).sum()
            total = mask.sum()
            percentage = (lb_positive / total) * 100
            ci_lower, ci_upper = calculate_proportion_ci(lb_positive, total)
            print(f"Pathology {label}".ljust(20) + "|".ljust(3) + f"{total}".ljust(12) + "|".ljust(3) + f"{percentage:.2f}% ({ci_lower:.2f}%-{ci_upper:.2f}%)")
    
    # Clinical stage subgroups
    stage_groups = [1, 2]  # II, III (assuming 0=I, 1=II, 2=III)
    stage_labels = ['II', 'III']
    
    for stage_val, label in zip(stage_groups, stage_labels):
        mask = (df['N_stage'] == stage_val)
        if mask.sum() > 0:
            lb_positive = (df[mask]['lb_cate_est'] > 0).sum()
            total = mask.sum()
            percentage = (lb_positive / total) * 100
            ci_lower, ci_upper = calculate_proportion_ci(lb_positive, total)
            print(f"Clinical stage {label}".ljust(20) + "|".ljust(3) + f"{total}".ljust(12) + "|".ljust(3) + f"{percentage:.2f}% ({ci_lower:.2f}%-{ci_upper:.2f}%)")
    
    # Genetic markers (in alphabetical order)
    gene_columns = ['CDK4', 'MYC_Gain', 'NKX2-1', 'SMAD4', 'TP53_alteration']
    gene_labels = ['CDK4', 'MYC', 'NKX2-1', 'SMAD4', 'TP53']
    
    for gene_col, label in zip(gene_columns, gene_labels):
        mask = (df[gene_col] == 1)  # Positive cases
        if mask.sum() > 0:
            lb_positive = (df[mask]['lb_cate_est'] > 0).sum()
            total = mask.sum()
            percentage = (lb_positive / total) * 100
            ci_lower, ci_upper = calculate_proportion_ci(lb_positive, total)
            print(f"{label}".ljust(20) + "|".ljust(3) + f"{total}".ljust(12) + "|".ljust(3) + f"{percentage:.2f}% ({ci_lower:.2f}%-{ci_upper:.2f}%)")
    print("=" * 70)

class CustomDecisionTreeRegressor:
    """Custom decision tree that maximizes percentage of y>0 in child nodes"""
    
    def __init__(self, max_depth=2, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree_ = None
        self.feature_names_ = None
        
    def fit(self, X, y):
        """Fit the custom decision tree"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        
        self.tree_ = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {
                'is_leaf': True,
                'value': np.mean(y),
                'samples': len(y),
                'positive_percentage': (y > 0).mean()
            }
        
        best_split = self._find_best_split(X, y, depth)
        if best_split is None:
            return {
                'is_leaf': True,
                'value': np.mean(y),
                'samples': len(y),
                'positive_percentage': (y > 0).mean()
            }
        
        feature_idx, threshold = best_split
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'is_leaf': False,
            'feature': feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree,
            'samples': len(y),
            'positive_percentage': (y > 0).mean()
        }
    
    def _find_best_split(self, X, y, depth):
        """Find the best split using custom criterion"""
        best_gain = -np.inf
        best_split = None
        
        # Exclude Age feature at depth 1 (second layer)
        excluded_features = []
        if depth == 1 and hasattr(self, 'feature_names_') and self.feature_names_:
            try:
                age_idx = self.feature_names_.index('Age')
                excluded_features.append(age_idx)
            except ValueError:
                pass
        
        for feature_idx in range(X.shape[1]):
            if feature_idx in excluded_features:
                continue
                
            feature_values = np.unique(X[:, feature_idx])
            for threshold in feature_values[:-1]:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                gain = self._calculate_gain(y, left_mask, right_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold)
        
        return best_split
    
    def _calculate_gain(self, y, left_mask, right_mask):
        """Calculate gain: prioritize left leaf, then right leaf"""
        left_percentage = (y[left_mask] > 0).mean()
        right_percentage = (y[right_mask] > 0).mean()
        
        # Prioritize left leaf first, then right leaf
        # Use a weighted combination that favors left leaf
        return left_percentage * 0.9 + right_percentage * 1.1
    
    def predict(self, X):
        """Make predictions"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = []
        for i in range(X.shape[0]):
            pred = self._predict_single(X[i], self.tree_)
            predictions.append(pred)
        return np.array(predictions)
    
    def _predict_single(self, x, node):
        """Predict for a single sample"""
        if node['is_leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def get_tree_info(self):
        """Get tree information for visualization"""
        return self._get_node_info(self.tree_, 0)
    
    def _get_node_info(self, node, depth):
        """Get node information for visualization"""
        if node['is_leaf']:
            return {
                'depth': depth,
                'is_leaf': True,
                'value': node['value'],
                'samples': node['samples'],
                'positive_percentage': node['positive_percentage']
            }
        else:
            feature_name = self.feature_names_[node['feature']] if self.feature_names_ else f"Feature_{node['feature']}"
            return {
                'depth': depth,
                'is_leaf': False,
                'feature': feature_name,
                'threshold': node['threshold'],
                'samples': node['samples'],
                'positive_percentage': node['positive_percentage'],
                'left': self._get_node_info(node['left'], depth + 1),
                'right': self._get_node_info(node['right'], depth + 1)
            }

def visualize_custom_tree(tree_info):
    """Visualize the custom decision tree with confidence intervals"""
    def print_node(node, indent=""):
        if node['is_leaf']:
            # Calculate confidence interval for leaf node
            positive_count = int(node['positive_percentage'] * node['samples'])
            ci_lower, ci_upper = calculate_proportion_ci(positive_count, node['samples'])
            percentage = node['positive_percentage'] * 100
            print(f"{indent}Leaf: value={node['value']:.4f}, n={node['samples']}, LBs>0%={percentage:.2f}% ({ci_lower:.2f}%-{ci_upper:.2f}%)")
        else:
            # Calculate confidence interval for internal node
            positive_count = int(node['positive_percentage'] * node['samples'])
            ci_lower, ci_upper = calculate_proportion_ci(positive_count, node['samples'])
            percentage = node['positive_percentage'] * 100
            print(f"{indent}{node['feature']} <= {node['threshold']:.4f}")
            print(f"{indent}  n={node['samples']}, LBs>0%={percentage:.2f}% ({ci_lower:.2f}%-{ci_upper:.2f}%)")
            print(f"{indent}  ├─ Left:")
            print_node(node['left'], indent + "    ")
            print(f"{indent}  └─ Right:")
            print_node(node['right'], indent + "    ")
    
    print("Custom Decision Tree (maximizing percentage of y>0):")
    print("=" * 80)
    print_node(tree_info)
    print("=" * 80)


def build_custom_tree(df, time_indicator, max_depth=2):

    # Merge N_stage = 0 and 1
    df['N_stage_merged'] = df['N_stage'].apply(lambda x: 0 if x in [0, 1] else x)
    
    # Merge Pathology = 1 and 2
    df['Pathology_merged'] = df['Pathology'].apply(lambda x: 1 if x in [1, 2] else x)
    
    # Prepare features and target
    # feature_cols = ['Age', 'Smoking_history','Sex', 'Pathology_merged', 'N_stage_merged', 'CDK4', 'MYC_Gain', 'NKX2-1', 'SMAD4', 'TP53_alteration']
    feature_cols = ['CDK4', 'MYC_Gain', 'NKX2-1', 'SMAD4', 'TP53_alteration']
    X = df[feature_cols]
    y = df[time_indicator]
    
    # Build custom tree
    tree_custom = CustomDecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree_custom.fit(X, y)
    
    # Get tree information
    tree_info = tree_custom.get_tree_info()
    
    # Visualize tree
    visualize_custom_tree(tree_info)
    
    return tree_info


def print_co_alteration_table(df_dfs, df_os):
    """
    Print specific biomarker co-alteration table with confidence intervals
    Shows only specific combinations with DFS and OS columns
    """
    # Define specific biomarker combinations to show
    combinations = [
        ('CDK4', 'MYC_Gain'),
        ('CDK4', 'NKX2-1'), 
        ('NKX2-1', 'MYC_Gain'),
        ('SMAD4', 'MYC_Gain'),
        ('TP53_alteration', 'MYC_Gain'),
        ('TP53_alteration', 'NKX2-1'),
        ('TP53_alteration', 'SMAD4')
    ]
    
    # Display names for combinations
    display_combinations = [
        'CDK4 + MYC',
        'CDK4 + NKX2-1',
        'NKX2-1 + MYC', 
        'SMAD4 + MYC',
        'TP53 + MYC',
        'TP53 + NKX2-1',
        'TP53 + SMAD4'
    ]
    
    print("Lower bounds (LBs) > 0 [%] (95% CI)")
    print("=" * 80)
    print("Biomarker co-alteration".ljust(25) + "DFS (95% CI)".ljust(25) + "OS (95% CI)")
    print("-" * 80)
    
    # Calculate percentages for each combination
    for i, (bio1, bio2) in enumerate(combinations):
        display_name = display_combinations[i]
        
        # Calculate for DFS
        rate_lb_dfs, _ = filter_df(df_dfs, bio1, bio2)
        percentage_dfs = rate_lb_dfs * 100
        # Get sample size for DFS
        df_dfs_subset = df_subgroups(df_dfs, bio1, bio2)
        n_dfs = len(df_dfs_subset)
        positive_count_dfs = int(rate_lb_dfs * n_dfs) if n_dfs > 0 else 0
        ci_lower_dfs, ci_upper_dfs = calculate_proportion_ci(positive_count_dfs, n_dfs)
        
        # Calculate for OS  
        rate_lb_os, _ = filter_df(df_os, bio1, bio2)
        percentage_os = rate_lb_os * 100
        # Get sample size for OS
        df_os_subset = df_subgroups(df_os, bio1, bio2)
        n_os = len(df_os_subset)
        positive_count_os = int(rate_lb_os * n_os) if n_os > 0 else 0
        ci_lower_os, ci_upper_os = calculate_proportion_ci(positive_count_os, n_os)
        
        # Format the row
        row = display_name.ljust(25)
        row += f"{percentage_dfs:.2f}% ({ci_lower_dfs:.2f}%-{ci_upper_dfs:.2f}%)".ljust(25)
        row += f"{percentage_os:.2f}% ({ci_lower_os:.2f}%-{ci_upper_os:.2f}%)"
        
        print(row)
    
    print("=" * 80)

def filter_df(df, biomarker_1, biomarker_2):
    df_1 = df[(df[biomarker_1] == 1) & (df[biomarker_2] == 1)]
    n_small_group = len(df_1)
    
    if n_small_group == 0:
        rate_lb = 0
        rate_up = 0
    else:
        lb_above_zero = df_1[df_1['lb_cate_est'] > 0]
        up_below_zero = df_1[df_1['up_cate_est'] < 0]

        rate_lb = len(lb_above_zero) / n_small_group
        rate_up = len(up_below_zero) / n_small_group

    return rate_lb, rate_up

def df_subgroups(df, biomarker_1, biomarker_2):
    df_1 = df[(df[biomarker_1] == 1) & (df[biomarker_2] == 1)]
    return df_1


def twoD_heatmap(df_dfs, df_os, biomarkers, project_root):
    # Map display names to actual column names
    name_mapping = {
        'Clinical_stage': 'N_stage',
        'TP53': 'TP53_alteration', 
        'MYC': 'MYC_Gain'
    }
    
    # Convert display names to actual column names for data processing
    actual_biomarkers = [name_mapping.get(bio, bio) for bio in biomarkers]
    
    # Calculate DFS data matrix
    rate_lb_mat_dfs = pd.DataFrame(index=biomarkers, columns=biomarkers, dtype=float)
    for i, biomarker_1 in enumerate(biomarkers):
        for j, biomarker_2 in enumerate(biomarkers):
            actual_bio_1 = actual_biomarkers[i]
            actual_bio_2 = actual_biomarkers[j]
            rate_lb, rate_up = filter_df(df_dfs, actual_bio_1, actual_bio_2)
            rate_lb_mat_dfs.loc[biomarker_1, biomarker_2] = rate_lb
    
    # Calculate OS data matrix
    rate_lb_mat_os = pd.DataFrame(index=biomarkers, columns=biomarkers, dtype=float)
    for i, biomarker_1 in enumerate(biomarkers):
        for j, biomarker_2 in enumerate(biomarkers):
            actual_bio_1 = actual_biomarkers[i]
            actual_bio_2 = actual_biomarkers[j]
            rate_lb, rate_up = filter_df(df_os, actual_bio_1, actual_bio_2)
            rate_lb_mat_os.loc[biomarker_1, biomarker_2] = rate_lb

    # Combine DFS and OS data matrices
    combined_mat = pd.DataFrame(index=biomarkers, columns=biomarkers, dtype=float)
    for i, biomarker_1 in enumerate(biomarkers):
        for j, biomarker_2 in enumerate(biomarkers):
            if i <= j:  # Upper triangle and diagonal: DFS
                combined_mat.loc[biomarker_1, biomarker_2] = rate_lb_mat_dfs.loc[biomarker_1, biomarker_2]
            else:     # Lower triangle: OS
                combined_mat.loc[biomarker_1, biomarker_2] = rate_lb_mat_os.loc[biomarker_1, biomarker_2]

    # --- plotting ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    mask_upper = np.tril(np.ones_like(combined_mat, dtype=bool), k=-1)
    im1 = sns.heatmap(combined_mat.astype(float), mask=mask_upper, annot=True, cmap='Purples',
                      vmin=0, vmax=1, square=True, cbar=False, ax=ax, linewidths=0, annot_kws={'size': 14})
    
    mask_lower = np.triu(np.ones_like(combined_mat, dtype=bool))
    im2 = sns.heatmap(combined_mat.astype(float), mask=mask_lower, annot=True, cmap='Blues', 
                      vmin=0, vmax=1, square=True, cbar=False, ax=ax, linewidths=0, annot_kws={'size': 14})
    
    # Add custom legend with colored squares in horizontal layout with frame
    from matplotlib.patches import FancyBboxPatch
    
    # Create background frame for legend
    legend_bg = FancyBboxPatch((-0.25, 0.42), 0.2, 0.16, boxstyle="round,pad=0.01",
                              facecolor='white', edgecolor='black', linewidth=1,
                              transform=ax.transAxes, clip_on=False)

    # Modify label fonts - use original names for data processing, simplified names for display
    tick_names = ['Clinical stage', 'Pathology', 'TP53', 'SMAD4', 'CDK4', 'NKX2-1', 'MYC']
    
    ax.set_xticklabels(tick_names, fontsize=10, rotation=0)
    ax.set_yticklabels(tick_names, fontsize=10)
    # ax.set_xlabel('Clinical / Genomic features', fontsize=12)


    plt.tight_layout(pad=0.5)
    plt.savefig(f"{project_root}/figures/adjuvant_data/heatmap.svg")
    plt.show()

    return rate_lb_mat_dfs, rate_lb_mat_os


def calculate_survival_rates_by_subgroup(data):
    data = data.copy()
    data['TP53_alteration'] = (data['TP53_frsh_trunc'] == 1) | (data['TP53_E4E5mis'] == 1) | (data['TP53_CNV'] == 1)
    data['TP53_alteration'] = data['TP53_alteration'].astype(int)
    
    results = []
    
    subgroups = {
        'Age': {
            '>=55': data['Age'] >= 55,
            '<55': data['Age'] < 55
        },
        'Sex': {
            'Male': data['Sex'] == 0,
            'Female': data['Sex'] == 1
        },
        'Pathology': {
            'I': data['Pathology'] == 0,
            'II': data['Pathology'] == 1,
            'III': data['Pathology'] == 2
        },
        'Clinical_stage': {
            'II': data['N_stage'] == 1,
            'III': data['N_stage'] == 2
        }
    }
    
    genes = ['CDK4_Gain', 'MYC_Gain', 'NKX2_1_Gain', 'SMAD4', 'TP53_alteration']
    for gene in genes:
        if gene in data.columns:
            subgroups[f'{gene}'] = {
                '0': data[gene] == 0,
                '1': data[gene] == 1
            }
    
    for subgroup_name, subgroup_conditions in subgroups.items():
        for condition_name, condition_mask in subgroup_conditions.items():
            subgroup_data = data[condition_mask]
            
            if len(subgroup_data) == 0:
                continue
            
            gefitinib_data = subgroup_data[subgroup_data['Adj'] == 'gefitinib']
            chemo_data = subgroup_data[subgroup_data['Adj'] == 'chemo']
            
            gefitinib_rate_dfs = 0
            gefitinib_n = 0
            if len(gefitinib_data) > 0:
                gefitinib_events_dfs = (gefitinib_data['DFS_status'] == 1).sum()
                gefitinib_n = len(gefitinib_data)
                gefitinib_rate_dfs = (1 - gefitinib_events_dfs / gefitinib_n) * 100
            
            chemo_rate_dfs = 0
            chemo_n = 0
            if len(chemo_data) > 0:
                chemo_events_dfs = (chemo_data['DFS_status'] == 1).sum()
                chemo_n = len(chemo_data)
                chemo_rate_dfs = (1 - chemo_events_dfs / chemo_n) * 100
            
            gefitinib_rate_os = 0
            gefitinib_n = 0
            if len(gefitinib_data) > 0:
                gefitinib_events_os = (gefitinib_data['OS_status'] == 1).sum()
                gefitinib_n = len(gefitinib_data)
                gefitinib_rate_os = (1 - gefitinib_events_os / gefitinib_n) * 100
            
            chemo_rate_os = 0
            chemo_n = 0
            if len(chemo_data) > 0:
                chemo_events_os = (chemo_data['OS_status'] == 1).sum()
                chemo_n = len(chemo_data)
                chemo_rate_os = (1 - chemo_events_os / chemo_n) * 100
            
            total_sample_size = len(subgroup_data)
            
            results.append({
                'Subgroup': f"{subgroup_name}_{condition_name}",
                'Sample_size': total_sample_size,
                'Gefitinib_rate_dfs%': round(gefitinib_rate_dfs, 2) if gefitinib_n > 0 else None,
                'Chemo_rate_dfs%': round(chemo_rate_dfs, 2) if chemo_n > 0 else None,
                'Gefitinib_rate_os%': round(gefitinib_rate_os, 2) if gefitinib_n > 0 else None,
                'Chemo_rate_os%': round(chemo_rate_os, 2) if chemo_n > 0 else None
            })
    
    return pd.DataFrame(results)

def generate_survival_analysis_table(data, project_root):
    """
    Generate comprehensive survival analysis table with LaTeX output
    """
    data_copy = data.copy()
    mask = data['N_stage'] == 0
    data = data[~mask]
       
    # Main execution
    dropout_results = calculate_survival_rates_by_subgroup(data)
    print("Survival Rates by Subgroup (1 - Event rates)")
    print("=" * 100)
    print(dropout_results.to_string(index=False))
    
    gefitinib_total = len(data[data['Adj'] == 'gefitinib'])
    gefitinib_events_dfs = len(data[(data['Adj'] == 'gefitinib') & (data['DFS_status'] == 1)])
    gefitinib_rate_dfs = (1 - gefitinib_events_dfs / gefitinib_total) * 100
    gefitinib_events_os = len(data[(data['Adj'] == 'gefitinib') & (data['OS_status'] == 1)])
    gefitinib_rate_os = (1 - gefitinib_events_os / gefitinib_total) * 100
    
    chemo_total = len(data[data['Adj'] == 'chemo'])
    chemo_events_dfs = len(data[(data['Adj'] == 'chemo') & (data['DFS_status'] == 1)])
    chemo_rate_dfs = (1 - chemo_events_dfs / chemo_total) * 100
    chemo_events_os = len(data[(data['Adj'] == 'chemo') & (data['OS_status'] == 1)])
    chemo_rate_os = (1 - chemo_events_os / chemo_total) * 100
    
    print("\n" + "=" * 100)
    print(f"{'OVERALL':<25} {gefitinib_total + chemo_total:<8} {gefitinib_rate_dfs:.2f}{'':<6} {chemo_rate_dfs:.2f}{'':<6} {gefitinib_rate_os:.2f}{'':<6} {chemo_rate_os:.2f}")
    overall_censoring_rate_dfs = (1 - len(data[data['DFS_status'] == 1]) / 170) * 100
    overall_censoring_rate_os = (1 - len(data[data['OS_status'] == 1]) / 170) * 100
    print(f"{'OVERALL':<25} overall dfs rate{overall_censoring_rate_dfs:.2f}{'':<6} overall os rate{overall_censoring_rate_os:.2f}{'':<6}")
    
    return dropout_results
