import sys, os
import numpy as np
import pandas as pd

def load_data(data_path):
    return pd.read_csv(data_path)

def read_in_adjuvant(data_path, time_indicator):
    '''Function to read in ADJUVANT data and return X, A, y, Delta and distinguished by time_indicator'''
    data = pd.read_csv(data_path)
    
    # First create DataFrame with all columns we need
    df = pd.DataFrame({
        'Age': data['Age'],
        'Sex': data['Sex'],
        'Smoking_history': data['Smoking_history'],
        'Pathology': data['Pathology'],
        'TP53_frsh_trunc': data['TP53_frsh_trunc'],
        'TP53_E4E5mis': data['TP53_E4E5mis'],
        'TP53_CNV': data['TP53_CNV'],
        'SMAD4': data['SMAD4'],
        'CDK4': data['CDK4_Gain'],
        'NKX2-1': data['NKX2_1_Gain'],
        'N_stage': data['N_stage'],
        'MYC_Gain': data['MYC_Gain'],
        'Treatment': data['Adj'],
        'Time': data[time_indicator],
        'Status': data[time_indicator + '_status'],
    })
    
    df['TP53_alteration'] = (df['TP53_frsh_trunc'] == 1) | (df['TP53_E4E5mis'] == 1) | (df['TP53_CNV'] == 1)
    df['TP53_alteration'] = df['TP53_alteration'].astype(int)
    mask = df['N_stage'] == 0
    df = df[~mask]
    df_sorted = df.sort_values(['Age'], ascending=[True])
    
    x_name = ['Age', 'Sex', 'Smoking_history', 'Pathology', 'N_stage', 'TP53_alteration', 'SMAD4', 'CDK4', 'NKX2-1', 'MYC_Gain']
    X = np.array(df_sorted[x_name], dtype=int)
    A = (df_sorted['Treatment'] == 'gefitinib').astype(int).to_numpy(dtype=int)
    y = df_sorted['Time'].to_numpy(dtype=float)
    Delta = 1 - df_sorted['Status'].to_numpy(dtype=int)
    
    return X, A, y, Delta, x_name


def compute_window_stats(df, cols, window_size=0.5):
    """
    Compute window statistics with NaN handling
    """
    df['X_bin'] = (df['X'] / window_size).round() * window_size
    
    result_df = pd.DataFrame({'X':np.arange(10, 100.01, 0.5)})
    
    for col in cols:
        grouped = df.groupby('X_bin')[col]
        
        # Create temporary dataframe with mean and std
        temp_df = pd.DataFrame({
            'X': grouped.mean().index,
            f'{col}_mean': grouped.mean().values,
            f'{col}_std': grouped.std().values
        })
        
        # Merge with result_df
        result_df = result_df.merge(temp_df, on='X', how='left')
    return result_df

def results_preprocess(df):
    # Select relevant columns
    columns_to_keep = ['True_CATE', 'SurvB_lb', 'SurvB_ub', 'Plug_in_lb', 'Plug_in_ub', 'Oracle_lb', 'Oracle_ub']

    # Process non-Oracle columns with smaller window
    non_oracle_cols = ['True_CATE', 'SurvB_lb', 'SurvB_ub', 'Plug_in_lb', 'Plug_in_ub']
    df_processed = compute_window_stats(df, non_oracle_cols, window_size=0.5)
    
    # Process Oracle columns with larger window for smoother curves
    oracle_cols = ['Oracle_lb', 'Oracle_ub']
    df_oracle = compute_window_stats(df, oracle_cols, window_size=0.5)  # Larger window for Oracle
    
    # Merge Oracle results with the main dataframe
    df_processed = df_processed.merge(df_oracle[['X', 'Oracle_lb_mean', 'Oracle_lb_std', 'Oracle_ub_mean', 'Oracle_ub_std']], on='X', how='left')
    
    return df_processed

if __name__ == "__main__":

    data_path = "data/adjuvant_data/multigene-171_combined.csv"
    X, A, y, Delta = read_in_adjuvant(data_path)
    
