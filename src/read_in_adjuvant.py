import pandas as pd
import numpy as np

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
    
    # Convert to numpy arrays after sorting
    x_name = ['Age', 'Sex', 'Smoking_history', 'Pathology', 'N_stage', 'TP53_alteration', 'SMAD4', 'CDK4', 'NKX2-1', 'MYC_Gain']
    X = np.array(df_sorted[x_name], dtype=int)
    A = (df_sorted['Treatment'] == 'gefitinib').astype(int).to_numpy(dtype=int)
    y = df_sorted['Time'].to_numpy(dtype=float)
    Delta = 1 - df_sorted['Status'].to_numpy(dtype=int)
    
    return X, A, y, Delta, x_name