import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
import copy

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.synthetic_data_generation import SyntheticDataModule

EPS = 1e-10

# ------------------------------------------------------------------ #
#  Utils                                                              #
# ------------------------------------------------------------------ #

def filter_uncensored_data(X, A, y, Delta):
    return X[Delta == 0], A[Delta == 0], y[Delta == 0]

def filter_censored_data(X, A, y, Delta):
    return X[Delta == 1], A[Delta == 1], y[Delta == 1]

# ------------------------------------------------------------------ #
#  Nuisance Estimation                                                #
# ------------------------------------------------------------------ #

def estimate_nu(model, train_data, test_data):
    X, A, y = train_data
    test_X   = test_data[0]
    model_use = copy.deepcopy(model)
    model_use.fit(np.concatenate((X, A.reshape(-1, 1)), axis=1), y)
    nu_A0 = model_use.predict(np.concatenate((test_X, np.zeros(len(test_X)).reshape(-1, 1)), axis=1))
    nu_A1 = model_use.predict(np.concatenate((test_X, np.ones(len(test_X)).reshape(-1, 1)),  axis=1))
    return nu_A0, nu_A1

def estimate_pi(model, train_data, test_data):
    X, A  = train_data
    X_test = test_data[0]
    model = copy.deepcopy(model)
    model.fit(X, A)
    pi = model.predict_proba(X_test)[:, 1]
    return np.clip(pi, EPS, 1 - EPS)

def estimate_xi(model, train_data, test_data):
    X, A, Delta = train_data
    X_test      = test_data[0]
    model = copy.deepcopy(model)
    model.fit(np.concatenate((X, A.reshape(-1, 1)), axis=1), Delta)
    xi_A0 = model.predict_proba(np.concatenate((X_test, np.zeros(len(X_test)).reshape(-1, 1)), axis=1))[:, 1]
    xi_A1 = model.predict_proba(np.concatenate((X_test, np.ones(len(X_test)).reshape(-1, 1)),  axis=1))[:, 1]
    return np.clip(xi_A0, EPS, 1 - EPS), np.clip(xi_A1, EPS, 1 - EPS)

def oracle_response_functions(response_type, X):
    data_gen = SyntheticDataModule()
    function_dict = {
        "sin":          data_gen.sin_response,
        "exp":          data_gen.exp_response,
        "logistic_sin": data_gen.logistic_sin_response,
    }
    if X.ndim == 2:
        X = X.flatten()
    n = len(X)
    nu_A0 = function_dict[response_type](n, np.zeros(n), X, U=None, noise=False)
    nu_A1 = function_dict[response_type](n, np.ones(n),  X, U=None, noise=False)
    return nu_A0, nu_A1

# ------------------------------------------------------------------ #
#  Bounds                                                             #
# ------------------------------------------------------------------ #

def oracle_bounds(X, A, y, Delta, bounds_type, gamma, response_type):
    nu_d0_A0, nu_d0_A1 = oracle_response_functions(response_type, X)

    model_xi = LogisticRegression()
    xi_A0, xi_A1 = estimate_xi(model_xi, train_data=[X, A, Delta], test_data=[X])

    censoring_strength = Delta.mean()
    if censoring_strength == 0:
        nu_d1_A0, nu_d1_A1 = 0, 0
    else:
        nu_d1_A0, nu_d1_A1 = oracle_response_functions(response_type, X)

    lb_A0 = nu_d0_A0 * (1 - xi_A0) + nu_d1_A0 * xi_A0
    lb_A1 = nu_d0_A1 * (1 - xi_A1) + nu_d1_A1 * xi_A1

    if bounds_type == "gamma":
        ub_A0 = lb_A0 + gamma * xi_A0
        ub_A1 = lb_A1 + gamma * xi_A1
    elif bounds_type == "non_informative":
        t_max = gamma
        ub_A0 = nu_d0_A0 * (1 - xi_A0) + t_max * xi_A0
        ub_A1 = nu_d0_A1 * (1 - xi_A1) + t_max * xi_A1

    lb_cate = lb_A1 - ub_A0
    up_cate = ub_A1 - lb_A0
    return lb_A0, ub_A0, lb_A1, ub_A1, lb_cate, up_cate


def plug_in_bounds(X, A, y, Delta, bounds_type, gamma):
    X_d0, A_d0, y_d0 = filter_uncensored_data(X, A, y, Delta)
    X_d1, A_d1, y_d1 = filter_censored_data(X, A, y, Delta)

    model_nu = RandomForestRegressor(n_estimators=6, max_depth=4)
    nu_d0_A0, nu_d0_A1 = estimate_nu(model_nu, train_data=[X_d0, A_d0, y_d0], test_data=[X])

    censoring_strength = Delta.mean()
    if censoring_strength == 0:
        nu_d1_A0, nu_d1_A1, xi_A0, xi_A1 = 0, 0, 0, 0
    else:
        model_xi = LogisticRegression()
        xi_A0, xi_A1 = estimate_xi(model_xi, train_data=[X, A, Delta], test_data=[X])
        nu_d1_A0, nu_d1_A1 = estimate_nu(model_nu, train_data=[X_d1, A_d1, y_d1], test_data=[X])

    lb_A0 = nu_d0_A0 * (1 - xi_A0) + nu_d1_A0 * xi_A0
    lb_A1 = nu_d0_A1 * (1 - xi_A1) + nu_d1_A1 * xi_A1

    if bounds_type == "gamma":
        ub_A0 = lb_A0 + gamma * xi_A0
        ub_A1 = lb_A1 + gamma * xi_A1
    elif bounds_type == "non_informative":
        t_max = gamma
        ub_A0 = nu_d0_A0 * (1 - xi_A0) + t_max * xi_A0
        ub_A1 = nu_d0_A1 * (1 - xi_A1) + t_max * xi_A1

    lb_cate = lb_A1 - ub_A0
    up_cate = ub_A1 - lb_A0
    return lb_A0, ub_A0, lb_A1, ub_A1, lb_cate, up_cate


def survb_bounds(X, A, y, Delta, bounds_type, gamma):
    n = len(X)
    phi_lb_A0  = np.empty(n); phi_ub_A0  = np.empty(n)
    phi_lb_A1  = np.empty(n); phi_ub_A1  = np.empty(n)
    phi_lb_cate = np.empty(n); phi_ub_cate = np.empty(n)
    kf = KFold(n_splits=3, shuffle=True, random_state=1)

    for tr, te in kf.split(X):
        X_d0, A_d0, y_d0 = filter_uncensored_data(X[tr], A[tr], y[tr], Delta[tr])
        X_d1, A_d1, y_d1 = filter_censored_data(X[tr], A[tr], y[tr], Delta[tr])

        pi_te = estimate_pi(LogisticRegression(), train_data=[X[tr], A[tr]], test_data=[X[te]])

        model_nu = RandomForestRegressor()
        nu_d0_A0_te, nu_d0_A1_te = estimate_nu(model_nu, train_data=[X_d0, A_d0, y_d0], test_data=[X[te]])

        if Delta.mean() == 0:
            nu_d1_A0_te = nu_d1_A1_te = xi_A0_te = xi_A1_te = 0
        else:
            nu_d1_A0_te, nu_d1_A1_te = estimate_nu(model_nu, train_data=[X_d1, A_d1, y_d1], test_data=[X[te]])
            xi_A0_te, xi_A1_te = estimate_xi(LogisticRegression(), train_data=[X[tr], A[tr], Delta[tr]], test_data=[X[te]])

        phi_a0_lb_te = (((1-A[te])*(1-Delta[te]))/(1-pi_te) * y[te]
                        - (1-A[te])/(1-pi_te) * nu_d0_A0_te * (1-Delta[te]) + nu_d0_A0_te * (1-xi_A0_te)
                        + ((1-A[te])*Delta[te])/(1-pi_te) * y[te]
                        - (1-A[te])/(1-pi_te) * nu_d1_A0_te * Delta[te] + nu_d1_A0_te * xi_A0_te)

        phi_a1_lb_te = ((A[te]*(1-Delta[te]))/pi_te * y[te]
                        - A[te]/pi_te * nu_d0_A1_te * (1-Delta[te]) + nu_d0_A1_te * (1-xi_A1_te)
                        + (A[te]*Delta[te])/pi_te * y[te]
                        - A[te]/pi_te * nu_d1_A1_te * Delta[te] + nu_d1_A1_te * xi_A1_te)

        if bounds_type == "gamma":
            phi_a0_ub_te = phi_a0_lb_te + gamma * xi_A0_te
            phi_a1_ub_te = phi_a1_lb_te + gamma * xi_A1_te
        elif bounds_type == "non_informative":
            t_max = gamma
            phi_a0_ub_te = nu_d0_A0_te * (1 - xi_A0_te) + t_max * xi_A0_te
            phi_a1_ub_te = nu_d0_A1_te * (1 - xi_A1_te) + t_max * xi_A1_te

        phi_lb_cate[te] = phi_a1_lb_te - phi_a0_ub_te
        phi_ub_cate[te] = phi_a1_ub_te - phi_a0_lb_te
        phi_lb_A0[te]   = phi_a0_lb_te;  phi_ub_A0[te] = phi_a0_ub_te
        phi_lb_A1[te]   = phi_a1_lb_te;  phi_ub_A1[te] = phi_a1_ub_te

    mu_lb_a0    = RandomForestRegressor().fit(X, phi_lb_A0).predict(X)
    mu_ub_a0    = RandomForestRegressor().fit(X, phi_ub_A0).predict(X)
    mu_lb_a1    = RandomForestRegressor().fit(X, phi_lb_A1).predict(X)
    mu_ub_a1    = RandomForestRegressor().fit(X, phi_ub_A1).predict(X)
    tau_lb_cate = RandomForestRegressor().fit(X, phi_lb_cate).predict(X)
    tau_ub_cate = RandomForestRegressor().fit(X, phi_ub_cate).predict(X)
    return mu_lb_a0, mu_ub_a0, mu_lb_a1, mu_ub_a1, tau_lb_cate, tau_ub_cate


# ------------------------------------------------------------------ #
#  Adjuvant (real data)                                               #
# ------------------------------------------------------------------ #

def survb_bounds_adjuvant(X, A, y, Delta, bounds_type, time_indicator):
    t_max_dic = {"OS": 50, "DFS": 50}
    t_max = t_max_dic[time_indicator]

    X_uncensored, A_uncensored, y_uncensored = filter_uncensored_data(X, A, y, Delta)
    X_censored,   A_censored,   y_censored   = filter_censored_data(X, A, y, Delta)

    model_e  = RandomForestClassifier()
    model_e.fit(X, A)
    pi = np.clip(model_e.predict_proba(X)[:, 1], EPS, 1 - EPS)

    model_xi = LogisticRegression(max_iter=5000)
    model_xi.fit(np.concatenate((X, A.reshape(-1, 1)), axis=1), Delta)
    xi_0 = model_xi.predict_proba(np.concatenate((X, np.zeros(len(X)).reshape(-1, 1)), axis=1))[:, 1]
    xi_1 = model_xi.predict_proba(np.concatenate((X, np.ones(len(X)).reshape(-1, 1)),  axis=1))[:, 1]

    input_unc = np.concatenate((X_uncensored, A_uncensored.reshape(-1, 1)), axis=1)
    model_nu_unc = RandomForestRegressor()
    model_nu_unc.fit(input_unc, y_uncensored)
    nu_A0_unc = model_nu_unc.predict(np.concatenate((X, np.zeros(len(X)).reshape(-1, 1)), axis=1))
    nu_A1_unc = model_nu_unc.predict(np.concatenate((X, np.ones(len(X)).reshape(-1, 1)),  axis=1))

    aipw_0_unc = ((1-A)*(1-Delta))/(1-pi) * y - (1-A)/(1-pi) * nu_A0_unc * (1-Delta) + nu_A0_unc * (1-xi_0)
    aipw_1_unc = (A*(1-Delta))/pi          * y - A/pi          * nu_A1_unc * (1-Delta) + nu_A1_unc * (1-xi_1)

    if len(X_censored) == 0:
        aipw_0_cen = aipw_1_cen = 0
    else:
        input_cen = np.concatenate((X_censored, A_censored.reshape(-1, 1)), axis=1)
        model_nu_cen = RandomForestRegressor()
        model_nu_cen.fit(input_cen, y_censored)
        nu_A0_cen = model_nu_cen.predict(np.concatenate((X, np.zeros(len(X)).reshape(-1, 1)), axis=1))
        nu_A1_cen = model_nu_cen.predict(np.concatenate((X, np.ones(len(X)).reshape(-1, 1)),  axis=1))
        aipw_0_cen = ((1-A)*Delta)/(1-pi) * y - (1-A)/(1-pi) * nu_A0_cen * Delta + nu_A0_cen * xi_0
        aipw_1_cen = (A*Delta)/pi         * y - A/pi         * nu_A1_cen * Delta + nu_A1_cen * xi_1

    lb_A_0 = aipw_0_unc + aipw_0_cen
    up_A_0 = aipw_0_unc + t_max * xi_0
    lb_A_1 = aipw_1_unc + aipw_1_cen
    up_A_1 = aipw_1_unc + t_max * xi_1
    return lb_A_0, up_A_0, lb_A_1, up_A_1, lb_A_1 - up_A_0, up_A_1 - lb_A_0


def filter_df(df, A):
    filtered_df = df[df['A'] == A]
    if 'NKX2-1' in filtered_df.columns:
        filtered_df = filtered_df.rename(columns={'NKX2-1': 'NKX2_1'})
    return filtered_df

def get_RMST(cph, df):
    t_max = df['y'].max()
    time_points = np.linspace(0, t_max, 100)
    rmst_list = []
    for i in range(len(df)):
        S_t = cph.predict_survival_function(df.iloc[i].drop(columns=['y', 'Delta']), times=time_points).iloc[:, 0].values
        rmst_list.append(np.trapz(S_t, time_points))
    return rmst_list

def cox_model(df):
    x_name = ['Age', 'Sex', 'Smoking_history', 'Pathology', 'N_stage',
               'TP53_alteration', 'SMAD4', 'CDK4', 'NKX2-1', 'MYC_Gain']
    x_name = [n.replace('NKX2-1', 'NKX2_1') for n in x_name]
    df = df.rename(columns={'NKX2-1': 'NKX2_1'})

    X_df = df[x_name]
    df_model = pd.concat([X_df, pd.DataFrame({'A': df['A'], 'y': df['y'], 'Delta': 1 - df['Delta']})], axis=1)

    df_A0 = filter_df(df_model, 0).drop(columns=['A'])
    df_A1 = filter_df(df_model, 1).drop(columns=['A'])

    formula = " + ".join(x_name)

    cph_A0 = CoxPHFitter()
    cph_A0.fit(df_A0, duration_col='y', event_col='Delta', formula=formula)
    A_0_est = np.array(get_RMST(cph_A0, df_model))

    cph_A1 = CoxPHFitter()
    cph_A1.fit(df_A1, duration_col='y', event_col='Delta', formula=formula)
    A_1_est = np.array(get_RMST(cph_A1, df_model))

    return A_0_est, A_1_est
