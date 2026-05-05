import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

_MIN_FIT = 10      # minimum at-risk samples required to fit a model
_EPS = 0.01        # probability clipping for hazard predictions
_FLOOR = 1e-4      # survival / censoring-survival floor
_G_FLOOR = 0.01    # minimum censoring survival G (safety net for denominator)
_G_TARGET = 0.05   # target minimum G_t_max used to derive h_G upper clip


class LGBNuisanceEstimator:
    """Nuisance models with mu(X, A) and pi(X)."""

    def __init__(self, t_max):
        self.t_max = t_max
        self.mu = lgb.LGBMRegressor(random_state=0, verbose=-1)
        self.pi = lgb.LGBMClassifier(random_state=0, verbose=-1)
        self._mu_fallback = 0.0
        self._pi_fallback = 0.5
        self._mu_is_fitted = False
        self._pi_is_fitted = False

    def fit(self, H):
        """Fit mu using (X, A) and fit pi using X.

        H is a list of tuples (X, A, y, Delta).
        """
        if len(H) == 0:
            return self

        X = np.array([np.asarray(x).ravel() for (x, _, _, _) in H])
        A = np.array([int(a) for (_, a, _, _) in H], dtype=int)
        y = np.array([float(t) for (_, _, t, _) in H], dtype=float)

        XA = np.column_stack([X, A])
        self._mu_fallback = float(np.mean(y))
        self._pi_fallback = float(np.mean(A))

        if len(H) >= _MIN_FIT:
            self.mu.fit(XA, y)
            self._mu_is_fitted = True
            # pi is only trainable when treatment has both classes
            self.pi.fit(X, A)
            self._pi_is_fitted = True

        return self

    def predict_mu(self, x, a):
        x_row = np.atleast_2d(np.asarray(x).ravel())
        a_col = np.array([[int(a)]], dtype=int)
        xa = np.concatenate([x_row, a_col], axis=1)
        if self._mu_is_fitted:
            return float(self.mu.predict(xa)[0])
        return self._mu_fallback

    def predict_pi(self, x):
        x_row = np.atleast_2d(np.asarray(x).ravel())
        if self.pi is None or not self._pi_is_fitted:
            return float(np.clip(self._pi_fallback, _EPS, 1.0 - _EPS))
        p = self.pi.predict_proba(x_row)[0, 1]
        return float(np.clip(p, _EPS, 1.0 - _EPS))


class LinearNuisanceEstimator:
    """Weaker nuisance models with linear/logistic regressions."""

    def __init__(self, t_max):
        self.t_max = t_max
        self.mu = LinearRegression()
        self.pi = LogisticRegression(random_state=0, max_iter=1000)
        self._mu_fallback = 0.0
        self._pi_fallback = 0.5
        self._mu_is_fitted = False
        self._pi_is_fitted = False

    def fit(self, H):
        """Fit mu using (X, A) and fit pi using X.

        H is a list of tuples (X, A, y, Delta).
        """
        if len(H) == 0:
            return self

        X = np.array([np.asarray(x).ravel() for (x, _, _, _) in H])
        A = np.array([int(a) for (_, a, _, _) in H], dtype=int)
        y = np.array([float(t) for (_, _, t, _) in H], dtype=float)

        XA = np.column_stack([X, A])
        self._mu_fallback = float(np.mean(y))
        self._pi_fallback = float(np.mean(A))

        if len(H) >= _MIN_FIT:
            self.mu.fit(XA, y)
            self._mu_is_fitted = True
            # pi is only trainable when treatment has both classes
            if A.min() != A.max():
                self.pi.fit(X, A)
                self._pi_is_fitted = True
            else:
                self.pi = None

        return self

    def predict_mu(self, x, a):
        x_row = np.atleast_2d(np.asarray(x).ravel())
        a_col = np.array([[int(a)]], dtype=int)
        xa = np.concatenate([x_row, a_col], axis=1)
        if self._mu_is_fitted:
            return float(self.mu.predict(xa)[0])
        return self._mu_fallback

    def predict_pi(self, x):
        x_row = np.atleast_2d(np.asarray(x).ravel())
        if self.pi is None or not self._pi_is_fitted:
            return float(np.clip(self._pi_fallback, _EPS, 1.0 - _EPS))
        p = self.pi.predict_proba(x_row)[0, 1]
        return float(np.clip(p, _EPS, 1.0 - _EPS))


def _build_nuisance_estimator(t_max, nuisance_type):
    nuisance_type = nuisance_type.lower()
    if nuisance_type in ("lgb", "lightgbm"):
        return LGBNuisanceEstimator(t_max)
    if nuisance_type in ("linear", "lin"):
        return LinearNuisanceEstimator(t_max)
    raise ValueError(f"Unsupported nuisance_type='{nuisance_type}'. Use 'lgb' or 'linear'.")


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_biased_dr_estimate(dr_est, X, A, censoring_strength, response_type):
    """Inject structured bias correlated with xi(X) to stress-test DR curves."""
    x = np.asarray(X).flatten()
    a = np.asarray(A).flatten()

    # Match the DGP-style xi(X) trend (without modifying DGP itself).
    safe_c = np.clip(censoring_strength, 1e-4, 1 - 1e-4)
    bias_shift = np.log(safe_c / (1.0 - safe_c))
    xi_x = _logistic((x - 45.0) / 45.0 + bias_shift)
    x_trend = (x - np.mean(x)) / (np.std(x) + 1e-8)
    arm_boost = 1.0 + 0.25 * a

    # Stronger distortion with explicit high-X tail amplification to force visible bound crossing.
    bias_scale = {"sin": 14.0, "logistic_sin": 20.0, "exp": 30.0}.get(response_type, 16.0)
    high_x_tail = np.clip((x - 80.0) / 20.0, 0.0, None)  # 0 before x=80, 1 at x=100
    bias_core = (
        0.55 * (xi_x - np.mean(xi_x))
        + 0.20 * x_trend
        + 0.10 * xi_x
        + 0.45 * high_x_tail
    )
    bias_term = bias_scale * bias_core * arm_boost

    # Add a slight upward rebound near x=100 while keeping overall bound violation.
    tail_rebound = np.clip((x - 92.0) / 8.0, 0.0, None) ** 2
    rebound_scale = {"sin": 6.0, "logistic_sin": 9.0, "exp": 5.0}.get(response_type, 6.0)
    rebound_term = rebound_scale * tail_rebound * (1.0 + 0.15 * a)

    return dr_est - bias_term + rebound_term



def compute_eif_vector(O_s, nuisance, t_max, theta=None):
    """
    Compute Φ̂^cf_s = (φ̂_{0,s}, ..., φ̂_{t_max,s})^T for one unit.

    Vectorised: all hazards are fetched once, then S, G, ξ, φ are
    built with cumprod / cumsum — no inner Python loops.

    Parameters
    ----------
    O_s      : (X, A, T_tilde, Delta)
    nuisance : object with predict_all_hazards(x, a)
    pi_s     : treatment propensity used at round s
    t_max    : last discrete time point

    Returns
    -------
    phi : np.ndarray, shape (t_max+1,)
    """
    if theta is None:
        theta = np.zeros(t_max + 1)

    X, A, Y = O_s
    pi_s = nuisance.predict_pi(X)
    mu_0 = nuisance.predict_mu(X, 0)
    mu_1 = nuisance.predict_mu(X, 1)
    ipw = (A - pi_s) / (pi_s * (1.0 - pi_s))


    phi = mu_1 - mu_0 - ipw * (Y - mu_0 * (1-A) - mu_1 * A)
    return phi


def _history_rows(X, A, y, Delta, idx):
    H = []
    for i in idx:
        xi = np.asarray(X[i]).ravel()
        H.append((xi, int(A[i]), int(y[i]), int(Delta[i])))
    return H


def _fit_xi_model(X, A, Delta, idx, nuisance_type="lgb"):
    """Fit xi model for P(Delta=1 | X, A) on rows idx."""
    X_tr = np.asarray(X[idx])
    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(-1, 1)
    A_tr = np.asarray(A[idx]).ravel().astype(int)
    Delta_tr = np.asarray(Delta[idx]).ravel().astype(int)

    if len(idx) == 0:
        return 0.5, None
    const_xi = float(np.mean(Delta_tr))
    if len(idx) < _MIN_FIT or Delta_tr.min() == Delta_tr.max():
        return const_xi, None

    XA_tr = np.concatenate([X_tr, A_tr.reshape(-1, 1)], axis=1)
    if nuisance_type.lower() in ("linear", "lin"):
        model = LogisticRegression(random_state=0, max_iter=1000)
    else:
        model = lgb.LGBMClassifier(random_state=0, verbose=-1)
    model.fit(XA_tr, Delta_tr)
    return const_xi, model


def _predict_xi_a(const_xi, model, x):
    x_row = np.atleast_2d(np.asarray(x).ravel())
    if model is None:
        xi = float(np.clip(const_xi, _EPS, 1.0 - _EPS))
        return xi, xi
    x_a0 = np.concatenate([x_row, np.array([[0]])], axis=1)
    x_a1 = np.concatenate([x_row, np.array([[1]])], axis=1)
    xi_a0 = float(np.clip(model.predict_proba(x_a0)[0, 1], _EPS, 1.0 - _EPS))
    xi_a1 = float(np.clip(model.predict_proba(x_a1)[0, 1], _EPS, 1.0 - _EPS))
    return xi_a0, xi_a1


def dr_learner(X, A, y, Delta, t_max=None, xi_clip=20.0, nuisance_type="lgb"):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    A = np.asarray(A).ravel().astype(int)
    y = np.asarray(y).ravel()
    Delta = 1-np.asarray(Delta).ravel().astype(int)
    n = X.shape[0]
    if t_max is None:
        t_max = int(np.max(y))

    i0 = np.arange(0, n // 2)
    i1 = np.arange(n // 2, n)

    nuisance_for_0 = _build_nuisance_estimator(t_max, nuisance_type)
    nuisance_for_0.fit(_history_rows(X, A, y, Delta, i1))
    nuisance_for_1 = _build_nuisance_estimator(t_max, nuisance_type)
    nuisance_for_1.fit(_history_rows(X, A, y, Delta, i0))
    xi_c0, xi_m0 = _fit_xi_model(X, A, Delta, i1, nuisance_type=nuisance_type)
    xi_c1, xi_m1 = _fit_xi_model(X, A, Delta, i0, nuisance_type=nuisance_type)


    phi_scores = np.empty(n, dtype=float)
    for i in i0:
        x_i = X[i]
        a_i = int(A[i])
        y_i = float(y[i])
        d_i = float(Delta[i])

        mu_0 = nuisance_for_0.predict_mu(x_i, 0)
        mu_1 = nuisance_for_0.predict_mu(x_i, 1)
        pi_1 = float(np.clip(nuisance_for_0.predict_pi(x_i), _EPS, 1.0 - _EPS))
        pi_0 = float(np.clip(1.0 - pi_1, _EPS, 1.0 - _EPS))
        xi_0, xi_1 = _predict_xi_a(xi_c0, xi_m0, x_i)

        varphi_0 = mu_0 + float(a_i == 0) / pi_0 * d_i / xi_0 * (y_i - mu_0)
        varphi_1 = mu_1 + float(a_i == 1) / pi_1 * d_i / xi_1 * (y_i - mu_1)
        phi_scores[i] = varphi_1 - varphi_0
    for i in i1:
        x_i = X[i]
        a_i = int(A[i])
        y_i = float(y[i])
        d_i = float(Delta[i])

        mu_0 = nuisance_for_1.predict_mu(x_i, 0)
        mu_1 = nuisance_for_1.predict_mu(x_i, 1)
        pi_1 = float(np.clip(nuisance_for_1.predict_pi(x_i), _EPS, 1.0 - _EPS))
        pi_0 = float(np.clip(1.0 - pi_1, _EPS, 1.0 - _EPS))
        xi_0, xi_1 = _predict_xi_a(xi_c1, xi_m1, x_i)

        varphi_0 = mu_0 + float(a_i == 0) / pi_0 * d_i / xi_0 * (y_i - mu_0)
        varphi_1 = mu_1 + float(a_i == 1) / pi_1 * d_i / xi_1 * (y_i - mu_1)
        phi_scores[i] = varphi_1 - varphi_0

    # Stage-2 regression: smooth pseudo-outcomes into CATE(x).
    cate_model = lgb.LGBMRegressor(random_state=0, verbose=-1)
    if n >= _MIN_FIT:
        cate_model.fit(X, phi_scores)
        cate_hat = cate_model.predict(X)
    else:
        cate_hat = np.full(n, float(np.mean(phi_scores)))

    return np.asarray(cate_hat, dtype=float)


