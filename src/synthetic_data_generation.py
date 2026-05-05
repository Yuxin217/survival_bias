import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

class SyntheticDataModule:
    def __init__(self,  
                n_train=2000,
                censoring_strength=0.3,
                x_dim=1,
                seed=42,
                beta_T=1.0,
                beta_C=1.0,
                ):
        
        self.n_train = n_train
        self.censoring_strength = censoring_strength
        self.x_dim = x_dim
        self.seed = seed
        self.beta_T = beta_T   # frailty effect on T
        self.beta_C = beta_C   # frailty effect on C

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def mu0_function(self, X):
        return (1/3) * (np.sin(12 * X) + X) + (1/60) * np.cos(20 * X)

    # ------------------------------------------------------------------ #
    #  Survival time functions T = tau(x,a)*a + mu0(x) - beta_T*U + eps  #
    # ------------------------------------------------------------------ #

    def sin_response(self, n_samples, A, X, U=None, noise=True):
        """
        tau(x,a) = (sin((x-10)/90 * 2pi) + 1.2) * 10 * a + x
        T = tau(x,a)*a + mu0(x) - beta_T*U + eps
        """
        tau = (np.sin((X - 10) / 90 * 2 * np.pi) + 1.2) * 10 * A + X
        mu_0 = self.mu0_function(X)
        eps = np.random.normal(0, 0.1, n_samples) if noise else 0.0
        U_effect = self.beta_T * U if U is not None else 0.0
        return tau + mu_0 - U_effect + eps

    def exp_response(self, n_samples, A, X, U=None, noise=True):
        """
        tau(x,a) = 20 * exp(a + 0.01*x)
        T = tau(x,a)*a + mu0(x) - beta_T*U + eps
        """
        tau = 20 * np.exp(A + 0.01 * X)
        mu_0 = self.mu0_function(X)
        eps = np.random.normal(0, 0.1, n_samples) if noise else 0.0
        U_effect = self.beta_T * U if U is not None else 0.0
        return A * tau + mu_0 - U_effect + eps

    def logistic_sin_response(self, n_samples, A, X, U=None, noise=True):
        """
        tau(x,a) = 30/(1+exp(-0.1*(x-50))) + 5*sin(0.2*x) + 10
        T = tau(x,a)*a + mu0(x) - beta_T*U + eps
        """
        tau = 30 / (1 + np.exp(-0.1 * (X - 50))) + 5 * np.sin(0.2 * X) + 10
        mu_0 = self.mu0_function(X)
        eps = np.random.normal(0, 0.1, n_samples) if noise else 0.0
        U_effect = self.beta_T * U if U is not None else 0.0
        return A * tau + mu_0 - U_effect + eps

    # ------------------------------------------------------------------ #
    #  Frailty-based censoring: C = c0 * exp(-beta_C * U + eta)          #
    # ------------------------------------------------------------------ #

    def _generate_C(self, U, c0):
        """
        C = c0 * exp(-beta_C * U + eta),  eta ~ N(0, 0.1)
        Sicker patients (larger U) -> smaller C -> more likely censored.
        """
        np.random.seed(self.seed + 20)
        eta = np.random.normal(0, 0.1, len(U))
        return c0 * np.exp(-self.beta_C * U + eta)

    def _calibrate_c0(self, T, U, target_rate, tol=1e-4, max_iter=200):
        """
        Binary search for c0 such that mean(C <= T) ~= target_rate.

        Logic:
          - large c0  →  C large  →  C > T mostly  →  rate = P(C<=T) small
          - small c0  →  C small  →  C < T mostly  →  rate = P(C<=T) large
        So: rate < target  →  need larger rate  →  decrease c0  →  hi = mid
            rate > target  →  need smaller rate →  increase c0  →  lo = mid
        """
        # Use median(T) as natural scale anchor
        T_scale = np.median(T)
        lo, hi = 1e-6 * T_scale, 1e4 * T_scale

        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            C = self._generate_C(U, mid)
            rate = np.mean(C <= T)
            if abs(rate - target_rate) < tol:
                return mid
            if rate < target_rate:   # too little censoring → decrease c0
                hi = mid
            else:                    # too much censoring   → increase c0
                lo = mid
        return mid

    # ------------------------------------------------------------------ #
    #  Propensity score                                                    #
    # ------------------------------------------------------------------ #

    def pi_function(self, X):
        """Observational propensity score sigma((x-55)/45)."""
        X_center = (X - 45) / 45
        probability = self.logistic(X_center)
        return np.clip(probability, 1e-10, 1 - 1e-10)

    # ------------------------------------------------------------------ #
    #  Main data generation (RCT)                                         #
    # ------------------------------------------------------------------ #

    def _generate_data(self, response_type="exp", target_value=0.3):
        """
        Generate RCT data with frailty-based informative censoring.

        DGP:
            X ~ Uniform[10, 100]
            U ~ N(0, 1)                          (latent frailty, unobserved)
            A ~ Bernoulli(0.5)                   (RCT)
            T = response(X, A, U) + eps          (survival time)
            C = c0 * exp(-beta_C * U + eta)      (censoring time)
            T_tilde = min(T, C)
            Delta = 1(C <= T)
        """
        response_fn = {
            "sin":          self.sin_response,
            "exp":          self.exp_response,
            "logistic_sin": self.logistic_sin_response,
        }
        assert response_type in response_fn, f"Unknown response_type: {response_type}"

        rng = np.random.RandomState(self.seed + 1)

        # Covariates & treatment
        self.X = rng.uniform(10, 100, self.n_train)
        self.A = rng.binomial(1, 0.5, self.n_train)

        # Latent frailty (NOT given to the learner)
        self.U = rng.normal(0, 1, self.n_train)

        # Survival time T
        np.random.seed(self.seed + 2)
        T = response_fn[response_type](self.n_train, self.A, self.X, U=self.U, noise=True)

        # Calibrate c0 and generate censoring time C
        c0 = self._calibrate_c0(T, self.U, target_rate=target_value)
        C = self._generate_C(self.U, c0)

        # Observed outcome
        T_tilde = np.minimum(T, C)
        Delta    = (C <= T).astype(int)

        self.Delta = Delta

        # Ground truth (no U effect, no noise, for oracle evaluation)
        y_1        = response_fn[response_type](self.n_train, np.ones(self.n_train),  self.X, U=None, noise=False)
        y_0        = response_fn[response_type](self.n_train, np.zeros(self.n_train), self.X, U=None, noise=False)  

        # Counterfactual T under A=1 and A=0 (with U, no noise)
        T1_cf = response_fn[response_type](self.n_train, np.ones(self.n_train),  self.X, U=self.U, noise=False)
        T0_cf = response_fn[response_type](self.n_train, np.zeros(self.n_train), self.X, U=self.U, noise=False)

        # Oracle lower bound: E[min(T^(a), C) | X] approximated by min(T^(a)_i, C_i)
        y_1_lb = np.minimum(T1_cf, C)
        y_0_lb = np.minimum(T0_cf, C)

        actual_rate = Delta.mean()
        print(f"  [DGP] response={response_type}, target_xi={target_value:.2f}, "
              f"actual_xi={actual_rate:.3f}, c0={c0:.4f}")

        df = pd.DataFrame({
            "X":                  self.X,
            "A":                  self.A,
            "Delta":              Delta,
            "y":                  T_tilde,          # observed time T_tilde = min(T, C)
            "T":                  T,                 # true survival time (for oracle)
            "C":                  C,                 # censoring time (for oracle)
            "y1":                 y_1,
            "y0":                 y_0,
            "y1_minus_y0":        y_1 - y_0,
            # lower-bound oracle: replace censored T with T_tilde
            "y1_censored":        y_1_lb,
            "y0_censored":        y_0_lb,
            "y1_minus_y0_censored": y_1_lb - y_0_lb,
        })

        return df

    # ------------------------------------------------------------------ #
    #  Main data generation (Observational)                               #
    # ------------------------------------------------------------------ #

    def _generate_data_obs(self, response_type="exp", target_value=0.3):
        """
        Generate observational data with frailty-based informative censoring.
        A ~ Bernoulli(sigma((X-55)/45))  (confounded propensity score).
        """
        response_fn = {
            "sin":          self.sin_response,
            "exp":          self.exp_response,
            "logistic_sin": self.logistic_sin_response,
        }
        assert response_type in response_fn

        rng = np.random.RandomState(self.seed + 1)

        self.X = rng.uniform(10, 100, self.n_train)
        self.U = rng.normal(0, 1, self.n_train)

        # Confounded treatment assignment
        p = self.pi_function(self.X)
        self.A = rng.binomial(1, p, self.n_train)

        # Survival time T
        np.random.seed(self.seed + 2)
        T = response_fn[response_type](self.n_train, self.A, self.X, U=self.U, noise=True)

        # Calibrate c0 and generate censoring time C
        c0 = self._calibrate_c0(T, self.U, target_rate=target_value)
        C  = self._generate_C(self.U, c0)

        T_tilde = np.minimum(T, C)
        Delta   = (C <= T).astype(int)
        self.Delta = Delta

        # Ground truth
        y_1 = response_fn[response_type](self.n_train, np.ones(self.n_train),  self.X, U=None, noise=False)
        y_0 = response_fn[response_type](self.n_train, np.zeros(self.n_train), self.X, U=None, noise=False)

        actual_rate = Delta.mean()
        print(f"  [DGP-obs] response={response_type}, target_xi={target_value:.2f}, "
              f"actual_xi={actual_rate:.3f}, c0={c0:.4f}")

        df = pd.DataFrame({
            "X":                  self.X,
            "A":                  self.A,
            "Delta":              Delta,
            "y":                  T_tilde,
            "T":                  T,
            "C":                  C,
            "y1":                 y_1,
            "y0":                 y_0,
            "y1_minus_y0":        y_1 - y_0,
            "y1_censored":        y_1,
            "y0_censored":        y_0,
            "y1_minus_y0_censored": y_1 - y_0,
        })

        return df

    # ------------------------------------------------------------------ #
    #  Visualisation (unchanged interface)                                #
    # ------------------------------------------------------------------ #

    def visualize_data(self, response_type):
        X = np.linspace(10, 100, 100)
        np.random.seed(42)

        response_fn = {
            "sin":          self.sin_response,
            "exp":          self.exp_response,
            "logistic_sin": self.logistic_sin_response,
        }
        fn = response_fn[response_type]

        y_1 = fn(100, np.ones(100),  X, U=None, noise=False)
        y_0 = fn(100, np.zeros(100), X, U=None, noise=False)

        plt.figure(figsize=(12, 8))
        plt.plot(X, y_1,        label='T(A=1)',  color='blue',  alpha=0.7)
        plt.plot(X, y_0,        label='T(A=0)',  color='green', alpha=0.7)
        plt.plot(X, y_1 - y_0,  label='tau(x)', color='red',   alpha=0.7)
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Survival time')
        plt.title(f'{response_type} — oracle survival functions')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs('data/synthetic_data/rct', exist_ok=True)
        plt.savefig(f'data/synthetic_data/rct/{response_type}_function.pdf',
                    bbox_inches='tight', dpi=300)


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    response_types           = ["sin", "exp", "logistic_sin"]
    censoring_strength_array = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    seed_list                = [22, 23, 24, 25, 26, 27]

    for seed in seed_list:
        for censoring_strength in censoring_strength_array:
            data_gen = SyntheticDataModule(
                n_train=2000, censoring_strength=censoring_strength,
                x_dim=1, seed=int(42 + 10 * censoring_strength + seed),
            )
            for response_type in response_types:
                # RCT
                df = data_gen._generate_data(response_type=response_type, target_value=censoring_strength)
                output_dir = f'data/synthetic_data/rct/seed_{seed}'
                os.makedirs(output_dir, exist_ok=True)
                df.to_csv(f'{output_dir}/{response_type}_function_{censoring_strength}.csv', index=False)

                # Observational
                df = data_gen._generate_data_obs(response_type=response_type, target_value=censoring_strength)
                output_dir = f'data/synthetic_data/obs/seed_{seed}'
                os.makedirs(output_dir, exist_ok=True)
                df.to_csv(f'{output_dir}/{response_type}_function_{censoring_strength}.csv', index=False)

        print(f"Saved seed {seed}")
