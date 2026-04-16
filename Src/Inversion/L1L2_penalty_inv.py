import numpy as np

class L1L2:
    """
    L1-L2 regularized inversion using ADMM.

    This class solves the following optimization problem:

        minimize 1/2 * ||Xβ - f||^2
               + l_1 * ||β||_1
               + (l_2 / 2) * ||β||_2^2

    Optionally, bound constraints and intercept estimation can be included.

    Parameters
    ----------
    l_1 : float
        L1 regularization parameter (sparsity control).

    l_2 : float
        L2 regularization parameter (smoothness control).

    rho_ : float
        ADMM penalty parameter for z-variable.

    eta_ : float
        ADMM penalty parameter for bound constraints.

    min_ : float
        Lower bound constraint for model parameters.

    max_ : float
        Upper bound constraint for model parameters.

    fit_intercept : bool
        If True, estimate intercept term b.

    eps_ADMM : float
        Convergence threshold.

    max_iter : int
        Maximum number of ADMM iterations.
    """

    def __init__(self, l_1, l_2, rho_, eta_,
                 min_=-np.inf, max_=np.inf,
                 fit_intercept=False,
                 eps_ADMM=1.e-3,
                 max_iter=100000):

        self.l_1 = l_1
        self.l_2 = l_2
        self.rho_ = rho_
        self.eta_ = eta_

        self.min_ = min_
        self.max_ = max_

        self.fit_intercept = fit_intercept
        self.eps_ADMM = eps_ADMM
        self.max_iter = max_iter

        # Model parameters
        self.beta_ = None  # model vector
        self.b = None      # intercept

        # ADMM auxiliary variables
        self.z_ = None     # for L1 regularization
        self.y_ = None     # for bound constraint

        # Dual variables
        self.v_ = None
        self.u_ = None

    @staticmethod
    def soft_thresholding(x, threshold):
        """Soft-thresholding operator (proximal operator of L1 norm)."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

    def algorithm(self, X, f, w):
        """
        Run ADMM optimization.

        Parameters
        ----------
        X : ndarray (n, m)
            Forward operator / design matrix.

        f : ndarray (n,)
            Observed data.

        w : ndarray (m,)
            Weight vector (used when intercept is included).
        """

        n, m = X.shape

        # Initialization
        self.beta_ = np.zeros(m)
        self.z_ = np.zeros(m)
        self.v_ = np.zeros(m)
        self.u_ = np.zeros(m)
        self.y_ = np.zeros(m)

        if self.fit_intercept:
            self.b = 0.0
            weight_ = w
            c = np.dot(X, weight_)
        else:
            print("intercept = False")

        XXT = X.dot(X.T)

        # =========================
        # CASE 1: No constraint
        # =========================
        if self.eta_ == 0:

            print("no-constrained")

            Inv = np.linalg.inv(np.eye(n) + XXT / self.rho_)

            for it in range(self.max_iter):

                beta_prev = self.beta_.copy()

                # ---------------------
                # Step 1: Update beta
                # ---------------------
                if not self.fit_intercept:
                    C = np.dot(X.T, f) + self.rho_ * (self.z_ + self.v_)
                else:
                    C = np.dot(X.T, f - self.b * c) + self.rho_ * (self.z_ + self.v_)

                self.beta_ = (
                    C / self.rho_
                    - np.dot(X.T, np.dot(Inv, np.dot(X, C))) / (self.rho_**2)
                )

                # ---------------------
                # Step 2: Update intercept
                # ---------------------
                if self.fit_intercept:
                    self.b = np.dot(c.T, f - np.dot(X, self.beta_)) / np.dot(c.T, c)

                # ---------------------
                # Step 3: Update z
                # ---------------------
                z_prev = self.z_.copy()

                if not self.fit_intercept:
                    self.z_ = self.soft_thresholding(
                        self.rho_ * (self.beta_ - self.v_), self.l_1
                    ) / (self.l_2 + self.rho_)
                else:
                    self.z_ = self.soft_thresholding(
                        self.beta_ - self.v_, self.l_1 / self.rho_
                    ) * self.rho_ / (self.l_2 + self.rho_)

                # ---------------------
                # Step 4: Dual update
                # ---------------------
                if not self.fit_intercept:
                    self.v_ += self.rho_ * (self.z_ - self.beta_)
                else:
                    self.v_ += (self.z_ - self.beta_)

                # ---------------------
                # Convergence check
                # ---------------------
                dr = self.z_ - self.beta_
                ds = self.rho_ * (self.z_ - z_prev)

                delta_r = np.linalg.norm(dr) / np.sqrt(m)
                delta_s = np.linalg.norm(ds) / np.sqrt(m)

                delta = max(delta_r, delta_s)

                if it % 1000 == 0:
                    print(f"iter = {it}, delta = {delta}")

                if delta <= self.eps_ADMM:
                    print(f"Converged at iteration {it}")
                    break

        # =========================
        # CASE 2: With constraint
        # =========================
        else:

            print("constrained")

            Inv = np.linalg.inv(np.eye(n) + XXT / (self.rho_ + self.eta_))

            for it in range(self.max_iter):

                beta_prev = self.beta_.copy()

                # ---------------------
                # Step 1: Update beta
                # ---------------------
                if not self.fit_intercept:
                    D = (
                        np.dot(X.T, f)
                        + self.rho_ * (self.z_ + self.v_)
                        + self.eta_ * (self.y_ + self.u_)
                    )
                else:
                    D = (
                        np.dot(X.T, f - self.b * c)
                        + self.rho_ * (self.z_ + self.v_)
                        + self.eta_ * (self.y_ - self.b * w + self.u_)
                    )

                denom = (self.rho_ + self.eta_)
                self.beta_ = (
                    D / denom
                    - np.dot(X.T, np.dot(Inv, np.dot(X, D))) / (denom**2)
                )

                # ---------------------
                # Step 2: Update intercept
                # ---------------------
                if self.fit_intercept:
                    self.b = (
                        np.dot(c.T, f - np.dot(X, self.beta_))
                        + self.eta_ * np.dot(w.T, self.y_ - self.beta_ + self.u_)
                    ) / (np.dot(c.T, c) + self.eta_ * np.dot(w.T, w))

                # ---------------------
                # Step 3: Update z (L1)
                # ---------------------
                z_prev = self.z_.copy()

                self.z_ = self.soft_thresholding(
                    self.beta_ - self.v_, self.l_1 / self.rho_
                ) * self.rho_ / (self.l_2 + self.rho_)

                # ---------------------
                # Step 4: Update y (bounds)
                # ---------------------
                y_prev = self.y_.copy()

                if not self.fit_intercept:
                    self.y_ = np.maximum(self.beta_ - self.u_, self.min_)
                else:
                    self.y_ = np.maximum(
                        self.min_ * w,
                        np.minimum(self.beta_ + self.b * w - self.u_, self.max_ * w)
                    )

                # ---------------------
                # Step 5: Dual updates
                # ---------------------
                self.v_ += self.rho_ * (self.z_ - self.beta_)

                if not self.fit_intercept:
                    self.u_ += self.eta_ * (self.y_ - self.beta_)
                else:
                    self.u_ += self.eta_ * (self.y_ - (self.beta_ + self.b * w))

                # ---------------------
                # Convergence check
                # ---------------------
                dr = self.z_ - self.beta_
                ds = self.rho_ * (self.z_ - z_prev)

                delta = max(
                    np.linalg.norm(dr) / np.sqrt(m),
                    np.linalg.norm(ds) / np.sqrt(m),
                )

                if not self.fit_intercept:
                    dr_bc = self.y_ - self.beta_
                else:
                    dr_bc = self.y_ - (self.beta_ + self.b * w)

                ds_bc = self.eta_ * (self.y_ - y_prev)

                delta_bc = max(
                    np.linalg.norm(dr_bc) / np.sqrt(m),
                    np.linalg.norm(ds_bc) / np.sqrt(m),
                )

                if it % 1000 == 0:
                    print(f"delta = {delta}, delta_BC = {delta_bc}")

                if max(delta, delta_bc) <= self.eps_ADMM:
                    print(f"Converged at iteration {it}")
                    break