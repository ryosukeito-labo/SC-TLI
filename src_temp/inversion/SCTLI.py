import numpy as np


def soft_thresholding(x, threshold):
    """
    Soft-thresholding operator (proximal operator of L1 norm).
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


class ADMMGroupLasso:
    """
    ADMM-based solver for Spatio-Temporal L1-L2 regularization with group sparsity.

    This implementation solves the following optimization problem:

        minimize 1/2 ||Xβ - f||^2
               + λ_s [ α ||β||_1 + (1 - α)/2 ||β||_2^2 ]
               + λ_t Σ ||(Dβ)_G||_2

    where:
        - The first term is the data misfit.
        - The second term is the elastic-net penalty in space.
        - The third term enforces structural coupling in time via group sparsity.

    Parameters
    ----------
    lambda_s : float
        Spatial regularization parameter.

    lambda_t : float
        Temporal (group) regularization parameter.

    alpha : float
        Balance between L1 and L2 (0 ≤ alpha ≤ 1).

    rho : float
        ADMM penalty parameter for s-variable.

    eta : float
        ADMM penalty parameter for t-variable.

    tol : float
        Convergence tolerance.

    max_iter : int
        Maximum number of iterations.
    """

    def __init__(self,
                 lambda_s=1.0,
                 lambda_t=1.0,
                 alpha=0.9,
                 rho=1.0,
                 eta=1.0,
                 tol=1.e-3,
                 max_iter=100000):

        self.lambda_s = lambda_s
        self.lambda_t = lambda_t
        self.alpha = alpha
        self.rho = rho
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter

        # Model
        self.beta_ = None

        # ADMM variables
        self.s_ = None
        self.t_ = None

        # Dual variables
        self.u_ = None
        self.v_ = None

    def fit(self, f, X, D, inv1, inv2, times, num_of_cells):
        """
        Run ADMM optimization.

        Parameters
        ----------
        f : ndarray (n,)
            Observed data.

        X : ndarray (n, m)
            Forward operator.

        D : ndarray (m, m) or linear operator
            Temporal difference operator.

        inv1, inv2 : ndarray or operator
            Precomputed matrices for fast inversion (Woodbury-type acceleration).

        times : int
            Number of time steps.

        num_of_cells : int
            Number of spatial cells per time step.

        Returns
        -------
        beta_ : ndarray (m,)
            Estimated model vector.
        """

        n, m = X.shape

        # ---------------------
        # Initialization
        # ---------------------
        self.beta_ = np.zeros(m)
        self.s_ = np.zeros(m)
        self.t_ = np.zeros(D.shape[0])
        self.u_ = np.zeros(m)
        self.v_ = np.zeros(D.shape[0])

        for i in range(self.max_iter):

            # ---------------------
            # β-update
            # ---------------------
            beta_prev = self.beta_.copy()

            p = (
                X.T.dot(f)
                + self.rho * (self.s_ + self.u_)
                + self.eta * D.T * (self.t_ + self.v_)
            )

            tmp = np.dot(X, inv1 * p)
            self.beta_ = inv1 * p - inv1 * np.dot(X.T, np.dot(inv2, tmp))

            # ---------------------
            # s-update (Elastic Net)
            # ---------------------
            s_prev = self.s_.copy()

            self.s_ = (
                self.eta / (self.lambda_s * (1 - self.alpha) + self.eta)
                * soft_thresholding(
                    self.beta_ - self.u_,
                    self.lambda_s * self.alpha / self.eta
                )
            )

            # ---------------------
            # t-update (Group Lasso)
            # ---------------------
            t_prev = self.t_.copy()

            q = D * self.beta_ - self.v_

            # Compute group norms
            r = np.zeros(num_of_cells)

            for j in range(times - 1):
                block = np.power(q[j*num_of_cells:(j+1)*num_of_cells], 2)
                r += block

            r = np.tile(np.sqrt(r), times-1)

            # Group shrinkage
            shrink = np.maximum(1.0 - self.lambda_t / (self.rho * r), 0.0)
            self.t_ = shrink * q

            # ---------------------
            # Dual updates
            # ---------------------
            self.u_ += self.rho * (self.s_ - self.beta_)
            self.v_ += self.eta * (self.t_ - D * self.beta_)

            # ---------------------
            # Convergence check
            # ---------------------
            dr_s = self.s_ - self.beta_
            ds_s = self.rho * (self.s_ - s_prev)

            dr_t = self.t_ - D * self.beta_
            ds_t = self.eta * (self.t_ - t_prev)

            delta_s = max(
                np.sqrt(np.linalg.norm(dr_s, ord=2) ** 2 / m),
                np.sqrt(np.linalg.norm(ds_s, ord=2) ** 2 / m),
            )

            delta_t = max(
                np.sqrt(np.linalg.norm(dr_t, ord=2) ** 2 / D.shape[0]),
                np.sqrt(np.linalg.norm(ds_t, ord=2) ** 2 / D.shape[0]),
            )

            delta = max(delta_s, delta_t)

            if i % 1000 == 0:
                print(f"iter = {i}, delta = {delta}")

            if delta <= self.tol:
                print(f"Converged at iteration {i}")
                break