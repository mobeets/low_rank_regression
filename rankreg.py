from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import block_diag, qr
import numpy as np

def gaussian_basis(X, mus=None, sigma=None, nbases=10):
    """
    Compute Gaussian basis functions using sklearn's rbf_kernel.

    Args:
        X (np.ndarray): Input feature matrix [T x 1].
        centers (np.ndarray): Centers for the Gaussian basis functions [K x 1].
        sigma (float): Standard deviation for the Gaussian kernels.

    Returns:
        np.ndarray: Basis-expanded feature matrix [T x K].
    """
    if mus is None:
        mus = np.linspace(X.min(), X.max(), nbases).reshape(-1,1)
    if sigma is None:
        sigma = np.mean(np.diff(mus[:,0])) / 2
    gamma = 1 / (2 * sigma**2)  # Convert sigma to RBF gamma parameter
    if len(X.shape) == 1:
        X = X.reshape(-1,1)
    return rbf_kernel(X, mus, gamma=gamma), (mus, sigma)

def add_gaussian_basis_2d(X, nbases=(10,10)):
    """
    Args:
    - X (array, (T,2))
    - nbases (array, (2,))

    Returns array (T,nbases[0],nbases[1]) basis-expanded representation of X
    """
    B1, (mus1, sig1) = gaussian_basis(X[:,0], nbases=nbases[0])
    B2, (mus2, sig2) = gaussian_basis(X[:,1], nbases=nbases[1])
    B = B1[:,:,None] * B2[:,None,:]
    return B, (mus1, sig1, mus2, sig2)

class RankKRegression(BaseEstimator, RegressorMixin):
    def __init__(self, rank=1, alpha_u=0.0, alpha_v=0, max_iter=1000, verbose=False):
        """
        Rank-k STRF regression (sklearn-compatible model)

        Args:
            rank (int): the rank of the resulting linear weights
            alpha_u (float): strength of L2 prior on U
            alpha_v (float): strength of L2 prior on V
        """
        self.rank = rank
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.max_iter = max_iter
        self.verbose = verbose
        self.coefficients_ = None  # Placeholder for model parameters
        self.U_ = None
        self.V_ = None
        self.S_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Train the model.

        Args:
            X (np.ndarray): Feature matrix of shape [n_samples, n_features1, n_features2].
            y (np.ndarray): Target vector of shape [n_samples,].

        Returns:
            self: Fitted model.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        theta = rankreg(X, y, rank=self.rank, alpha_u=self.alpha_u, alpha_v=self.alpha_v, verbose=self.verbose, max_iter=self.max_iter)
        self.coefficients_ = theta['C']
        self.U_ = theta['U']
        self.V_ = theta['V']
        self.S_ = theta['S']
        self.intercept_ = theta['intercept']
        return self

    def predict(self, X):
        """
        Predict using the trained model.

        Args:
            X (np.ndarray): Feature matrix of shape [n_samples, n_features1, n_features2].

        Returns:
            np.ndarray: Predicted values, shape [n_samples,].
        """
        if self.coefficients_ is None:
            raise ValueError("Model is not fitted yet. Call `fit` first.")
        
        return response(X, self.coefficients_) + self.intercept_

    def score(self, X, y):
        """
        Compute R^2 score (coefficient of determination).

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True target values.

        Returns:
            float: R^2 score.
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - ss_residual / ss_total

def response(M, C):
    return np.einsum('tkm,km->t', M, C)

def extract_theta(U, S, V):
    intercept = U[0,0]*V[0,0]
    U = U[1:]
    V = V[1:]
    S = S
    print(U.shape, S.shape, V.shape)
    C = (U @ np.diag(S) @ V.T)
    return {'C': C, 'U': U, 'V': V, 'S': S, 'intercept': intercept}

def rankreg(X, y, rank, alpha_u=0, alpha_v=0, max_iter=1000, tol=1e-6, verbose=True):
    """
    X: array (T,K,M)
    y: array (T,)
    rank: int, desired rank of W where y ≈ sum_ij Xij * Wij + b
    alpha_u: float, strength of ridge prior for left singular vectors of W
    alpha_v: float, strength of ridge prior for right singular vectors of W
    max_iter: int, max iterations for optimization
    tol: float, convergence tolerance
    verbose: bool, for printing optimization details
    
    Returns:
    U: array (K, rank), canonical left singular vectors (orthonormal)
    V: array (M, rank), right singular vectors scaled by singular values
    b: scalar offset term
    """
    # Get X dimensions
    T, K, M = X.shape

    # Augment X with bias term
    X_aug = augment_with_bias(X) # now X has shape (T,K+1,M+1)

    # Initialize random U and V (including entries for bias term)
    U = np.zeros((1+K, rank))
    V = np.zeros((1+M, rank))
    S = np.ones(rank)

    converged = False
    for i in range(max_iter):
        # Solve for V given U (Least Squares with Ridge Regularization)
        XU = np.einsum('tkm,kr->tmr', X_aug, U) # (T, M+1, rank)
        XU_mat = XU.reshape(T, -1)  # Shape (T, (M+1) * rank)
        V_new = np.linalg.solve(
            (XU_mat.T @ XU_mat) + alpha_u*np.eye(XU_mat.shape[1]), # Ensure correct shape
            XU_mat.T @ y,
        ).reshape(M+1, rank)

        # Solve for U given V (Least Squares with Ridge Regularization)
        XV = np.einsum('tkm,mr->tkr', X_aug, V_new)  # Shape (T, K+1, rank)
        XV_mat = XV.reshape(T, -1)  # Shape (T, (K+1) * rank)
        U_new = np.linalg.solve(
            XV_mat.T @ XV_mat + alpha_v*np.eye(XV_mat.shape[1]), # Ensure correct shape
            XV_mat.T @ y,
        ).reshape(K+1, rank)

        # Use SVD to get orthonormal U and V, after removing bias term
        bias = U_new[0,0] * V_new[0,0]
        U_new = U_new[1:] # (K, rank)
        V_new = V_new[1:] # (M, rank)
        U_new, S_new, V_new = np.linalg.svd(U_new @ V_new.T, full_matrices=False)
        U_new = U_new[:,:rank] # (K, rank)
        V_new = V_new[:rank,:].T # (K, rank)
        S_new = S_new[:rank] # (rank,)

        # Add back in bias term
        U_new = np.vstack([np.zeros((1,rank)), U_new]) # (1+K, rank)
        V_new = np.vstack([np.zeros((1,rank)), V_new]) # (1+M, rank)
        # S_new = np.hstack([bias, S_new])
        U_new[0,0] = 1
        V_new[0,0] = bias

        # Check for convergence
        if np.linalg.norm(U_new - U) < tol and np.linalg.norm(V_new - V) < tol:
            converged = True
            break
        U, S, V = U_new, S_new, V_new

    if not converged:
        print(f'WARNING: Stopped after reaching {max_iter=}. Consider increasing max_iter to ensure convergence.')
    return extract_theta(U, S, V)

def augment_with_bias(X):
    T, K, M = X.shape
    result = np.zeros((T, 1+K, 1+M))
    for t in range(T):
        result[t] = block_diag([1], X[t,:,:])
    return result

# Example usage:
if __name__ == "__main__":
    # generate fake data
    nsamples = 1000
    X = np.random.randn(nsamples,2)
    nse = 0.1*np.random.randn(nsamples)
    y = X[:,0]**2 + 0.5*(X[:,1] - 1)**2 + nse

    # fit model
    B, basis_info = add_gaussian_basis_2d(X, nbases=(12,8))
    model = RankKRegression(rank=1)
    model.fit(B, y)
    yhat = model.predict(B)
    print('R^2: {:0.2f}'.format(model.score(B, y)))

    import matplotlib.pyplot as plt

    # plot results
    plt.plot(y, yhat, '.')
    plt.xlabel('True Y')
    plt.ylabel('Predicted Y')
    plt.show()

    # plot coefficients
    plt.plot(basis_info[0], model.b_, '.-', label='feature 1')
    plt.plot(basis_info[2], model.w_, '.-', label='feature 2')
    plt.xlabel('feature value')
    plt.ylabel('feature weight')
    plt.legend()
    plt.show()
