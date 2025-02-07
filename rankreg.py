from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
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

def add_gaussian_basis_2d(X, nbases):
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
    def __init__(self, rank=1, alpha=0.0, beta=0):
        """
        Rank-k STRF regression (sklearn-compatible model)

        Args:
            rank (int): the rank of the resulting linear weights
            alpha (float): A hyperparameter for the model.
            beta (float): Another hyperparameter.
        """
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.coefficients_ = None  # Placeholder for model parameters
        self.w_ = None
        self.b_ = None
        self.intercept_ = None

    def fit(self, X, y, verbose=True):
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
        theta = rankreg(X, y, self.rank, alpha=self.alpha, beta=self.beta, verbose=verbose)
        self.coefficients_ = theta['C']
        self.intercept_ = theta['intercept']
        self.w_ = theta['w_h']
        self.b_ = theta['b_h']
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
        
        return response(augment(X, self.rank), self.coefficients_)

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

def rankreg(X, y, rank, mus=None, **args):
    """
    X (array; T x K x K)
    y (array; (T,))
    rank (int)
    params (tuple)
    """
    # Augment and fit
    M = augment(X, rank).transpose(1,2,0)
    # mus = np.tile(mus, rank)
    b_h, w_h = alternating_lsq(M, y, mus, **args)
    theta = extract_theta(b_h, w_h)
    return theta

def extract_theta(b_h, w_h):
    C = b_h[:,None] @ w_h[None,:]
    intercept = w_h[0]*b_h[0]
    w_h = w_h[1:]
    b_h = b_h[1:]
    return {'C': C, 'b_h': b_h, 'w_h': w_h, 'intercept': intercept}

def augment(M, ncopies):
    def block_diag_stack_numpy(X, ncopies):
        """
        return a block-diagonal matrix by stacking N copies of the matrix X
        """
        L, K = X.shape
        B = np.zeros((ncopies*L, ncopies*K))
        for i in range(ncopies):
            B[i*L : (i+1)*L, i*K : (i+1)*K] = X
        return B
    
    def add_bias_entry(X):
        """
        adds a 1 in the upper left corner of M, for fitting the bias
        """
        L, K = X.shape
        B = np.zeros((L+1, K+1)) # Create a zero matrix of shape (T+1, K+1)
        B[0, 0] = 1 # Set the top-left element to 1
        B[1:,1:] = X # Insert M in the bottom-right block
        return B

    T, L, K = M.shape
    result = np.zeros((T, 1+L*ncopies, 1+K*ncopies))
    for t in range(T):
        B = block_diag_stack_numpy(M[t,:,:], ncopies)
        result[t] = add_bias_entry(B)
    return result

def alternating_lsq(M, Y, mus, alpha=0, beta=0, maxiters=1000, tol=1e-6, verbose=True):
    nbases = M.shape[0]
    b_h = np.ones((nbases,))
    last_C = 0
    converged = False
    for i in range(maxiters):
        w_h = fit_w(M, b_h, Y, alpha=alpha)
        b_h = fit_b_fminunc(M, w_h, Y, mus, beta=beta)
        
        # use tol to decide when to stop
        C = b_h[:,None] @ w_h[None,:]
        if np.linalg.norm(C - last_C) < tol:
            if verbose:
                print(f'Stopping after {i} iterations.')
            converged = True
            break
        last_C = C
    if verbose and not converged:
        print(f'Stopped after reaching {maxiters=}.')
    return b_h, w_h

def fit_w(M, b, Y, alpha=0):
    B = mult_b(M, b)
    Reg = alpha * np.eye(B.shape[0])
    return np.linalg.solve(B @ B.T + Reg, B @ Y.T)

def mult_b(M, b):
    return np.einsum('ijk,i->jk', M, b)

def fit_b_fminunc(M, w, Y, mus, beta=0, sig=1.0):
    b0 = fit_b(M, w, Y)
    if beta == 0:
        return b0
    raise Exception("Not implemented.")
    
    W = mult_w(M, w)
    loglike = lambda b: 0.5 / sig**2 * np.sum((Y - b @ W)**2)
    
    S = norm.pdf(np.linspace(M.min(), M.max(), 100), mus)
    D = np.eye(S.shape[0]) # placeholder smoothness prior
    logprior = lambda b: beta / 2 * np.sum((b[1:] @ S.T) @ D @ (b[1:] @ S.T).T)
    
    obj_fun = lambda b: loglike(b) + logprior(b)
    res = minimize(obj_fun, b0, method='BFGS')
    return res.x

def fit_b(M, w, Y):
    W = mult_w(M, w)
    return np.linalg.solve(W @ W.T, W @ Y.T)

def mult_w(M, w):
    return np.einsum('ijk,j->ik', M, w)

def response(M, C):
    return np.einsum('ijk,jk->i', M, C)

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
