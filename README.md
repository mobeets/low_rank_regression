# Low-rank Regression

This code estimates models described in _Inferring input nonlinearities in neural encoding models_ (Ahrens et al., 2008), but with what seems to be a more robust/identifiable implementation.

## Overview

Suppose you have 2D covariates $X \in \mathbb{R}^{L \times M}$, and observations $y \in \mathbb{R}$, and you would like to find weights $W \in \mathbb{R}^{L \times M}$ where:

$$ y \approx \sum_{ij} X_{ij} W_{ij} + \epsilon $$ 

where $\epsilon$ is Gaussian noise.

One possibility is that $W$ is low-rank, meaning it can be written as $W = U S V^\top$, where $U \in \mathbb{R}^{L \times K}$, $V \in \mathbb{R}^{M \times K}$, and $S$ is a diagonal matrix, for some rank $K$. For example, if $W$ is rank-1, it can be written as $W = \boldsymbol{u} s \boldsymbol{v}^\top$ for vectors $u$ and $v$.

This code will perform low-rank linear regression for this setting.

__Note: This is distinct from Reduced Rank Regression, which is when we have $Y$, a vector of observations, rather than a scalar $y$, as we do here.__

## Quickstart

```python
# load data:
# - X is an array with shape (T, 2)
# - y is an array with shape (T,)

# represent X using gaussian basis functions for X[:,0] and X[:,1]
B, basis_params = add_gaussian_basis_2d(X, nbases=(10,10)) # here we use K=10, M=10 basis functions
# B is an array with shape (T, K, M), where K and M are the number of basis functions for X[:,0] and X[:,1], respectively
# basis_params contains the basis function parameters: (mus1, sig1, mus2, sig2); you can also supply these to add_gaussian_basis_2d
# alternatively, if X is discrete, with K possible values in X[:,0], and M possible values in X[:,1], then instead of using basis functions we want B to be a discrete representation of X, with shape (T, K, M)

# create and fit a rank-1 model
mdl = RankKRegression(rank=1, alpha_u=0.01, alpha_v=0.01) # the hyperparameters alpha_u and alpha_v act as regularizers
mdl.fit(B, y) # find W such that y(t) ≈ B(t) * W
yhat = mdl.predict(B) # get predictions
```

## Simulations

To illustrate, below we sample data using a constructed W which is rank-2:

<img width="50%" src="plots/example_truth.png"/>

W, or the "STRF", is the outer product of the vectors in U and V.

We then use the resulting simulated data to estimate W using Rank-2 regression:

<img width="50%" src="plots/example_rank2.png"/>

We can compare the estimated weights using Ridge, Rank-1, and Rank-2 regression:

<img width="50%" src="plots/example_compare.png"/>

Note that the Ridge regression can also fit the data well, but the resulting weights are _not_ low rank.
