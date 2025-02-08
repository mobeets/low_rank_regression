#%% imports

import numpy as np
from rankreg import RankKRegression, add_gaussian_basis_2d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

#%% generate fake data

rng = np.random.default_rng(seed=234)
T = 1000
X = rng.standard_normal((T,2))
nse = 0.5*rng.standard_normal(T)

# sample a random rank-1 STRF
B, _ = add_gaussian_basis_2d(X)
rank = 2
C_true = 0
for i in range(rank):
    b_true = np.sort(rng.standard_normal(B.shape[1]))
    w_true = np.sort(rng.standard_normal(B.shape[2]))
    C_true += b_true[:,None] @ w_true[None,:]
y = np.einsum('ijk,jk->i', B, C_true) + nse

plt.imshow(C_true, aspect='auto', origin='lower', cmap='viridis'), plt.colorbar(), plt.title('True STRF'), plt.show()

#%% get basis functions

B, (mus1, sig1, mus2, sig2) = add_gaussian_basis_2d(X, nbases=(10,10))

# fit low rank regression
mdl = RankKRegression(rank=1, alpha1=0.0, max_iter=10000)
mdl.fit(B, y)
yhat = mdl.predict(B)
mdl2 = RankKRegression(rank=1, alpha1=0.0, max_iter=10000)
mdl2.fit(B, y - yhat)
yhat2 = mdl2.predict(B)
C = mdl.coefficients_[1:,1:] + mdl2.coefficients_[1:,1:]

print('R^2: {:0.2f}'.format(mdl.score(B, y)))

# visualize model fits
plt.subplot(2,2,1)
plt.plot(y, yhat, '.'), plt.xlabel('true y'), plt.ylabel('predicted y')
plt.subplot(2,2,2)
h = plt.plot(mus1, mdl.b_, '.-'), plt.xlabel('feature 1')
plt.subplot(2,2,3)
h = plt.plot(mus2, mdl.w_, '.-'), plt.xlabel('feature 2')
plt.subplot(2,2,4)
# C = mdl.coefficients_[1:,1:]
plt.imshow(C, aspect='auto', origin='lower', cmap='viridis')
plt.xticks([]); plt.yticks([])
plt.colorbar()
plt.xlabel('feature 2'), plt.ylabel('feature 1')
plt.tight_layout()

#%% model comparison

from sklearn.model_selection import KFold

def compare_models(X, y, model, seed=42):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))  # R² score
    return scores

mdl_l = LinearRegression()
mdl_r = Ridge(alpha=0.1)
mdl = RankKRegression(rank=1, alpha1=0.0, alpha2=0.0, max_iter=5000)

scores_l = compare_models(B.reshape(B.shape[0], -1), y, mdl_l)
scores_r = compare_models(B.reshape(B.shape[0], -1), y, mdl_r)
scores = compare_models(B, y, mdl)
# print(np.median(scores_l), np.median(scores_r), np.median(scores))

plt.figure(figsize=(9,3), dpi=300)
plt.subplot(1,3,1)
plt.imshow(C_true, aspect='auto', origin='lower', cmap='viridis'), plt.colorbar(), plt.title('True STRF')
plt.xticks([]); plt.yticks([])
plt.subplot(1,3,2)
plt.imshow(mdl.coefficients_[1:,1:].reshape(nbases[0],-1), aspect='auto', origin='lower', cmap='viridis'), plt.colorbar(), plt.title('Rank-1 STRF')
plt.xticks([]); plt.yticks([])
plt.subplot(1,3,3)
plt.imshow(mdl_r.coef_.reshape(nbases[0],-1), aspect='auto', origin='lower', cmap='viridis'), plt.colorbar(), plt.title('Ridge STRF')
plt.xticks([]); plt.yticks([])
plt.tight_layout()
