#%% imports

import numpy as np
from rankreg import RankKRegression, add_gaussian_basis_2d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from scipy.stats import norm

#%% generate fake data

rng = np.random.default_rng(seed=999)
T = 1000
X = rng.standard_normal((T,2))
nse = 0.1*rng.standard_normal(T)

# sample a random rank-1 STRF
B, (mus1, sig1, mus2, sig2) = add_gaussian_basis_2d(X)
rank = 2
U_true = norm.pdf(mus1, loc=rng.choice(mus1[1:-1], rank).T, scale=1)
V_true = norm.pdf(mus2, loc=rng.choice(mus2[1:-1], rank).T, scale=1)
S_true = 1 + np.arange(rank)[::-1]
C_true = 5 * (U_true @ np.diag(S_true) @ V_true.T)
y = np.einsum('tkm,km->t', B, C_true) + nse

plt.subplot(2,2,1)
for i in range(rank):
    plt.bar(1+i, S_true[i])
plt.title('feature weights'), plt.xticks(1+np.arange(rank))
plt.subplot(2,2,2)
h = plt.plot(mus1, U_true, '.-'), plt.title('U features')
plt.subplot(2,2,3)
h = plt.plot(mus2, V_true, '.-'), plt.title('V features')
plt.subplot(2,2,4)
plt.imshow(C_true, aspect='auto', origin='lower', cmap='viridis')
plt.xticks([]); plt.yticks([])
plt.xlabel('feature V'), plt.ylabel('feature U'), plt.title('True STRF')
plt.tight_layout()

#%% get basis functions
# improvements to make:
# - computing model likelihoods and BIC?
# - comparing additive vs rank-1 model in simulated data via BIC
# - extension to populations using reduced-rank regression

B, (mus1, sig1, mus2, sig2) = add_gaussian_basis_2d(X, nbases=(10,8))

# fit low rank regression
mdl = RankKRegression(rank=2, alpha_u=0.01, alpha_v=0.01, verbose=True)
mdl.fit(B, y)
yhat = mdl.predict(B)

# check for sign flips that help visualization (since sign is arbitrary)
for r in range(rank):
    if np.mean(mdl.U_[:,r]) < 0 and np.mean(mdl.V_[:,r]) < 0:
        mdl.U_[:,r] = -mdl.U_[:,r]
        mdl.V_[:,r] = -mdl.V_[:,r]

print('R^2: {:0.2f}'.format(mdl.score(B, y)))

# visualize model fits
plt.subplot(2,2,1)
plt.plot(y, yhat, '.'), plt.xlabel('true y'), plt.ylabel('predicted y')
plt.subplot(2,2,2)
h = plt.plot(mus1, mdl.U_, '.-')
plt.subplot(2,2,3)
h = plt.plot(mus2, mdl.V_, '.-')
plt.subplot(2,2,4)
C = mdl.coefficients_
plt.imshow(C, aspect='auto', origin='lower', cmap='viridis')
plt.xticks([]); plt.yticks([]), plt.title(f'Rank-{mdl.rank} STRF')
plt.xlabel('feature V'), plt.ylabel('feature U')
plt.tight_layout()

#%% model comparison

from sklearn.model_selection import KFold

def compare_models(X, y, model, seed=666):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))  # R² score
    return scores

mdl_l = LinearRegression()
mdl_r = Ridge(alpha=1)
mdl_1 = RankKRegression(rank=1, alpha_u=0.01, alpha_v=0.01)
mdl_2 = RankKRegression(rank=2, alpha_u=0.01, alpha_v=0.01)

scores_l = compare_models(B.reshape(B.shape[0], -1), y, mdl_l)
scores_r = compare_models(B.reshape(B.shape[0], -1), y, mdl_r)
scores_1 = compare_models(B, y, mdl_1)
scores_2 = compare_models(B, y, mdl_2)
print(np.median(scores_l), np.median(scores_r), np.median(scores_1), np.median(scores_2))

plt.figure(figsize=(5,5), dpi=300)
plt.subplot(2,2,1)
plt.imshow(C_true, aspect='auto', origin='lower', cmap='viridis'), plt.title('True STRF')
plt.xticks([]); plt.yticks([])
plt.subplot(2,2,2)
plt.imshow(mdl_r.coef_.reshape(B.shape[1],-1), aspect='auto', origin='lower', cmap='viridis'), plt.title('Ridge STRF')
plt.xticks([]); plt.yticks([])
plt.subplot(2,2,3)
plt.imshow(mdl_1.coefficients_, aspect='auto', origin='lower', cmap='viridis'), plt.title(f'Rank-{mdl_1.rank} STRF')
plt.xticks([]); plt.yticks([])
plt.subplot(2,2,4)
plt.imshow(mdl_2.coefficients_, aspect='auto', origin='lower', cmap='viridis'), plt.title(f'Rank-{mdl_2.rank} STRF')
plt.xticks([]); plt.yticks([])
plt.tight_layout()
