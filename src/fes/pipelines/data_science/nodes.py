import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from fes.methods.iht import l0_reg


def fit_model(y, X):
    """
    Parameters
    ----------
    y: (n,1) vector of observations
    X: (n,m) design matrix
    Returns regressor: fitted regressor compatible with sklearn interface
    -------

    """
    regressor = LinearRegression(fit_intercept=False)
    regressor.fit(X, y)
    return regressor


def evaluate_perm_importance(regressor, y, X, w, y_true, features_mask, parameters):
    """
    Parameters
    ----------
    regressor: fitted regressor compatible with sklearn interface
    y: (n,1) vector of observations
    X: (n,m) design matrix
    w: (m,1) vector of true coefficients
    y_true: (n,1) vector of noiseless observations
    features_mask: (m,1) informative features mask
    parameters
    """
    pi_parameters = {k: parameters[k] for k in parameters["evaluation_params_list"]["perm_importance"]}

    n_repeats = pi_parameters['n_repeats']
    explanation_rate = parameters['explanation_rate']

    print(f"Evaluation on sparse test data with permutation importance", end='\n\n')

    true_num_features = show_oracle_estimate(y, y_true, features_mask)

    results = permutation_importance(regressor, X, y, n_repeats=n_repeats)

    importances_scores = np.random.normal(results.importances_mean, results.importances_std)
    sorted_is_idx = np.argsort(importances_scores)[::-1]

    # Feature selection with known number of informative features
    top_features_idx = sorted_is_idx[:true_num_features]

    w_hat_top = np.zeros_like(w)
    w_hat_top[top_features_idx] = regressor.coef_.transpose()[top_features_idx]

    y_hat_top = X @ w_hat_top

    show_top_k_estimate(true_num_features, y, y_hat_top)

    # Feature selection with unknown number of informative features
    is_cum_sum = np.cumsum(importances_scores[sorted_is_idx])
    features_hat_idx_mask = is_cum_sum <= is_cum_sum[-1] * explanation_rate
    features_hat_idx = sorted_is_idx[features_hat_idx_mask]

    w_hat_er = np.zeros_like(w)
    w_hat_er[features_hat_idx] = regressor.coef_.transpose()[features_hat_idx]

    y_hat_er = X @ w_hat_er

    show_exp_rate_estimate(explanation_rate, features_hat_idx, y, y_hat_er)


def evaluate_iht(y, X, w, y_true, features_mask, parameters):
    """
    Parameters
    ----------
    y: (n,1) vector of observations
    X: (n,m) design matrix
    w: (m,1) vector of true coefficients
    y_true: (n,1) vector of noiseless observations
    features_mask: (m,1) informative features mask
    parameters
    """
    iht_parameters = {k: parameters[k] for k in parameters["evaluation_params_list"]["iht"]}

    explanation_rate = iht_parameters['explanation_rate']

    k = iht_parameters['k']
    tol = iht_parameters['tol']
    max_iter = iht_parameters['max_iter']
    verbose = iht_parameters['verbose']

    print(f"Evaluation on sparse test data with IHT", end='\n\n')

    true_num_features = show_oracle_estimate(y, y_true, features_mask)

    # Feature selection with known number of informative features
    w_hat_top, sup_hat_top = l0_reg(X, y, true_num_features, tol=tol, max_iter=max_iter, verbose=verbose)

    y_hat_top = X @ w_hat_top

    show_top_k_estimate(true_num_features, y, y_hat_top)

    # Feature selection with unknown number of informative features
    w_er, _ = l0_reg(X, y, k, tol=tol, max_iter=max_iter, verbose=verbose)

    sorted_norm_idx = np.argsort(abs(w_er).reshape(-1))[::-1]
    norm_cum_sum = np.cumsum(abs(w_er[sorted_norm_idx]).reshape(-1))
    features_hat_idx_mask = norm_cum_sum <= norm_cum_sum[-1] * explanation_rate
    features_hat_idx = sorted_norm_idx[features_hat_idx_mask]

    w_hat_er = np.zeros_like(w)
    w_hat_er[features_hat_idx] = w_er[features_hat_idx]

    y_hat_er = X @ w_hat_er

    show_exp_rate_estimate(explanation_rate, features_hat_idx, y, y_hat_er)


"""
Support utils
"""


def show_oracle_estimate(y, y_true, features_mask):
    true_num_features = sum(features_mask.reshape(-1))

    true_mse = mean_squared_error(y_true, y)
    true_r2 = r2_score(y_true, y)

    print(f"True number of informative features: {true_num_features}")
    print(f"Best possible approximation due to noise:")
    print(f"{true_mse:.3f} MSE, {true_r2:.3f} R2", end='\n\n')

    return true_num_features


def show_top_k_estimate(k, y, y_hat):
    top_mse = mean_squared_error(y, y_hat)
    top_r2 = r2_score(y, y_hat)

    print(f"Approximation with top {k} features:")
    print(f"{top_mse:.3f} MSE, {top_r2:.3f} R2", end='\n\n')


def show_exp_rate_estimate(explanation_rate, features_hat_idx, y, y_hat):
    mse = mean_squared_error(y, y_hat)
    r2 = r2_score(y, y_hat)

    print(f"Approximation with {explanation_rate} explanation rate:")
    print(f"Number of proposed features: {len(features_hat_idx)}, {mse:.3f} MSE, {r2:.3f} R2", end='\n\n')
