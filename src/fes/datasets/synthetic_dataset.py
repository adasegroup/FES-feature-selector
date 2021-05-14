from typing import Dict, Any
from utils import calculate_snr

import numpy as np
from numpy import random

from kedro.io.core import AbstractDataSet


class SyntheticDataset(AbstractDataSet):

    def __init__(self, option, n, m, noise_std, redundancy_rate, features_fill, poly_degree=None, num_groups=None, seed=None):
        """
        Parameters
        ----------
        option: data type; can be 'sparse' or 'grouped'
        n: number of observations
        m: number of features
        noise_std: observation noise
        redundancy_rate: the rate at which groups/features are disabled (i.e. set to zero)
        features_fill: fill value for groups
        poly_degree: max degree of the polynomial for features
        num_groups: number of groups to split features into (only for 'grouped')
        seed: seed
        -------
        """

        self.option = option

        self.n = n
        self.m = m

        self.noise_std = noise_std
        self.redundancy_rate = redundancy_rate
        self.features_fill = features_fill

        self.poly_degree = poly_degree
        self.num_groups = num_groups

        if seed is not None:
            print(f"The seed for the synthetic dataset generation is set to {seed}", end='\n\n')
            random.seed(seed)

    def _load(self) -> Any:
        if self.option == 'sparse':
            return generate_sparse_data(self.n, self.m, self.noise_std, self.redundancy_rate, self.features_fill, self.poly_degree)

        # TODO. Test when ista_sght will be ready
        # elif self.option == 'grouped':
        #     return generate_grouped_data(self.n, self.m, self.noise_std, self.redundancy_rate, self.features_fill, self.num_groups)

        else:
            raise NotImplementedError

    def _save(self, data: Any) -> None:
        pass

    def _describe(self) -> Dict[str, Any]:
        pass


def generate_sparse_data(n, m, noise_std, redundancy_rate, features_fill, poly_degree):
    """
    Returns y: vector of observations (n,1),
            X: design matrix (n, m)
            w: vector of true coefficients (m,1)
            y_true: vector of noiseless observations (n,1)
            features_mask: (m,1) informative features mask
    -------
    """
    w = np.zeros((m, 1))

    # Decide the number of features and their locations
    num_sparse_feat = np.clip(random.binomial(m, 1 - redundancy_rate), a_min=1, a_max=None)

    sparse_feat_idx = random.choice(m, num_sparse_feat, replace=False)

    # Trim idx to poly_degree
    if poly_degree is not None:
        num_poly = num_sparse_feat // poly_degree
        sparse_feat_idx = sparse_feat_idx[:num_poly * poly_degree].reshape(num_poly, poly_degree)

    # Fill features with values
    if features_fill == 'const':
        w[sparse_feat_idx] = random.randint(1, 4 * m, (num_sparse_feat, 1))

    elif features_fill == 'normal':
        for i in range(poly_degree):
            if i == 0:
                w[sparse_feat_idx[:, i]] = random.standard_normal((num_poly, 1))

            else:
                w[sparse_feat_idx[:, i]] = w[sparse_feat_idx[:, i - 1]] * w[sparse_feat_idx[:, 0]]

    else:
        raise ValueError(f"Unknown fill value: {features_fill}")

    features_mask = w != 0

    # Generate observations
    X = random.standard_normal((n, m))

    y_true = X @ w
    y = y_true + np.random.standard_normal((n, 1)) * noise_std

    print("Synthetic sparse test dataset is generated")
    print(f"Number of observations: {n}, features dim. {m}, number of informative features {sum(features_mask.reshape(-1))}")
    print(f"Observations SNR: {calculate_snr(y_true, noise_std):.3f} dB")
    print(f"Features fill: {features_fill}", end='\n\n')

    return y, X, w, y_true, features_mask


def generate_grouped_data(n, m, noise_std, redundancy_rate, features_fill, num_groups):
    if num_groups is None or num_groups < 2:
        raise ValueError("The number of groups cannot be None or less than 2")

    # Split x_hat into groups
    group_end_idx = random.choice(m - 2, num_groups - 1, replace=False) + 1
    group_end_idx.sort()

    w, groups_labels = np.zeros((m, 1)), np.zeros((m, 1))

    _x_hat, _groups_labels = np.split(w, group_end_idx),\
                             np.split(groups_labels, group_end_idx)

    # Decide whether to keep the group and fill it with values if needed
    for i, (x_hat_group, group_labels) in enumerate(zip(_x_hat, _groups_labels)):
        if i == 0 or random.binomial(1, 1 - redundancy_rate) == 1:
            group_labels[:] = i + 1

            if features_fill == 'const':
                x_hat_group[:] = random.randint(1, 4 * num_groups)

            elif features_fill == 'normal':
                x_hat_group[:] = random.standard_normal(x_hat_group.shape)

            else:
                raise ValueError(f"Unknown fill value: {features_fill}")

    features_mask = w != 0

    # Generate observations
    X = random.standard_normal((n, m))

    y_true = X @ w
    y = y_true + np.random.standard_normal((n)) * noise_std

    return y, X, w, y_true, features_mask, groups_labels