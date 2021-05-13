"""
Support datasets utils
"""

import numpy as np


def calculate_snr(y_true, noise_std):
    return (20 * np.log10(abs(np.where(noise_std == 0, 0, y_true / noise_std)))).mean()
