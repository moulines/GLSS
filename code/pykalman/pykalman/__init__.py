'''
=============
Kalman Module
=============

This module provides inference methods for state-space estimation in continuous
spaces.
'''

from .standard import KalmanFilter
from .unscented import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter
from .variant import VariantKalmanFilter

__all__ = [
    "KalmanFilter",
    "VariantKalmanFilter",
    "AdditiveUnscentedKalmanFilter",
    "UnscentedKalmanFilter",
    "datasets",
    "sqrt"
]
