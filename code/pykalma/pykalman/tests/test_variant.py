from variant import VariantKalmanFilter
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt

T = 100
sigma2eta = 1
eta = np.random.randn(T)*np.sqrt(sigma2eta)
offsets = np.ones(T)*0
states = np.cumsum(eta+offsets)
#print(len(states))

sigma2eps = 10
eps = np.random.randn(T)*np.sqrt(sigma2eps)
measurements = states + eps

# plt.plot(np.arange(T),measurements, label='measurements')
# plt.plot(np.arange(T),states, label='states')
# plt.legend()
# plt.show()

kf = VariantKalmanFilter(initial_state_mean=0, n_dim_obs=1,em_vars='all')
model = kf.em(measurements, offsets, n_iter=20)
l1 = model.loglikelihood(measurements, offsets)
print(model.transition_offsets)
print(model.transition_matrices)
print(model.transition_covariance)
print(model.observation_offsets)
print(model.observation_covariance)
kalman_filter = model.filter(measurements, offsets)[0]
smoothed_states = model.smooth(measurements, offsets)[0]

plt.plot(np.arange(T),kalman_filter, label='kalman_filter')
plt.plot(np.arange(T),smoothed_states, label='smoothed_states', color='red')
plt.plot(np.arange(T),states, label='states')
plt.legend()
plt.show()