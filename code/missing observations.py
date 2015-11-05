import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### initialization
T=100
#configure missing intervals
missing_intervals = [(25,40),(75,84)]

sigma2eta = 1
eta = pd.Series(np.random.randn(T)*np.sqrt(sigma2eta))
x = np.cumsum(eta)

sigma2eps = 10
eps = pd.Series(np.random.randn(T)*np.sqrt(sigma2eps))
y = x + eps

# missing data
for i in missing_intervals:
	x[i[0]:i[1]] = np.nan
	eta[i[0]:i[1]] = np.nan
	eps[i[0]:i[1]] = np.nan
	y[i[0]:i[1]] = np.nan

### Kalman filter
### we take innovation=0 and K=0 (gain of Kalman) for the missing intervals
# initialisation
xpred = pd.Series(np.zeros(T))
xpred[0] = np.random.randn()+4
sigma2pred = pd.Series(np.zeros(T))
sigma2pred[0] = (xpred[0]-x[0])*(xpred[0]-x[0])
P = pd.Series(np.zeros(T))
P[0] = (xpred[0]-x[0])*(xpred[0]-x[0])
F = pd.Series(np.zeros(T)+sigma2eps)
#innovation
I = pd.Series(np.zeros(T))
#gain of Kalman
K = pd.Series(np.zeros(T))
# recursion
for t in range(T-1):
	# Innovation and gain of Kalman
	if pd.isnull(y[t+1]):
		I[t+1]=0
		P[t+1] = P[t]*(1-K[t])+sigma2eta
		F[t+1] = P[t+1]+sigma2eps
		K[t+1]=0
		xpred[t+1] = xpred[t]
	else:
		I[t+1] = y[t+1] - xpred[t]
		P[t+1] = P[t]*(1-K[t])+sigma2eta
		F[t+1] = P[t+1]+sigma2eps
		K[t+1] = P[t+1]/F[t+1]
		xpred[t+1] = xpred[t] + K[t+1]*I[t+1]
	sigma2pred[t+1] = sigma2eps*P[t+1]/F[t+1]

#plot Kalman filter
x.plot(label='x')
xpred.plot(label='xpred')
plt.legend()
plt.show()

P.plot()
plt.show()

### state smoothing ###
r = pd.Series(np.zeros(T))
# smoothed state and its variance
xsmooth = pd.Series(np.zeros(T))
sigma2smooth = pd.Series(np.zeros(T))
N = pd.Series(np.zeros(T))

for t in range(T-1,0,-1):
	# compute xsmooth and its variance
	if pd.isnull(y[t]):
		r[t-1] = r[t]
		N[t-1] = N[t]
	else:
		r[t-1] = I[t]/F[t] + (1-K[t])*r[t]
		N[t-1] = 1/F[t] + np.square(1-K[t])*N[t]
	xsmooth[t] = xpred[t-1] + P[t]*r[t-1]
	sigma2smooth[t] = P[t] - np.square(P[t])*N[t-1]

#plot smoothed filter and variance
f, axarr = plt.subplots(2,sharex=True)
axarr[0].plot(xsmooth[1:T], label='xsmooth', color='red')
axarr[0].plot(x[1:T], label='x')
axarr[0].legend()
axarr[1].plot(sigma2smooth[1:T], label='sigma2smooth')
plt.show()


