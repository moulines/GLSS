import numpy as np
import matplotlib.pyplot as plt

T=100
sigma2eta = 0.7
eta = np.random.randn(T)*np.sqrt(sigma2eta)
x = np.cumsum(eta)

sigma2eps = 10
eps = np.random.randn(T)*np.sqrt(sigma2eps)
y = x + eps

plt.plot(np.arange(T),y, label='y')
plt.plot(np.arange(T),x, label='x')
plt.legend()
plt.show()

### Kalman filter ###
# initialisation
xpred = np.zeros(T)
xpred[0] = 7
sigma2pred = np.zeros(T)
sigma2pred[0] = (xpred[0]-x[0])*(xpred[0]-x[0])
P = np.zeros(T)
F = np.zeros(T)+sigma2eps
#innovation
I = np.zeros(T)
#gain of Kalman
K = np.zeros(T)
# recurrence
for t in range(T-1):
	# Innovation
	I[t+1] = y[t+1] - xpred[t]
	P[t+1] = sigma2pred[t]+sigma2eta
	F[t+1] = P[t+1]+sigma2eps
	# Kalman gain
	K[t+1] = P[t+1]/F[t+1]
	xpred[t+1] = xpred[t] + K[t+1]*I[t+1]
	sigma2pred[t+1] = sigma2eps*P[t+1]/F[t+1]
### plot x and xpred
plt.plot(np.arange(T),xpred, label='xpred', color='red')
plt.plot(np.arange(T),x, label='x')
plt.legend()
plt.show()

print (sigma2pred)
plt.plot(np.arange(T), sigma2pred)
plt.show()

### state smoothing ###
r = np.zeros(T)
# smoothed state and its variance
xsmooth = np.zeros(T)
sigma2smooth = np.zeros(T)
N = np.zeros(T)
# smoothed observation disturbance
disturbance = np.zeros(T)
u = np.zeros(T) # smooth error
u[T-1] = I[T-1]/F[T-1] - K[T-1]*r[T-1]
disturbance[T-1] = sigma2eps*u[T-1]
sigma2disturbance = np.zeros(T)
sigma2u = np.zeros(T)
sigma2u[T-1] = 1/F[T-1] + np.square(K[T-1])*N[T-1]
sigma2disturbance[T-1] = sigma2eps - np.square(sigma2eps)*sigma2u[T-1]

for t in range(T-1,0,-1):
	# compute xsmooth and its variance
	r[t-1] = I[t]/F[t] + (1-K[t])*r[t]
	N[t-1] = 1/F[t] + np.square(1-K[t])*N[t]
	xsmooth[t] = xpred[t-1] + P[t]*r[t-1]
	sigma2smooth[t] = P[t] - np.square(P[t])*N[t-1]
	# calculate smoothing error u, smoothed disturbance and its variance
	u[t-1] = I[t-1]/F[t-1] - K[t-1]*r[t-1]
	disturbance[t-1] = sigma2eps*u[t-1]
	sigma2u[t-1] = 1/F[t-1] + np.square(K[t-1])*N[t-1]
	sigma2disturbance[t-1] = sigma2eps - np.square(sigma2eps)*sigma2u[t-1]

#plot statement smoothing and variance
f, axarr = plt.subplots(2,sharex=True)
axarr[0].plot(np.arange(1,T),xsmooth[1:T], label='xsmooth', color='red')
axarr[0].plot(np.arange(1,T),x[1:T], label='x')
axarr[0].legend()

axarr[1].plot(np.arange(1,T),sigma2smooth[1:T], label='sigma2smooth')
plt.show()

#plot smoothed observation disturbance and obervation error variance
f, axarr = plt.subplots(2,sharex=True)
axarr[0].plot(np.arange(1,T),disturbance[1:T], label='smoothed obs disturbance', color='red')
axarr[0].plot(np.arange(1,T),np.zeros(T-1), label='x')
axarr[0].legend()

axarr[1].plot(np.arange(1,T),sigma2disturbance[1:T], label='observation error var')
plt.show()



