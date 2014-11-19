import matplotlib.pyplot as plt
import numpy as np

d = np.loadtxt('gnuplot.dat')
d2 = np.loadtxt('scaled.dat')

plt.plot(d[:,0], d[:,1], label="direct")
plt.plot(d[:,0], d[:,2], label="lut")
plt.plot(d[:,0], d[:,3], label="scaled lut")
plt.scatter(d2[:,0], d2[:,1],label="scaled points")


plt.legend()
plt.show()
